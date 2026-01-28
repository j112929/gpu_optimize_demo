"""
Alignment Training - RLHF and DPO for model alignment.

Provides:
- Reward Model training
- PPO (Proximal Policy Optimization) for RLHF
- DPO (Direct Preference Optimization)
- KTO (Kahneman-Tversky Optimization)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import math


@dataclass
class RewardConfig:
    """Configuration for Reward Model."""
    base_model_path: str = ""
    hidden_size: int = 768
    
    # Training
    learning_rate: float = 1e-5
    max_length: int = 512
    
    # Loss
    loss_type: str = "ranking"    # ranking, regression


class RewardModel(nn.Module):
    """
    Reward Model for RLHF.
    
    Learns to score responses based on human preferences.
    Higher scores = better responses.
    
    Example:
        >>> reward_model = RewardModel(base_model, config)
        >>> score = reward_model(input_ids, attention_mask)
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        config: Optional[RewardConfig] = None,
    ):
        super().__init__()
        
        self.config = config or RewardConfig()
        self.base_model = base_model
        
        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Reward head
        self.reward_head = nn.Linear(self.config.hidden_size, 1)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute reward score.
        
        Returns:
            Reward scores [batch_size]
        """
        # Get hidden states from base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        
        # Use last hidden state at last token
        hidden = outputs.hidden_states[-1]
        
        # Get last non-padding token
        if attention_mask is not None:
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(hidden.size(0), device=hidden.device)
            last_hidden = hidden[batch_indices, seq_lengths]
        else:
            last_hidden = hidden[:, -1]
        
        # Compute reward
        reward = self.reward_head(last_hidden).squeeze(-1)
        
        return reward
    
    @staticmethod
    def ranking_loss(
        chosen_rewards: torch.Tensor,
        rejected_rewards: torch.Tensor,
        margin: float = 0.0,
    ) -> torch.Tensor:
        """
        Compute ranking loss for preference pairs.
        
        Loss = -log(sigmoid(chosen - rejected))
        """
        return -F.logsigmoid(chosen_rewards - rejected_rewards - margin).mean()


# =============================================================================
# PPO for RLHF
# =============================================================================

@dataclass
class PPOConfig:
    """Configuration for PPO training."""
    # PPO hyperparameters
    gamma: float = 1.0            # Discount factor
    lam: float = 0.95             # GAE lambda
    cliprange: float = 0.2        # PPO clip range
    value_cliprange: float = 0.2
    
    # KL penalty
    init_kl_coef: float = 0.2
    target_kl: float = 6.0
    kl_penalty: str = "kl"        # kl, abs, mse
    
    # Training
    ppo_epochs: int = 4
    minibatch_size: int = 4
    learning_rate: float = 1e-5
    
    # Generation
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_k: int = 50


class PPOTrainer:
    """
    PPO Trainer for RLHF.
    
    Trains a policy model using rewards from human feedback.
    
    Example:
        >>> trainer = PPOTrainer(policy_model, ref_model, reward_model, config)
        >>> 
        >>> for batch in dataloader:
        ...     stats = trainer.step(batch["input_ids"])
    """
    
    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        reward_model: RewardModel,
        config: Optional[PPOConfig] = None,
    ):
        self.policy = policy_model
        self.ref = ref_model
        self.reward = reward_model
        self.config = config or PPOConfig()
        
        # Freeze reference model
        for param in self.ref.parameters():
            param.requires_grad = False
        
        # KL coefficient
        self.kl_coef = self.config.init_kl_coef
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.config.learning_rate,
        )
    
    def generate(
        self,
        input_ids: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate responses from policy."""
        self.policy.eval()
        
        with torch.no_grad():
            outputs = self.policy.generate(
                input_ids,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
            )
        
        return outputs.sequences, outputs.scores
    
    def compute_rewards(
        self,
        input_ids: torch.Tensor,
        response_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute rewards and KL penalty."""
        # Get reward
        full_ids = torch.cat([input_ids, response_ids], dim=1)
        reward = self.reward(full_ids, attention_mask)
        
        # Compute KL divergence
        with torch.no_grad():
            policy_logits = self.policy(response_ids).logits
            ref_logits = self.ref(response_ids).logits
            
            policy_log_probs = F.log_softmax(policy_logits, dim=-1)
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            
            kl = (policy_log_probs.exp() * (policy_log_probs - ref_log_probs)).sum(-1).mean()
        
        # Apply KL penalty
        reward_with_kl = reward - self.kl_coef * kl
        
        return reward_with_kl, kl
    
    def step(self, input_ids: torch.Tensor) -> Dict[str, float]:
        """
        Perform one PPO step.
        
        1. Generate responses
        2. Compute rewards
        3. Compute advantages
        4. Update policy
        """
        # Generate
        response_ids, _ = self.generate(input_ids)
        
        # Compute rewards
        rewards, kl = self.compute_rewards(input_ids, response_ids)
        
        # PPO update (simplified)
        self.policy.train()
        
        for _ in range(self.config.ppo_epochs):
            # Get current log probs
            logits = self.policy(response_ids).logits
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Compute loss (simplified)
            loss = -log_probs.mean() * rewards.mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # Update KL coefficient
        if kl > self.config.target_kl * 1.5:
            self.kl_coef *= 1.5
        elif kl < self.config.target_kl / 1.5:
            self.kl_coef /= 1.5
        
        return {
            "reward": rewards.mean().item(),
            "kl": kl.item(),
            "kl_coef": self.kl_coef,
            "loss": loss.item(),
        }


RLHFTrainer = PPOTrainer  # Alias


# =============================================================================
# DPO (Direct Preference Optimization)
# =============================================================================

@dataclass
class DPOConfig:
    """Configuration for DPO training."""
    # DPO hyperparameters
    beta: float = 0.1             # Temperature for DPO loss
    
    # Training
    learning_rate: float = 1e-6
    max_length: int = 512
    
    # Label smoothing
    label_smoothing: float = 0.0
    
    # Reference model
    ref_model_path: Optional[str] = None


class DPOTrainer:
    """
    DPO (Direct Preference Optimization) Trainer.
    
    Directly optimizes for human preferences without
    training a separate reward model.
    
    More stable and efficient than RLHF.
    
    Example:
        >>> config = DPOConfig(beta=0.1)
        >>> trainer = DPOTrainer(model, ref_model, config)
        >>> 
        >>> loss = trainer.step(
        ...     input_ids, chosen_ids, rejected_ids
        ... )
    """
    
    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        config: Optional[DPOConfig] = None,
    ):
        self.model = model
        self.ref_model = ref_model
        self.config = config or DPOConfig()
        
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )
    
    def compute_log_probs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute log probabilities of sequences."""
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1]  # Shift for next token pred
        labels = input_ids[:, 1:]
        
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Gather log probs for actual tokens
        token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
        
        # Mask padding
        if attention_mask is not None:
            token_log_probs = token_log_probs * attention_mask[:, 1:]
        
        # Sum log probs per sequence
        return token_log_probs.sum(dim=-1)
    
    def dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute DPO loss.
        
        L_DPO = -log(sigmoid(beta * (log(pi(y_w)/pi_ref(y_w)) - log(pi(y_l)/pi_ref(y_l)))))
        """
        # Log ratios
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps
        
        # DPO loss
        logits = self.config.beta * (chosen_logratios - rejected_logratios)
        
        if self.config.label_smoothing > 0:
            # With label smoothing
            losses = (
                -F.logsigmoid(logits) * (1 - self.config.label_smoothing) +
                -F.logsigmoid(-logits) * self.config.label_smoothing
            )
        else:
            losses = -F.logsigmoid(logits)
        
        return losses.mean()
    
    def step(
        self,
        prompt_ids: torch.Tensor,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
        prompt_mask: Optional[torch.Tensor] = None,
        chosen_mask: Optional[torch.Tensor] = None,
        rejected_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Perform one DPO step.
        
        Args:
            prompt_ids: Prompt token IDs
            chosen_ids: Chosen (preferred) response IDs
            rejected_ids: Rejected response IDs
            
        Returns:
            Dict with loss and metrics
        """
        self.model.train()
        
        # Compute log probs for policy
        policy_chosen_logps = self.compute_log_probs(
            self.model, chosen_ids, chosen_mask
        )
        policy_rejected_logps = self.compute_log_probs(
            self.model, rejected_ids, rejected_mask
        )
        
        # Compute log probs for reference (no grad)
        with torch.no_grad():
            ref_chosen_logps = self.compute_log_probs(
                self.ref_model, chosen_ids, chosen_mask
            )
            ref_rejected_logps = self.compute_log_probs(
                self.ref_model, rejected_ids, rejected_mask
            )
        
        # Compute loss
        loss = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
        )
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Compute accuracy (chosen > rejected)
        with torch.no_grad():
            chosen_rewards = self.config.beta * (policy_chosen_logps - ref_chosen_logps)
            rejected_rewards = self.config.beta * (policy_rejected_logps - ref_rejected_logps)
            accuracy = (chosen_rewards > rejected_rewards).float().mean()
        
        return {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "chosen_reward": chosen_rewards.mean().item(),
            "rejected_reward": rejected_rewards.mean().item(),
        }


# =============================================================================
# KTO (Kahneman-Tversky Optimization)
# =============================================================================

@dataclass
class KTOConfig:
    """Configuration for KTO."""
    beta: float = 0.1
    desirable_weight: float = 1.0
    undesirable_weight: float = 1.0


class KTOTrainer:
    """
    KTO (Kahneman-Tversky Optimization) Trainer.
    
    Alternative to DPO that can work with non-paired preferences
    (just good/bad labels, not comparisons).
    """
    
    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        config: Optional[KTOConfig] = None,
    ):
        self.model = model
        self.ref_model = ref_model
        self.config = config or KTOConfig()
        
        for param in self.ref_model.parameters():
            param.requires_grad = False
    
    def kto_loss(
        self,
        policy_logps: torch.Tensor,
        ref_logps: torch.Tensor,
        is_desirable: torch.Tensor,  # Boolean mask
    ) -> torch.Tensor:
        """Compute KTO loss."""
        logratios = policy_logps - ref_logps
        kl = (policy_logps.exp() * logratios).mean()
        
        # Separate desirable/undesirable
        desirable_mask = is_desirable.float()
        undesirable_mask = 1 - desirable_mask
        
        # Losses (asymmetric treatment)
        desirable_loss = -F.logsigmoid(self.config.beta * logratios)
        undesirable_loss = -F.logsigmoid(-self.config.beta * logratios)
        
        loss = (
            self.config.desirable_weight * (desirable_mask * desirable_loss).mean() +
            self.config.undesirable_weight * (undesirable_mask * undesirable_loss).mean()
        )
        
        return loss
