"""
Advanced Preference Optimization Methods.

Provides:
- ORPO (Odds Ratio Preference Optimization) - No reference model needed
- SimPO (Simple Preference Optimization) - Length-normalized, no reference
- IPO (Identity Preference Optimization) - More robust to noise

These are improvements over DPO with simpler training requirements.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union


# =============================================================================
# ORPO - Odds Ratio Preference Optimization
# =============================================================================

@dataclass
class ORPOConfig:
    """Configuration for ORPO training."""
    
    lambda_orpo: float = 0.1        # Weight for odds ratio loss
    learning_rate: float = 1e-5
    max_length: int = 512
    beta: float = 0.1               # Scaling factor
    

class ORPOTrainer:
    """
    ORPO (Odds Ratio Preference Optimization) Trainer.
    
    Combines SFT and preference optimization in a single objective.
    No reference model needed, making training simpler.
    
    Loss = SFT_loss - lambda * log(odds_ratio)
    
    Paper: https://arxiv.org/abs/2403.07691
    
    Example:
        >>> trainer = ORPOTrainer(model, config=ORPOConfig(lambda_orpo=0.1))
        >>> for batch in preference_data:
        >>>     stats = trainer.step(
        >>>         input_ids=batch["input_ids"],
        >>>         chosen_ids=batch["chosen"],
        >>>         rejected_ids=batch["rejected"],
        >>>     )
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[ORPOConfig] = None,
    ):
        self.model = model
        self.config = config or ORPOConfig()
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
        )
    
    def compute_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log probabilities and per-token logits.
        
        Returns:
            (log_probs, per_token_logps)
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        
        # Shift for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        # Per-token log probs
        log_probs = F.log_softmax(shift_logits, dim=-1)
        per_token_logps = torch.gather(
            log_probs, 
            dim=-1, 
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Sum over sequence
        if attention_mask is not None:
            mask = attention_mask[..., 1:].contiguous()
            per_token_logps = per_token_logps * mask
            sequence_logps = per_token_logps.sum(dim=-1)
        else:
            sequence_logps = per_token_logps.sum(dim=-1)
        
        return sequence_logps, per_token_logps
    
    def compute_odds_ratio(
        self,
        chosen_logps: torch.Tensor,
        rejected_logps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute odds ratio: P(chosen) / P(rejected) in log space.
        
        log_odds = log(P(chosen) / (1 - P(chosen))) - log(P(rejected) / (1 - P(rejected)))
        """
        # Convert log probs to probs (with clamping for stability)
        chosen_probs = torch.clamp(torch.exp(chosen_logps), 1e-10, 1 - 1e-10)
        rejected_probs = torch.clamp(torch.exp(rejected_logps), 1e-10, 1 - 1e-10)
        
        # Log odds
        chosen_log_odds = torch.log(chosen_probs / (1 - chosen_probs))
        rejected_log_odds = torch.log(rejected_probs / (1 - rejected_probs))
        
        return chosen_log_odds - rejected_log_odds
    
    def orpo_loss(
        self,
        chosen_logps: torch.Tensor,
        rejected_logps: torch.Tensor,
        chosen_per_token: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute ORPO loss.
        
        L = L_SFT - lambda * log(sigmoid(log_odds_ratio))
        
        Returns:
            (loss, metrics_dict)
        """
        # SFT loss (negative log likelihood of chosen)
        sft_loss = -chosen_per_token.mean()
        
        # Odds ratio loss
        log_odds = self.compute_odds_ratio(chosen_logps, rejected_logps)
        odds_ratio_loss = -F.logsigmoid(log_odds).mean()
        
        # Combined loss
        loss = sft_loss + self.config.lambda_orpo * odds_ratio_loss
        
        # Metrics
        with torch.no_grad():
            accuracy = (chosen_logps > rejected_logps).float().mean()
        
        metrics = {
            "loss": loss.item(),
            "sft_loss": sft_loss.item(),
            "odds_ratio_loss": odds_ratio_loss.item(),
            "accuracy": accuracy.item(),
            "chosen_logps": chosen_logps.mean().item(),
            "rejected_logps": rejected_logps.mean().item(),
        }
        
        return loss, metrics
    
    def step(
        self,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
        chosen_mask: Optional[torch.Tensor] = None,
        rejected_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Perform one ORPO training step.
        
        Args:
            chosen_ids: Token IDs for chosen responses
            rejected_ids: Token IDs for rejected responses
            chosen_mask: Attention mask for chosen
            rejected_mask: Attention mask for rejected
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Compute log probs
        chosen_logps, chosen_per_token = self.compute_log_probs(
            chosen_ids, chosen_mask
        )
        rejected_logps, _ = self.compute_log_probs(
            rejected_ids, rejected_mask
        )
        
        # Compute loss
        loss, metrics = self.orpo_loss(
            chosen_logps, rejected_logps, chosen_per_token
        )
        
        # Backward
        loss.backward()
        self.optimizer.step()
        
        return metrics


# =============================================================================
# SimPO - Simple Preference Optimization
# =============================================================================

@dataclass
class SimPOConfig:
    """Configuration for SimPO training."""
    
    beta: float = 2.0               # Scaling factor (higher than DPO)
    gamma: float = 0.5              # Target reward margin
    learning_rate: float = 1e-6
    max_length: int = 512
    length_normalization: bool = True  # Key feature of SimPO
    

class SimPOTrainer:
    """
    SimPO (Simple Preference Optimization) Trainer.
    
    Key differences from DPO:
    1. No reference model needed
    2. Uses length-normalized log probabilities
    3. Target reward margin (gamma) for better separation
    
    Paper: https://arxiv.org/abs/2405.14734
    
    Example:
        >>> trainer = SimPOTrainer(model, config=SimPOConfig(beta=2.0, gamma=0.5))
        >>> for batch in preference_data:
        >>>     stats = trainer.step(
        >>>         chosen_ids=batch["chosen"],
        >>>         rejected_ids=batch["rejected"],
        >>>     )
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[SimPOConfig] = None,
    ):
        self.model = model
        self.config = config or SimPOConfig()
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
        )
    
    def compute_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Compute log probabilities and sequence length.
        
        Returns:
            (log_probs, length)
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        
        # Shift for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        # Per-token log probs
        log_probs = F.log_softmax(shift_logits, dim=-1)
        per_token_logps = torch.gather(
            log_probs,
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Handle mask
        if attention_mask is not None:
            mask = attention_mask[..., 1:].contiguous()
            per_token_logps = per_token_logps * mask
            length = mask.sum(dim=-1)
        else:
            length = torch.tensor(per_token_logps.size(-1), device=per_token_logps.device)
        
        sequence_logps = per_token_logps.sum(dim=-1)
        
        return sequence_logps, length
    
    def simpo_loss(
        self,
        chosen_logps: torch.Tensor,
        rejected_logps: torch.Tensor,
        chosen_length: torch.Tensor,
        rejected_length: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute SimPO loss.
        
        L = -log(sigmoid(beta * (r_chosen - r_rejected - gamma)))
        
        Where r = log_prob / length (length normalized reward)
        
        Returns:
            (loss, metrics_dict)
        """
        # Length normalization (key feature of SimPO)
        if self.config.length_normalization:
            chosen_rewards = chosen_logps / chosen_length.float()
            rejected_rewards = rejected_logps / rejected_length.float()
        else:
            chosen_rewards = chosen_logps
            rejected_rewards = rejected_logps
        
        # SimPO loss with margin
        reward_diff = chosen_rewards - rejected_rewards - self.config.gamma
        loss = -F.logsigmoid(self.config.beta * reward_diff).mean()
        
        # Metrics
        with torch.no_grad():
            accuracy = (chosen_rewards > rejected_rewards).float().mean()
            reward_margin = (chosen_rewards - rejected_rewards).mean()
        
        metrics = {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "reward_margin": reward_margin.item(),
            "chosen_rewards": chosen_rewards.mean().item(),
            "rejected_rewards": rejected_rewards.mean().item(),
        }
        
        return loss, metrics
    
    def step(
        self,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
        chosen_mask: Optional[torch.Tensor] = None,
        rejected_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Perform one SimPO training step.
        
        Args:
            chosen_ids: Token IDs for chosen responses
            rejected_ids: Token IDs for rejected responses
            chosen_mask: Attention mask for chosen
            rejected_mask: Attention mask for rejected
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Compute log probs with lengths
        chosen_logps, chosen_length = self.compute_log_probs(
            chosen_ids, chosen_mask
        )
        rejected_logps, rejected_length = self.compute_log_probs(
            rejected_ids, rejected_mask
        )
        
        # Compute loss
        loss, metrics = self.simpo_loss(
            chosen_logps, rejected_logps,
            chosen_length, rejected_length,
        )
        
        # Backward
        loss.backward()
        self.optimizer.step()
        
        return metrics


# =============================================================================
# IPO - Identity Preference Optimization
# =============================================================================

@dataclass
class IPOConfig:
    """Configuration for IPO training."""
    
    tau: float = 0.1                # Regularization parameter
    learning_rate: float = 1e-6
    max_length: int = 512
    

class IPOTrainer:
    """
    IPO (Identity Preference Optimization) Trainer.
    
    More robust to noise in preference data than DPO.
    Uses a different loss formulation that doesn't overfit to noise.
    
    Paper: https://arxiv.org/abs/2310.12036
    
    Example:
        >>> trainer = IPOTrainer(model, ref_model, config=IPOConfig(tau=0.1))
        >>> stats = trainer.step(chosen_ids, rejected_ids)
    """
    
    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        config: Optional[IPOConfig] = None,
    ):
        self.model = model
        self.ref_model = ref_model
        self.config = config or IPOConfig()
        
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
        )
    
    def compute_log_probs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute sequence log probabilities."""
        with torch.set_grad_enabled(model.training):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        log_probs = F.log_softmax(shift_logits, dim=-1)
        per_token_logps = torch.gather(
            log_probs,
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        if attention_mask is not None:
            mask = attention_mask[..., 1:].contiguous()
            per_token_logps = per_token_logps * mask
        
        return per_token_logps.sum(dim=-1)
    
    def ipo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute IPO loss.
        
        L = (log_ratio_diff - 1/(2*tau))^2
        
        Where log_ratio_diff = (log(pi/pi_ref)(y_w) - log(pi/pi_ref)(y_l))
        
        Returns:
            (loss, metrics_dict)
        """
        # Log ratios
        chosen_log_ratio = policy_chosen_logps - ref_chosen_logps
        rejected_log_ratio = policy_rejected_logps - ref_rejected_logps
        
        # IPO loss (squared difference from target)
        target = 1.0 / (2.0 * self.config.tau)
        log_ratio_diff = chosen_log_ratio - rejected_log_ratio
        loss = ((log_ratio_diff - target) ** 2).mean()
        
        # Metrics
        with torch.no_grad():
            accuracy = (chosen_log_ratio > rejected_log_ratio).float().mean()
        
        metrics = {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "chosen_log_ratio": chosen_log_ratio.mean().item(),
            "rejected_log_ratio": rejected_log_ratio.mean().item(),
        }
        
        return loss, metrics
    
    def step(
        self,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
        chosen_mask: Optional[torch.Tensor] = None,
        rejected_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Perform one IPO training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Policy log probs
        policy_chosen_logps = self.compute_log_probs(
            self.model, chosen_ids, chosen_mask
        )
        policy_rejected_logps = self.compute_log_probs(
            self.model, rejected_ids, rejected_mask
        )
        
        # Reference log probs
        with torch.no_grad():
            ref_chosen_logps = self.compute_log_probs(
                self.ref_model, chosen_ids, chosen_mask
            )
            ref_rejected_logps = self.compute_log_probs(
                self.ref_model, rejected_ids, rejected_mask
            )
        
        # Compute loss
        loss, metrics = self.ipo_loss(
            policy_chosen_logps, policy_rejected_logps,
            ref_chosen_logps, ref_rejected_logps,
        )
        
        # Backward
        loss.backward()
        self.optimizer.step()
        
        return metrics


# =============================================================================
# Utilities
# =============================================================================

def select_preference_method(
    reference_model_available: bool = True,
    data_quality: str = "high",  # "high", "medium", "low"
    memory_constrained: bool = False,
) -> str:
    """
    Recommend a preference optimization method based on constraints.
    
    Args:
        reference_model_available: Whether a reference model is available
        data_quality: Quality of preference data
        memory_constrained: Whether memory is a concern
        
    Returns:
        Recommended method: "dpo", "orpo", "simpo", or "ipo"
    """
    if not reference_model_available or memory_constrained:
        if data_quality == "high":
            return "simpo"  # Best for high-quality data, no ref needed
        else:
            return "orpo"   # More robust, no ref needed
    else:
        if data_quality == "low":
            return "ipo"    # Most robust to noise
        else:
            return "dpo"    # Standard choice with ref model
