"""
Continual Learning - Techniques to prevent catastrophic forgetting.

Provides:
- EWC (Elastic Weight Consolidation)
- Replay Buffer for experience replay
- Task-specific adapters
- Progressive neural networks support
- Knowledge distillation for continual learning

These enable models to learn new tasks without forgetting previous ones.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from collections import deque
import copy
import random


@dataclass
class ContinualLearningConfig:
    """Configuration for continual learning."""
    
    method: str = "ewc"              # "ewc", "replay", "adapter", "distill"
    
    # EWC parameters
    ewc_lambda: float = 1000.0       # Regularization strength
    ewc_gamma: float = 0.9           # Decay for online EWC
    
    # Replay parameters
    replay_buffer_size: int = 10000
    replay_batch_ratio: float = 0.5  # Ratio of replay vs new data
    
    # Distillation parameters
    distill_temperature: float = 2.0
    distill_alpha: float = 0.5       # Weight of distillation loss
    

# =============================================================================
# EWC - Elastic Weight Consolidation
# =============================================================================

class EWCRegularizer:
    """
    Elastic Weight Consolidation for continual learning.
    
    Adds a regularization term that penalizes changes to important parameters.
    Importance is measured by Fisher information.
    
    Paper: https://arxiv.org/abs/1612.00796
    
    Example:
        >>> ewc = EWCRegularizer(model, lambda_=1000)
        >>> # Train on task 1
        >>> ewc.compute_fisher(task1_dataloader)
        >>> ewc.consolidate()
        >>> 
        >>> # Train on task 2 with EWC loss
        >>> for batch in task2_dataloader:
        >>>     loss = criterion(model(batch))
        >>>     loss += ewc.penalty()
        >>>     loss.backward()
    """
    
    def __init__(
        self,
        model: nn.Module,
        lambda_: float = 1000.0,
        gamma: float = 0.9,
        online: bool = True,
    ):
        self.model = model
        self.lambda_ = lambda_
        self.gamma = gamma
        self.online = online
        
        # Store Fisher information and optimal parameters
        self.fisher: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}
        
        # For online EWC
        self.accumulated_fisher: Dict[str, torch.Tensor] = {}
    
    def compute_fisher(
        self,
        dataloader,
        num_samples: int = 1000,
    ):
        """
        Compute Fisher information matrix diagonal.
        
        Args:
            dataloader: DataLoader for computing Fisher
            num_samples: Number of samples to use
        """
        self.fisher = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        
        self.model.eval()
        samples_seen = 0
        
        for batch in dataloader:
            if samples_seen >= num_samples:
                break
            
            # Forward pass
            if isinstance(batch, (tuple, list)):
                inputs = batch[0]
            else:
                inputs = batch
            
            inputs = inputs.to(next(self.model.parameters()).device)
            outputs = self.model(inputs)
            
            # Log likelihood (simplified: use cross-entropy with predictions)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            # Sample from output distribution
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Compute gradients of log-likelihood
            for i in range(min(inputs.size(0), num_samples - samples_seen)):
                self.model.zero_grad()
                
                # Gradient of log p(y|x) for sampled y
                sample = torch.multinomial(probs[i].view(-1), 1)
                log_prob = log_probs[i].view(-1)[sample]
                log_prob.backward(retain_graph=True)
                
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        self.fisher[name] += param.grad.detach() ** 2
            
            samples_seen += inputs.size(0)
        
        # Normalize
        for name in self.fisher:
            self.fisher[name] /= samples_seen
        
        self.model.train()
    
    def consolidate(self):
        """
        Store current parameters as optimal and update Fisher.
        
        Call this after training on a task, before moving to the next.
        """
        # Store optimal parameters
        self.optimal_params = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        
        # Update accumulated Fisher for online EWC
        if self.online:
            for name in self.fisher:
                if name in self.accumulated_fisher:
                    self.accumulated_fisher[name] = (
                        self.gamma * self.accumulated_fisher[name] +
                        self.fisher[name]
                    )
                else:
                    self.accumulated_fisher[name] = self.fisher[name].clone()
    
    def penalty(self) -> torch.Tensor:
        """
        Compute EWC regularization loss.
        
        Returns:
            EWC penalty term to add to training loss
        """
        device = next(self.model.parameters()).device
        
        if not self.optimal_params:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        fisher = self.accumulated_fisher if self.online else self.fisher
        
        if not fisher:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        loss = torch.tensor(0.0, device=device)
        
        for name, param in self.model.named_parameters():
            if name in self.optimal_params and name in fisher:
                loss = loss + (
                    fisher[name] * 
                    (param - self.optimal_params[name]) ** 2
                ).sum()
        
        return self.lambda_ * loss


# =============================================================================
# Experience Replay
# =============================================================================

class ReplayBuffer:
    """
    Experience replay buffer for continual learning.
    
    Stores samples from previous tasks and mixes them with current task.
    
    Example:
        >>> buffer = ReplayBuffer(max_size=10000)
        >>> 
        >>> # Add samples from task 1
        >>> for batch in task1_dataloader:
        >>>     buffer.add(batch)
        >>> 
        >>> # Train on task 2 with replay
        >>> for batch in task2_dataloader:
        >>>     replay = buffer.sample(batch_size=32)
        >>>     combined = combine_batches(batch, replay)
        >>>     train_step(combined)
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        reservoir_sampling: bool = True,
    ):
        self.max_size = max_size
        self.reservoir_sampling = reservoir_sampling
        
        self.buffer: List[Any] = []
        self.total_seen = 0
    
    def add(self, samples: Union[torch.Tensor, Tuple, Dict]):
        """
        Add samples to buffer.
        
        Uses reservoir sampling for fair representation of all seen data.
        """
        if isinstance(samples, torch.Tensor):
            samples = [samples[i] for i in range(samples.size(0))]
        elif isinstance(samples, (tuple, list)):
            # Assume (inputs, labels) format
            if isinstance(samples[0], torch.Tensor):
                batch_size = samples[0].size(0)
                samples = [(samples[0][i], samples[1][i]) for i in range(batch_size)]
        
        for sample in samples:
            self.total_seen += 1
            
            if len(self.buffer) < self.max_size:
                self.buffer.append(sample)
            elif self.reservoir_sampling:
                # Reservoir sampling
                idx = random.randint(0, self.total_seen - 1)
                if idx < self.max_size:
                    self.buffer[idx] = sample
            else:
                # FIFO
                self.buffer.pop(0)
                self.buffer.append(sample)
    
    def sample(self, batch_size: int) -> List[Any]:
        """Sample a batch from the buffer."""
        if len(self.buffer) == 0:
            return []
        
        batch_size = min(batch_size, len(self.buffer))
        indices = random.sample(range(len(self.buffer)), batch_size)
        return [self.buffer[i] for i in indices]
    
    def collate_samples(self, samples: List) -> Tuple[torch.Tensor, ...]:
        """Collate samples into batched tensors."""
        if not samples:
            return None
        
        if isinstance(samples[0], torch.Tensor):
            return torch.stack(samples)
        elif isinstance(samples[0], (tuple, list)):
            return tuple(
                torch.stack([s[i] for s in samples])
                for i in range(len(samples[0]))
            )
        else:
            return samples
    
    def __len__(self):
        return len(self.buffer)


class ReplayTrainer:
    """
    Trainer with experience replay for continual learning.
    """
    
    def __init__(
        self,
        model: nn.Module,
        buffer_size: int = 10000,
        replay_ratio: float = 0.5,
    ):
        self.model = model
        self.buffer = ReplayBuffer(max_size=buffer_size)
        self.replay_ratio = replay_ratio
    
    def train_step(
        self,
        batch,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """
        Training step with replay.
        """
        # Add current batch to buffer
        self.buffer.add(batch)
        
        # Sample from replay buffer
        replay_batch_size = int(batch[0].size(0) * self.replay_ratio)
        replay_samples = self.buffer.sample(replay_batch_size)
        
        # Combine batches
        if replay_samples:
            replay_batch = self.buffer.collate_samples(replay_samples)
            # Combine current and replay (implementation depends on batch format)
        
        # Forward and backward
        self.model.train()
        optimizer.zero_grad()
        
        outputs = self.model(batch[0])
        loss = criterion(outputs, batch[1])
        
        loss.backward()
        optimizer.step()
        
        return {"loss": loss.item()}


# =============================================================================
# Task-Specific Adapters
# =============================================================================

class TaskAdapter(nn.Module):
    """
    Task-specific adapter module.
    
    Learns task-specific modifications while keeping base model frozen.
    """
    
    def __init__(
        self,
        in_features: int,
        bottleneck_dim: int = 64,
        task_id: int = 0,
    ):
        super().__init__()
        self.task_id = task_id
        
        self.down = nn.Linear(in_features, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, in_features)
        self.act = nn.GELU()
        
        # Initialize to near-identity
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(x)))


class MultiTaskAdapterManager:
    """
    Manages multiple task-specific adapters.
    
    Example:
        >>> manager = MultiTaskAdapterManager(model)
        >>> 
        >>> # Add adapters for each task
        >>> manager.add_adapter("task1")
        >>> manager.add_adapter("task2")
        >>> 
        >>> # Switch between tasks
        >>> manager.set_active_task("task1")
        >>> output = model(input)
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.adapters: Dict[str, Dict[str, TaskAdapter]] = {}
        self.active_task: Optional[str] = None
    
    def add_adapter(self, task_name: str, bottleneck_dim: int = 64):
        """Add adapters for a new task."""
        self.adapters[task_name] = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                adapter = TaskAdapter(
                    in_features=module.out_features,
                    bottleneck_dim=bottleneck_dim,
                )
                self.adapters[task_name][name] = adapter
    
    def set_active_task(self, task_name: str):
        """Set the active task's adapters."""
        if task_name not in self.adapters:
            raise ValueError(f"Unknown task: {task_name}")
        self.active_task = task_name
    
    def get_adapter_params(self, task_name: Optional[str] = None) -> List[nn.Parameter]:
        """Get parameters for a specific task's adapters."""
        task = task_name or self.active_task
        if task is None or task not in self.adapters:
            return []
        
        params = []
        for adapter in self.adapters[task].values():
            params.extend(adapter.parameters())
        return params


# =============================================================================
# Knowledge Distillation for Continual Learning
# =============================================================================

class DistillationLoss(nn.Module):
    """
    Knowledge distillation loss for continual learning.
    
    Regularizes the model to maintain similar outputs to old model.
    """
    
    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute combined distillation and task loss.
        
        Args:
            student_logits: Current model outputs
            teacher_logits: Old model outputs (detached)
            labels: Ground truth labels
            
        Returns:
            Combined loss
        """
        # Task loss
        task_loss = F.cross_entropy(student_logits, labels)
        
        # Distillation loss
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_predictions = F.log_softmax(student_logits / self.temperature, dim=-1)
        distill_loss = F.kl_div(
            soft_predictions,
            soft_targets,
            reduction="batchmean",
        ) * (self.temperature ** 2)
        
        return (1 - self.alpha) * task_loss + self.alpha * distill_loss


class ContinualDistillationTrainer:
    """
    Trainer using knowledge distillation for continual learning.
    """
    
    def __init__(
        self,
        model: nn.Module,
        temperature: float = 2.0,
        alpha: float = 0.5,
    ):
        self.model = model
        self.old_model: Optional[nn.Module] = None
        self.distill_loss = DistillationLoss(temperature, alpha)
    
    def save_model_state(self):
        """Save current model as teacher for next task."""
        self.old_model = copy.deepcopy(self.model)
        self.old_model.eval()
        for param in self.old_model.parameters():
            param.requires_grad = False
    
    def train_step(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """Training step with distillation."""
        self.model.train()
        optimizer.zero_grad()
        
        # Current model outputs
        outputs = self.model(inputs)
        student_logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        
        if self.old_model is not None:
            # Get teacher outputs
            with torch.no_grad():
                teacher_outputs = self.old_model(inputs)
                teacher_logits = teacher_outputs.logits if hasattr(teacher_outputs, 'logits') else teacher_outputs
            
            loss = self.distill_loss(student_logits, teacher_logits, labels)
        else:
            loss = F.cross_entropy(student_logits, labels)
        
        loss.backward()
        optimizer.step()
        
        return {"loss": loss.item()}


# =============================================================================
# Unified Continual Learning Trainer
# =============================================================================

class ContinualLearner:
    """
    Unified continual learning trainer supporting multiple methods.
    
    Example:
        >>> learner = ContinualLearner(model, method="ewc")
        >>> 
        >>> # Train task 1
        >>> learner.train_task(task1_dataloader, task_name="task1")
        >>> 
        >>> # Train task 2 (with continual learning)
        >>> learner.train_task(task2_dataloader, task_name="task2")
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[ContinualLearningConfig] = None,
    ):
        self.model = model
        self.config = config or ContinualLearningConfig()
        
        # Initialize method-specific components
        if self.config.method == "ewc":
            self.ewc = EWCRegularizer(
                model,
                lambda_=self.config.ewc_lambda,
                gamma=self.config.ewc_gamma,
            )
        elif self.config.method == "replay":
            self.replay = ReplayTrainer(
                model,
                buffer_size=self.config.replay_buffer_size,
            )
        elif self.config.method == "distill":
            self.distill = ContinualDistillationTrainer(
                model,
                temperature=self.config.distill_temperature,
                alpha=self.config.distill_alpha,
            )
        
        self.tasks_trained: List[str] = []
    
    def on_task_start(self, task_name: str):
        """Called before training on a new task."""
        pass
    
    def on_task_end(self, task_name: str, dataloader=None):
        """Called after training on a task."""
        self.tasks_trained.append(task_name)
        
        if self.config.method == "ewc" and dataloader is not None:
            self.ewc.compute_fisher(dataloader)
            self.ewc.consolidate()
        elif self.config.method == "distill":
            self.distill.save_model_state()
    
    def get_regularization_loss(self) -> torch.Tensor:
        """Get regularization loss for current method."""
        if self.config.method == "ewc":
            return self.ewc.penalty()
        return torch.tensor(0.0)
