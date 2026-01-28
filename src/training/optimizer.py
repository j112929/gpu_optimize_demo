"""
Optimizer Utilities - Optimized optimizer creation and scheduling.

Provides:
- Fused optimizers for faster training
- Learning rate schedules
- Parameter group utilities
"""

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    LambdaLR,
)
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
import math


@dataclass
class OptimizerConfig:
    """Configuration for optimizer."""
    # Optimizer type
    optimizer: str = "adamw"  # adam, adamw, sgd, adafactor
    
    # Learning rate
    lr: float = 1e-4
    weight_decay: float = 0.01
    
    # Adam specific
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    
    # SGD specific
    momentum: float = 0.9
    nesterov: bool = True
    
    # Fused operations
    fused: bool = True  # Use fused kernels if available
    
    # Parameter groups
    no_decay_keywords: List[str] = None
    
    def __post_init__(self):
        if self.no_decay_keywords is None:
            self.no_decay_keywords = ["bias", "LayerNorm", "layernorm", "ln"]


def create_optimizer(
    model: nn.Module,
    config: Optional[OptimizerConfig] = None,
    lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    **kwargs,
) -> torch.optim.Optimizer:
    """
    Create optimizer with proper parameter groups.
    
    Automatically separates parameters that should/shouldn't have weight decay.
    
    Args:
        model: Model to optimize
        config: Optimizer configuration
        lr: Learning rate override
        weight_decay: Weight decay override
        **kwargs: Additional optimizer arguments
        
    Returns:
        Configured optimizer
        
    Example:
        >>> optimizer = create_optimizer(model, lr=1e-4, weight_decay=0.01)
    """
    config = config or OptimizerConfig()
    
    lr = lr or config.lr
    weight_decay = weight_decay or config.weight_decay
    
    # Create parameter groups
    param_groups = create_param_groups(
        model,
        weight_decay=weight_decay,
        no_decay_keywords=config.no_decay_keywords,
    )
    
    # Create optimizer
    optimizer_cls = {
        "adam": Adam,
        "adamw": AdamW,
        "sgd": SGD,
    }.get(config.optimizer.lower(), AdamW)
    
    optimizer_kwargs = {"lr": lr, **kwargs}
    
    if config.optimizer.lower() in ("adam", "adamw"):
        optimizer_kwargs["betas"] = config.betas
        optimizer_kwargs["eps"] = config.eps
        
        # Use fused optimizer if available
        if config.fused and torch.cuda.is_available():
            try:
                optimizer_kwargs["fused"] = True
            except:
                pass
    elif config.optimizer.lower() == "sgd":
        optimizer_kwargs["momentum"] = config.momentum
        optimizer_kwargs["nesterov"] = config.nesterov
    
    return optimizer_cls(param_groups, **optimizer_kwargs)


def create_param_groups(
    model: nn.Module,
    weight_decay: float = 0.01,
    no_decay_keywords: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Create parameter groups with proper weight decay.
    
    Separates:
    - Decay group: weights of Linear, Conv, Embedding
    - No-decay group: biases, LayerNorm, BatchNorm
    """
    no_decay_keywords = no_decay_keywords or ["bias", "LayerNorm", "layernorm", "ln"]
    
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if any(keyword in name for keyword in no_decay_keywords):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


# =============================================================================
# Learning Rate Schedules
# =============================================================================

def get_cosine_schedule(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    num_warmup_steps: int = 0,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """
    Cosine learning rate schedule with warmup.
    
    Args:
        optimizer: Optimizer
        num_training_steps: Total training steps
        num_warmup_steps: Warmup steps
        min_lr_ratio: Minimum LR as ratio of initial LR
        
    Returns:
        Learning rate scheduler
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        
        return max(
            min_lr_ratio,
            0.5 * (1.0 + math.cos(math.pi * progress))
        )
    
    return LambdaLR(optimizer, lr_lambda)


def get_linear_schedule(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    num_warmup_steps: int = 0,
) -> LambdaLR:
    """
    Linear learning rate schedule with warmup.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        return max(
            0.0,
            float(num_training_steps - current_step) /
            float(max(1, num_training_steps - num_warmup_steps))
        )
    
    return LambdaLR(optimizer, lr_lambda)


def get_constant_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
) -> LambdaLR:
    """Constant LR with warmup."""
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0
    
    return LambdaLR(optimizer, lr_lambda)


def get_one_cycle_schedule(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    max_lr: float,
    pct_start: float = 0.3,
) -> OneCycleLR:
    """
    One-cycle learning rate schedule.
    
    Ramps up, then cosine decay. Often gives best results.
    """
    return OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=num_training_steps,
        pct_start=pct_start,
        anneal_strategy="cos",
    )


# =============================================================================
# Advanced Optimizers
# =============================================================================

class LAMB(torch.optim.Optimizer):
    """
    LAMB optimizer for large batch training.
    
    Layer-wise Adaptive Moments optimizer enables training
    with batch sizes up to 64K without accuracy loss.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.01,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("LAMB does not support sparse gradients")
                
                state = self.state[p]
                
                # Initialize state
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1
                
                # Momentum and variance
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                
                exp_avg_corrected = exp_avg / bias_correction1
                exp_avg_sq_corrected = exp_avg_sq / bias_correction2
                
                # Adam update
                adam_step = exp_avg_corrected / (exp_avg_sq_corrected.sqrt() + group["eps"])
                
                # Weight decay
                if group["weight_decay"] != 0:
                    adam_step.add_(p, alpha=group["weight_decay"])
                
                # LAMB trust ratio
                weight_norm = p.norm()
                adam_norm = adam_step.norm()
                
                if weight_norm > 0 and adam_norm > 0:
                    trust_ratio = weight_norm / adam_norm
                else:
                    trust_ratio = 1.0
                
                p.add_(adam_step, alpha=-group["lr"] * trust_ratio)
        
        return loss


# =============================================================================
# Utilities
# =============================================================================

def get_num_params(model: nn.Module, trainable_only: bool = True) -> int:
    """Get number of parameters in model."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def freeze_params(model: nn.Module, except_patterns: Optional[List[str]] = None):
    """
    Freeze model parameters.
    
    Args:
        model: Model to freeze
        except_patterns: Parameter names matching these patterns stay trainable
    """
    except_patterns = except_patterns or []
    
    for name, param in model.named_parameters():
        should_freeze = True
        
        for pattern in except_patterns:
            if pattern in name:
                should_freeze = False
                break
        
        param.requires_grad = not should_freeze


def unfreeze_params(model: nn.Module, patterns: Optional[List[str]] = None):
    """
    Unfreeze model parameters.
    
    Args:
        model: Model to unfreeze
        patterns: Only unfreeze parameters matching these patterns (None = all)
    """
    for name, param in model.named_parameters():
        if patterns is None:
            param.requires_grad = True
        else:
            for pattern in patterns:
                if pattern in name:
                    param.requires_grad = True
                    break
