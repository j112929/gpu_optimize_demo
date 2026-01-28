"""
Automatic Mixed Precision (AMP) - Train 2x faster with half memory.

Provides utilities for mixed precision training with FP16/BF16,
automatic loss scaling, and gradient handling.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Union
from contextlib import contextmanager
import functools


@dataclass
class AMPConfig:
    """Configuration for Automatic Mixed Precision."""
    # Precision
    enabled: bool = True
    dtype: torch.dtype = torch.float16  # float16 or bfloat16
    
    # Grad Scaler
    init_scale: float = 65536.0
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    
    # Behavior
    cache_enabled: bool = True


class AMPTrainer:
    """
    Mixed Precision Training Wrapper.
    
    Automatically handles:
    - Forward pass in FP16/BF16
    - Gradient scaling
    - Overflow detection
    - Automatic scale adjustment
    
    Example:
        >>> amp_trainer = AMPTrainer()
        >>> 
        >>> for batch in dataloader:
        ...     with amp_trainer.autocast():
        ...         loss = model(batch)
        ...     
        ...     amp_trainer.backward(loss)
        ...     amp_trainer.step(optimizer)
        ...     optimizer.zero_grad()
    """
    
    def __init__(self, config: Optional[AMPConfig] = None):
        self.config = config or AMPConfig()
        
        self.scaler = GradScaler(
            init_scale=self.config.init_scale,
            growth_factor=self.config.growth_factor,
            backoff_factor=self.config.backoff_factor,
            growth_interval=self.config.growth_interval,
            enabled=self.config.enabled,
        )
        
        self._overflow_count = 0
        self._step_count = 0
    
    @contextmanager
    def autocast(self):
        """Context manager for mixed precision forward pass."""
        with autocast(
            enabled=self.config.enabled,
            dtype=self.config.dtype,
            cache_enabled=self.config.cache_enabled,
        ):
            yield
    
    def backward(self, loss: torch.Tensor):
        """
        Backward pass with gradient scaling.
        
        Args:
            loss: Loss tensor to backpropagate
        """
        self.scaler.scale(loss).backward()
    
    def step(self, optimizer: torch.optim.Optimizer):
        """
        Optimizer step with unscaling and overflow check.
        
        Args:
            optimizer: Optimizer to step
        """
        self.scaler.step(optimizer)
        self.scaler.update()
        self._step_count += 1
        
        # Track overflows
        if self.scaler.get_scale() < self.config.init_scale:
            self._overflow_count += 1
    
    def unscale_gradients(self, optimizer: torch.optim.Optimizer):
        """Unscale gradients before clipping."""
        self.scaler.unscale_(optimizer)
    
    @property
    def scale(self) -> float:
        """Current gradient scale."""
        return self.scaler.get_scale()
    
    @property
    def overflow_rate(self) -> float:
        """Fraction of steps with gradient overflow."""
        if self._step_count == 0:
            return 0.0
        return self._overflow_count / self._step_count
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state for checkpointing."""
        return {
            "scaler": self.scaler.state_dict(),
            "overflow_count": self._overflow_count,
            "step_count": self._step_count,
        }
    
    def load_state_dict(self, state: Dict[str, Any]):
        """Load state from checkpoint."""
        self.scaler.load_state_dict(state["scaler"])
        self._overflow_count = state["overflow_count"]
        self._step_count = state["step_count"]


def auto_cast_forward(model: nn.Module, dtype: torch.dtype = torch.float16) -> nn.Module:
    """
    Wrap model's forward to always use autocast.
    
    Args:
        model: PyTorch model
        dtype: Mixed precision dtype
        
    Returns:
        Model with autocast-wrapped forward
    """
    original_forward = model.forward
    
    @functools.wraps(original_forward)
    def wrapped_forward(*args, **kwargs):
        with autocast(dtype=dtype):
            return original_forward(*args, **kwargs)
    
    model.forward = wrapped_forward
    return model


def scale_loss(
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler] = None,
) -> torch.Tensor:
    """
    Scale loss for mixed precision training.
    
    Args:
        loss: Loss tensor
        optimizer: Optimizer
        scaler: Optional GradScaler (creates one if not provided)
        
    Returns:
        Scaled loss ready for backward
    """
    if scaler is None:
        scaler = GradScaler()
    
    return scaler.scale(loss)


# =============================================================================
# BFloat16 Support
# =============================================================================

def supports_bfloat16() -> bool:
    """Check if current GPU supports BFloat16."""
    if not torch.cuda.is_available():
        return False
    
    capability = torch.cuda.get_device_capability()
    # BF16 supported on Ampere (8.0) and later
    return capability[0] >= 8


def get_optimal_dtype() -> torch.dtype:
    """Get optimal mixed precision dtype for current hardware."""
    if supports_bfloat16():
        # BF16 has better numerical stability
        return torch.bfloat16
    return torch.float16


# =============================================================================
# Training Loop Helper
# =============================================================================

class MixedPrecisionTrainingLoop:
    """
    Complete training loop with mixed precision.
    
    Example:
        >>> loop = MixedPrecisionTrainingLoop(model, optimizer, criterion)
        >>> for epoch in range(epochs):
        ...     for batch in dataloader:
        ...         loss = loop.train_step(batch, labels)
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        config: Optional[AMPConfig] = None,
        max_grad_norm: Optional[float] = 1.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.max_grad_norm = max_grad_norm
        
        self.amp = AMPTrainer(config)
    
    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        """
        Single training step with mixed precision.
        
        Returns:
            Loss value
        """
        self.optimizer.zero_grad()
        
        with self.amp.autocast():
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
        
        self.amp.backward(loss)
        
        if self.max_grad_norm is not None:
            self.amp.unscale_gradients(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm
            )
        
        self.amp.step(self.optimizer)
        
        return loss.item()
    
    @torch.no_grad()
    def eval_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Evaluation step with mixed precision.
        
        Returns:
            Dict with loss and predictions
        """
        self.model.eval()
        
        with self.amp.autocast():
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
        
        return {
            "loss": loss.item(),
            "outputs": outputs,
        }


# =============================================================================
# Utilities
# =============================================================================

def estimate_memory_savings(model: nn.Module) -> Dict[str, float]:
    """
    Estimate memory savings from mixed precision.
    
    Returns:
        Dict with memory estimates in GB
    """
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    
    fp32_size = param_bytes / (1024 ** 3)  # GB
    fp16_size = fp32_size / 2
    
    # Activations typically 2-4x parameters
    fp32_activations = fp32_size * 3
    fp16_activations = fp16_size * 3
    
    return {
        "fp32_params_gb": fp32_size,
        "fp16_params_gb": fp16_size,
        "fp32_total_gb": fp32_size + fp32_activations,
        "fp16_total_gb": fp16_size + fp16_activations,
        "savings_gb": (fp32_size + fp32_activations) - (fp16_size + fp16_activations),
        "savings_percent": 50.0,
    }
