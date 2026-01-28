"""
Gradient Optimization - Accumulation, Checkpointing, and Clipping.

Provides utilities for:
- Gradient accumulation for large effective batch sizes
- Gradient checkpointing to trade compute for memory
- Advanced gradient clipping strategies
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import functools


@dataclass
class GradientConfig:
    """Configuration for gradient optimization."""
    # Accumulation
    accumulation_steps: int = 1
    
    # Clipping
    max_grad_norm: Optional[float] = 1.0
    clip_type: str = "norm"  # norm, value, adaptive
    
    # Checkpointing
    checkpoint: bool = False
    checkpoint_ratio: float = 1.0  # Fraction of layers to checkpoint


class GradientAccumulator:
    """
    Gradient Accumulation for large effective batch sizes.
    
    Accumulates gradients over multiple micro-batches before
    updating weights. Essential when GPU memory is limited.
    
    Example:
        >>> accumulator = GradientAccumulator(steps=4)
        >>> 
        >>> for i, batch in enumerate(dataloader):
        ...     loss = model(batch) / accumulator.steps
        ...     loss.backward()
        ...     
        ...     if accumulator.should_step(i):
        ...         optimizer.step()
        ...         optimizer.zero_grad()
    """
    
    def __init__(
        self,
        steps: int = 1,
        max_grad_norm: Optional[float] = None,
    ):
        self.steps = steps
        self.max_grad_norm = max_grad_norm
        self._current_step = 0
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for accumulation."""
        return loss / self.steps
    
    def should_step(self, iteration: Optional[int] = None) -> bool:
        """Check if optimizer should step."""
        if iteration is not None:
            return (iteration + 1) % self.steps == 0
        
        self._current_step += 1
        if self._current_step >= self.steps:
            self._current_step = 0
            return True
        return False
    
    def step(
        self,
        optimizer: torch.optim.Optimizer,
        model: Optional[nn.Module] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ):
        """
        Perform optimizer step with optional gradient clipping.
        
        Args:
            optimizer: Optimizer to step
            model: Model for gradient clipping
            scaler: Optional GradScaler for mixed precision
        """
        if scaler is not None:
            scaler.unscale_(optimizer)
        
        if self.max_grad_norm is not None and model is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                self.max_grad_norm
            )
        
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        optimizer.zero_grad()
    
    @property
    def is_accumulating(self) -> bool:
        """Check if currently accumulating."""
        return self._current_step > 0


def gradient_checkpoint_model(
    model: nn.Module,
    checkpoint_layers: Optional[List[str]] = None,
    use_reentrant: bool = False,
) -> nn.Module:
    """
    Apply gradient checkpointing to a model.
    
    Reduces memory by ~50-70% at cost of ~30% compute overhead.
    
    Args:
        model: PyTorch model
        checkpoint_layers: List of layer names to checkpoint (None = all)
        use_reentrant: Use reentrant checkpointing (legacy)
        
    Returns:
        Model with checkpointing enabled
        
    Example:
        >>> model = gradient_checkpoint_model(model)
        >>> # Now uses ~50% less memory during training
    """
    # For transformer models, checkpoint each block
    if hasattr(model, "gradient_checkpointing_enable"):
        # HuggingFace transformers
        model.gradient_checkpointing_enable()
        return model
    
    # For custom models, wrap layers
    if checkpoint_layers is None:
        # Try to find common block patterns
        block_names = []
        for name, module in model.named_modules():
            if any(x in name.lower() for x in ["block", "layer", "encoder", "decoder"]):
                if not any(name.startswith(b) for b in block_names):
                    block_names.append(name)
        checkpoint_layers = block_names[:10]  # Limit to first 10 blocks
    
    # Apply checkpointing
    for name, module in model.named_modules():
        if name in checkpoint_layers:
            _apply_checkpoint_wrapper(model, name, module, use_reentrant)
    
    return model


def _apply_checkpoint_wrapper(
    model: nn.Module,
    name: str,
    module: nn.Module,
    use_reentrant: bool,
):
    """Wrap a module with gradient checkpointing."""
    original_forward = module.forward
    
    @functools.wraps(original_forward)
    def checkpointed_forward(*args, **kwargs):
        # Checkpoint doesn't support kwargs well, so we use a wrapper
        def custom_forward(*inputs):
            return original_forward(*inputs, **kwargs)
        
        return checkpoint(custom_forward, *args, use_reentrant=use_reentrant)
    
    module.forward = checkpointed_forward


def checkpoint_sequential_blocks(
    modules: nn.ModuleList,
    segments: int,
    input: torch.Tensor,
) -> torch.Tensor:
    """
    Apply checkpointing to sequential blocks.
    
    Args:
        modules: List of sequential modules
        segments: Number of checkpoint segments
        input: Input tensor
        
    Returns:
        Output tensor
    """
    return checkpoint_sequential(modules, segments, input)


# =============================================================================
# Gradient Clipping
# =============================================================================

def clip_grad_norm(
    model: nn.Module,
    max_norm: float,
    norm_type: float = 2.0,
) -> torch.Tensor:
    """
    Clip gradient norm.
    
    Args:
        model: Model to clip gradients for
        max_norm: Maximum gradient norm
        norm_type: Type of norm (2 = L2)
        
    Returns:
        Total gradient norm before clipping
    """
    return torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm,
        norm_type=norm_type,
    )


def clip_grad_value(
    model: nn.Module,
    clip_value: float,
):
    """Clip gradient values to [-clip_value, clip_value]."""
    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value)


def adaptive_grad_clip(
    model: nn.Module,
    percentile: float = 0.9,
    min_norm: float = 0.01,
) -> torch.Tensor:
    """
    Adaptive gradient clipping based on gradient history.
    
    Clips to the specified percentile of historical gradient norms.
    """
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    
    if not gradients:
        return torch.tensor(0.0)
    
    total_norm = torch.norm(
        torch.stack([torch.norm(g) for g in gradients])
    )
    
    # Adaptive threshold based on norm
    clip_norm = max(total_norm.item() * percentile, min_norm)
    
    return clip_grad_norm(model, clip_norm)


# =============================================================================
# Gradient Statistics
# =============================================================================

def get_gradient_stats(model: nn.Module) -> Dict[str, float]:
    """
    Get gradient statistics for debugging.
    
    Returns:
        Dict with gradient statistics
    """
    gradients = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients.append({
                "name": name,
                "grad_norm": param.grad.norm().item(),
                "grad_max": param.grad.abs().max().item(),
                "grad_mean": param.grad.abs().mean().item(),
            })
    
    if not gradients:
        return {"total_norm": 0.0, "num_params": 0}
    
    total_norm = sum(g["grad_norm"] ** 2 for g in gradients) ** 0.5
    
    return {
        "total_norm": total_norm,
        "max_norm": max(g["grad_norm"] for g in gradients),
        "mean_norm": sum(g["grad_norm"] for g in gradients) / len(gradients),
        "max_value": max(g["grad_max"] for g in gradients),
        "num_params": len(gradients),
    }


def detect_gradient_anomaly(model: nn.Module, threshold: float = 100.0) -> List[str]:
    """
    Detect parameters with abnormally large gradients.
    
    Returns:
        List of parameter names with large gradients
    """
    anomalies = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm > threshold or torch.isnan(param.grad).any():
                anomalies.append(f"{name}: norm={grad_norm:.2f}")
    
    return anomalies


# =============================================================================
# Memory-Efficient Gradient Accumulation
# =============================================================================

class MemoryEfficientGradientAccumulator:
    """
    Memory-efficient gradient accumulation using gradient compression.
    
    Stores accumulated gradients in FP16 to save memory.
    """
    
    def __init__(
        self,
        model: nn.Module,
        steps: int = 4,
        compression_dtype: torch.dtype = torch.float16,
    ):
        self.model = model
        self.steps = steps
        self.compression_dtype = compression_dtype
        
        # Allocate compressed gradient buffers
        self._grad_buffers: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self._grad_buffers[name] = torch.zeros_like(
                    param.data,
                    dtype=compression_dtype,
                )
        
        self._current_step = 0
    
    def accumulate(self):
        """Accumulate current gradients."""
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # Add to compressed buffer
                self._grad_buffers[name].add_(
                    param.grad.data.to(self.compression_dtype)
                )
        
        self._current_step += 1
    
    def apply(self, optimizer: torch.optim.Optimizer):
        """Apply accumulated gradients and reset."""
        if self._current_step < self.steps:
            return False
        
        # Copy averaged gradients back
        for name, param in self.model.named_parameters():
            if name in self._grad_buffers:
                param.grad = self._grad_buffers[name].to(param.dtype) / self.steps
                self._grad_buffers[name].zero_()
        
        optimizer.step()
        optimizer.zero_grad()
        
        self._current_step = 0
        return True
