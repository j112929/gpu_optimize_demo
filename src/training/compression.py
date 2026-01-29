"""
Gradient Compression - Reduce communication overhead in distributed training.

Provides:
- Top-K sparsification (keep only top K% gradients)
- Random sparsification
- Quantization (FP16, INT8, 1-bit)
- Error feedback for lossless compression
- PowerSGD for low-rank gradient compression

Key benefits:
- Up to 100× reduction in communication volume
- Critical for multi-node training
- Minimal accuracy loss with error feedback
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import math


@dataclass
class GradientCompressionConfig:
    """Configuration for gradient compression."""
    
    # Compression method
    method: str = "topk"             # "topk", "random", "quantize", "powersgd", "none"
    
    # Top-K parameters
    topk_ratio: float = 0.01         # Keep top 1% of gradients
    
    # Random sparsification
    random_ratio: float = 0.1        # Keep 10% randomly
    
    # Quantization parameters
    quantize_bits: int = 8           # 8 for INT8, 1 for 1-bit SGD
    
    # PowerSGD parameters
    powersgd_rank: int = 4           # Low-rank approximation rank
    powersgd_start_iter: int = 10    # Start PowerSGD after N iterations
    
    # Error feedback (critical for convergence)
    error_feedback: bool = True
    
    # Warm-up (use full gradients initially)
    warmup_steps: int = 100
    

# =============================================================================
# Gradient Compressor Base
# =============================================================================

class GradientCompressor:
    """
    Base class for gradient compression.
    
    Compresses gradients before all-reduce operation to reduce
    communication bandwidth requirements.
    """
    
    def __init__(self, config: GradientCompressionConfig):
        self.config = config
        self.step_count = 0
        
        # Error feedback buffers
        self.error_buffers: Dict[str, torch.Tensor] = {}
    
    def compress(
        self,
        gradient: torch.Tensor,
        name: str = "",
    ) -> Tuple[torch.Tensor, Any]:
        """
        Compress gradient.
        
        Args:
            gradient: Gradient tensor to compress
            name: Parameter name (for error feedback tracking)
            
        Returns:
            compressed: Compressed gradient
            context: Information needed for decompression
        """
        raise NotImplementedError
    
    def decompress(
        self,
        compressed: torch.Tensor,
        context: Any,
    ) -> torch.Tensor:
        """
        Decompress gradient.
        
        Args:
            compressed: Compressed gradient
            context: Decompression context
            
        Returns:
            Decompressed gradient
        """
        raise NotImplementedError
    
    def step(self):
        """Called after each optimization step."""
        self.step_count += 1


# =============================================================================
# Top-K Compression
# =============================================================================

class TopKCompressor(GradientCompressor):
    """
    Top-K gradient compression.
    
    Keeps only the top K% of gradients by magnitude.
    With error feedback, achieves near-lossless compression.
    
    Example:
        >>> compressor = TopKCompressor(GradientCompressionConfig(topk_ratio=0.01))
        >>> compressed, ctx = compressor.compress(gradient, "layer1.weight")
        >>> # Communicate compressed gradient
        >>> decompressed = compressor.decompress(compressed, ctx)
    """
    
    def compress(
        self,
        gradient: torch.Tensor,
        name: str = "",
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple]:
        """Top-K compression with error feedback."""
        
        # Add error feedback
        if self.config.error_feedback and name in self.error_buffers:
            gradient = gradient + self.error_buffers[name]
        
        # Flatten
        flat = gradient.flatten()
        numel = flat.numel()
        k = max(1, int(numel * self.config.topk_ratio))
        
        # Get top-k values and indices
        values, indices = torch.topk(flat.abs(), k)
        top_values = flat[indices]
        
        # Compute error for feedback
        if self.config.error_feedback:
            mask = torch.zeros_like(flat)
            mask[indices] = 1
            error = flat * (1 - mask)
            self.error_buffers[name] = error.view(gradient.shape)
        
        context = (gradient.shape, gradient.dtype, indices, numel)
        return (top_values, indices), context
    
    def decompress(
        self,
        compressed: Tuple[torch.Tensor, torch.Tensor],
        context: Tuple,
    ) -> torch.Tensor:
        """Decompress Top-K gradient."""
        values, indices = compressed
        shape, dtype, _, numel = context
        
        # Reconstruct
        flat = torch.zeros(numel, dtype=dtype, device=values.device)
        flat[indices] = values
        
        return flat.view(shape)


# =============================================================================
# Random Sparsification
# =============================================================================

class RandomCompressor(GradientCompressor):
    """
    Random gradient sparsification.
    
    Randomly selects a subset of gradients to communicate.
    Simpler than Top-K but less accurate.
    """
    
    def compress(
        self,
        gradient: torch.Tensor,
        name: str = "",
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple]:
        """Random sparsification."""
        
        # Add error feedback
        if self.config.error_feedback and name in self.error_buffers:
            gradient = gradient + self.error_buffers[name]
        
        flat = gradient.flatten()
        numel = flat.numel()
        k = max(1, int(numel * self.config.random_ratio))
        
        # Random indices
        indices = torch.randperm(numel, device=gradient.device)[:k]
        values = flat[indices]
        
        # Scale to maintain expected value
        values = values / self.config.random_ratio
        
        # Error feedback
        if self.config.error_feedback:
            mask = torch.zeros_like(flat)
            mask[indices] = 1
            error = flat * (1 - mask)
            self.error_buffers[name] = error.view(gradient.shape)
        
        context = (gradient.shape, gradient.dtype, numel)
        return (values, indices), context
    
    def decompress(
        self,
        compressed: Tuple[torch.Tensor, torch.Tensor],
        context: Tuple,
    ) -> torch.Tensor:
        """Decompress random sparsified gradient."""
        values, indices = compressed
        shape, dtype, numel = context
        
        flat = torch.zeros(numel, dtype=dtype, device=values.device)
        flat[indices] = values * self.config.random_ratio  # Unscale
        
        return flat.view(shape)


# =============================================================================
# Quantization Compression
# =============================================================================

class QuantizationCompressor(GradientCompressor):
    """
    Gradient quantization to reduce communication.
    
    Supports:
    - INT8 quantization (8-bit)
    - INT4 quantization (4-bit)
    - 1-bit SGD (sign only)
    """
    
    def compress(
        self,
        gradient: torch.Tensor,
        name: str = "",
    ) -> Tuple[torch.Tensor, Tuple]:
        """Quantize gradient."""
        
        # Add error feedback
        if self.config.error_feedback and name in self.error_buffers:
            gradient = gradient + self.error_buffers[name]
        
        if self.config.quantize_bits == 1:
            # 1-bit SGD: only transmit sign
            quantized = gradient.sign()
            scale = gradient.abs().mean()
            
            # Error feedback for 1-bit
            if self.config.error_feedback:
                error = gradient - quantized * scale
                self.error_buffers[name] = error
            
            context = (gradient.shape, gradient.dtype, scale)
            return quantized.to(torch.int8), context
        
        else:
            # INT8 / INT4 quantization
            bits = self.config.quantize_bits
            qmin = -(2 ** (bits - 1))
            qmax = 2 ** (bits - 1) - 1
            
            # Compute scale
            scale = gradient.abs().max() / qmax
            scale = torch.clamp(scale, min=1e-8)
            
            # Quantize
            quantized = torch.clamp(
                torch.round(gradient / scale),
                qmin, qmax
            ).to(torch.int8)
            
            # Error feedback
            if self.config.error_feedback:
                dequantized = quantized.float() * scale
                error = gradient - dequantized
                self.error_buffers[name] = error
            
            context = (gradient.shape, gradient.dtype, scale)
            return quantized, context
    
    def decompress(
        self,
        compressed: torch.Tensor,
        context: Tuple,
    ) -> torch.Tensor:
        """Dequantize gradient."""
        shape, dtype, scale = context
        return (compressed.to(dtype) * scale).view(shape)


# =============================================================================
# PowerSGD (Low-Rank Gradient Compression)
# =============================================================================

class PowerSGDCompressor(GradientCompressor):
    """
    PowerSGD: Low-rank gradient compression.
    
    Uses power iteration to approximate gradients with low-rank matrices.
    Very effective for dense layers with high compression ratios.
    
    Paper: https://arxiv.org/abs/1905.13727
    
    Example:
        >>> compressor = PowerSGDCompressor(GradientCompressionConfig(powersgd_rank=4))
        >>> compressed, ctx = compressor.compress(gradient, "layer1.weight")
    """
    
    def __init__(self, config: GradientCompressionConfig):
        super().__init__(config)
        self.rank = config.powersgd_rank
        
        # Low-rank approximation matrices
        self.P_buffers: Dict[str, torch.Tensor] = {}
        self.Q_buffers: Dict[str, torch.Tensor] = {}
    
    def compress(
        self,
        gradient: torch.Tensor,
        name: str = "",
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple]:
        """Low-rank compression using power iteration."""
        
        # Add error feedback
        if self.config.error_feedback and name in self.error_buffers:
            gradient = gradient + self.error_buffers[name]
        
        # Reshape to 2D matrix
        original_shape = gradient.shape
        if gradient.dim() == 1:
            # Vectors: just return as-is for small tensors
            if gradient.numel() < 1000:
                return (gradient, None), (original_shape, gradient.dtype, False)
            # For larger vectors, reshape to 2D
            gradient = gradient.view(-1, 1)
        
        if gradient.dim() > 2:
            gradient = gradient.view(gradient.shape[0], -1)
        
        m, n = gradient.shape
        
        # Initialize or get Q matrix
        if name not in self.Q_buffers or self.Q_buffers[name].shape[0] != n:
            self.Q_buffers[name] = torch.randn(n, self.rank, device=gradient.device)
            self.Q_buffers[name], _ = torch.linalg.qr(self.Q_buffers[name])
        
        Q = self.Q_buffers[name]
        
        # Power iteration step
        # P = G @ Q
        P = torch.matmul(gradient, Q)
        
        # Orthogonalize P
        P, _ = torch.linalg.qr(P)
        
        # Q = G^T @ P
        Q = torch.matmul(gradient.T, P)
        
        # Store for next iteration
        self.Q_buffers[name] = Q
        
        # Error feedback
        if self.config.error_feedback:
            reconstructed = torch.matmul(P, Q.T)
            error = gradient - reconstructed
            self.error_buffers[name] = error.view(original_shape)
        
        context = (original_shape, gradient.dtype, True)
        return (P, Q), context
    
    def decompress(
        self,
        compressed: Tuple[torch.Tensor, torch.Tensor],
        context: Tuple,
    ) -> torch.Tensor:
        """Reconstruct from low-rank approximation."""
        P, Q = compressed
        shape, dtype, is_matrix = context
        
        if not is_matrix:
            return P.view(shape)
        
        # Reconstruct: G ≈ P @ Q^T
        reconstructed = torch.matmul(P, Q.T)
        return reconstructed.view(shape)


# =============================================================================
# Distributed Compression Wrapper
# =============================================================================

class DistributedGradientCompressor:
    """
    Wrapper for distributed gradient compression.
    
    Integrates compression with PyTorch distributed communication.
    
    Example:
        >>> compressor = DistributedGradientCompressor(
        >>>     GradientCompressionConfig(method="topk", topk_ratio=0.01)
        >>> )
        >>> 
        >>> # In training loop
        >>> for param in model.parameters():
        >>>     if param.grad is not None:
        >>>         compressor.compress_and_allreduce(param.grad, name)
    """
    
    def __init__(self, config: GradientCompressionConfig):
        self.config = config
        
        # Create compressor based on method
        if config.method == "topk":
            self.compressor = TopKCompressor(config)
        elif config.method == "random":
            self.compressor = RandomCompressor(config)
        elif config.method == "quantize":
            self.compressor = QuantizationCompressor(config)
        elif config.method == "powersgd":
            self.compressor = PowerSGDCompressor(config)
        else:
            self.compressor = None
    
    def compress_and_allreduce(
        self,
        gradient: torch.Tensor,
        name: str = "",
    ) -> torch.Tensor:
        """
        Compress gradient, all-reduce, and decompress.
        
        Args:
            gradient: Gradient tensor
            name: Parameter name
            
        Returns:
            All-reduced gradient
        """
        if self.compressor is None or self.compressor.step_count < self.config.warmup_steps:
            # No compression during warmup
            if dist.is_initialized():
                dist.all_reduce(gradient)
                gradient /= dist.get_world_size()
            return gradient
        
        # Compress
        compressed, context = self.compressor.compress(gradient, name)
        
        # All-reduce compressed tensors
        if dist.is_initialized():
            if isinstance(compressed, tuple):
                # Handle sparse representations
                for tensor in compressed:
                    if tensor is not None:
                        dist.all_reduce(tensor)
                        tensor /= dist.get_world_size()
            else:
                dist.all_reduce(compressed)
                compressed /= dist.get_world_size()
        
        # Decompress
        decompressed = self.compressor.decompress(compressed, context)
        
        return decompressed
    
    def step(self):
        """Called after each optimization step."""
        if self.compressor:
            self.compressor.step()


# =============================================================================
# Hook-based Gradient Compression
# =============================================================================

class GradientCompressionHook:
    """
    Register gradient compression as a backward hook.
    
    Example:
        >>> hook = GradientCompressionHook(model, config)
        >>> # Training proceeds normally, gradients are compressed automatically
        >>> hook.remove()  # Cleanup
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: GradientCompressionConfig,
    ):
        self.model = model
        self.compressor = DistributedGradientCompressor(config)
        self.handles: List[Any] = []
        
        # Register hooks
        for name, param in model.named_parameters():
            if param.requires_grad:
                handle = param.register_post_accumulate_grad_hook(
                    lambda p, n=name: self._hook(p, n)
                )
                self.handles.append(handle)
    
    def _hook(self, param: torch.Tensor, name: str):
        """Gradient compression hook."""
        if param.grad is not None:
            param.grad.data = self.compressor.compress_and_allreduce(
                param.grad.data, name
            )
    
    def step(self):
        """Called after each optimization step."""
        self.compressor.step()
    
    def remove(self):
        """Remove all hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


# =============================================================================
# Utilities
# =============================================================================

def estimate_compression_ratio(config: GradientCompressionConfig) -> float:
    """Estimate theoretical compression ratio."""
    if config.method == "topk":
        return 1.0 / config.topk_ratio
    elif config.method == "random":
        return 1.0 / config.random_ratio
    elif config.method == "quantize":
        return 32.0 / config.quantize_bits  # Assuming FP32 original
    elif config.method == "powersgd":
        # Depends on matrix shape, approximate
        return 10.0  # Typical compression ratio
    return 1.0


def create_gradient_compressor(
    method: str = "topk",
    ratio: float = 0.01,
    error_feedback: bool = True,
) -> DistributedGradientCompressor:
    """
    Create a gradient compressor with sensible defaults.
    
    Args:
        method: Compression method ("topk", "random", "quantize", "powersgd")
        ratio: Compression ratio (for topk/random)
        error_feedback: Whether to use error feedback
        
    Returns:
        Configured compressor
    """
    config = GradientCompressionConfig(
        method=method,
        topk_ratio=ratio if method == "topk" else 0.01,
        random_ratio=ratio if method == "random" else 0.1,
        error_feedback=error_feedback,
    )
    return DistributedGradientCompressor(config)
