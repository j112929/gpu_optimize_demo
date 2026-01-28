"""
Fused Operations - Memory-efficient fused kernels using Triton.

Fusing multiple operations into a single kernel reduces memory bandwidth
requirements and kernel launch overhead, leading to significant speedups.
"""

import torch
import triton
import triton.language as tl
from typing import Optional


# =============================================================================
# Fused GELU Activation
# =============================================================================

@triton.jit
def _gelu_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for GELU activation."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Using the faster approximation
    coeff = 0.7978845608028654  # sqrt(2/pi)
    x_cubed = x * x * x
    inner = coeff * (x + 0.044715 * x_cubed)
    
    # tanh approximation using exp
    exp_2inner = tl.exp(2.0 * inner)
    tanh_val = (exp_2inner - 1.0) / (exp_2inner + 1.0)
    
    output = 0.5 * x * (1.0 + tanh_val)
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)


def fused_gelu(x: torch.Tensor) -> torch.Tensor:
    """
    Fused GELU activation using Triton.
    
    Faster than PyTorch's nn.GELU for large tensors due to reduced
    memory bandwidth and kernel launch overhead.
    
    Args:
        x: Input tensor
        
    Returns:
        GELU(x)
        
    Example:
        >>> x = torch.randn(1024, 1024, device='cuda')
        >>> y = fused_gelu(x)
    """
    assert x.is_cuda, "Input must be on CUDA device"
    
    output = torch.empty_like(x)
    n_elements = x.numel()
    
    # Configure grid
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    _gelu_kernel[grid](
        x,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


# =============================================================================
# Fused Softmax
# =============================================================================

@triton.jit
def _softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for row-wise softmax."""
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Load row with masking
    mask = col_offsets < n_cols
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
    
    # Compute max for numerical stability
    row_max = tl.max(row, axis=0)
    
    # Subtract max and compute exp
    numerator = tl.exp(row - row_max)
    
    # Compute sum
    denominator = tl.sum(numerator, axis=0)
    
    # Compute softmax
    softmax_output = numerator / denominator
    
    # Store result
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)


def fused_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Fused softmax using Triton.
    
    More memory-efficient than PyTorch softmax as it doesn't need to
    store intermediate results for the backward pass.
    
    Args:
        x: Input tensor
        dim: Dimension to apply softmax (default: -1)
        
    Returns:
        Softmax(x) along specified dimension
        
    Example:
        >>> x = torch.randn(128, 512, device='cuda')
        >>> y = fused_softmax(x)
    """
    assert x.is_cuda, "Input must be on CUDA device"
    
    # Handle dimension
    if dim != -1 and dim != x.ndim - 1:
        x = x.transpose(dim, -1)
        need_transpose = True
    else:
        need_transpose = False
    
    # Reshape to 2D
    original_shape = x.shape
    x = x.contiguous()
    x_2d = x.view(-1, x.shape[-1])
    
    n_rows, n_cols = x_2d.shape
    
    # Ensure BLOCK_SIZE is power of 2 and >= n_cols
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    output = torch.empty_like(x_2d)
    
    # Launch kernel
    _softmax_kernel[(n_rows,)](
        x_2d,
        output,
        x_2d.stride(0),
        output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape back
    output = output.view(original_shape)
    
    if need_transpose:
        output = output.transpose(dim, -1)
    
    return output


# =============================================================================
# Fused Add + LayerNorm
# =============================================================================

@triton.jit
def _add_layernorm_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    mean_ptr,
    rstd_ptr,
    stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for fused residual add + layer normalization."""
    row = tl.program_id(0)
    
    # Compute offsets
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    
    # Load x and residual, compute sum
    x_row_ptr = x_ptr + row * stride
    residual_row_ptr = residual_ptr + row * stride
    
    x = tl.load(x_row_ptr + cols, mask=mask, other=0.0)
    residual = tl.load(residual_row_ptr + cols, mask=mask, other=0.0)
    
    # Fused add
    y = x + residual
    
    # Compute mean
    mean = tl.sum(y, axis=0) / N
    
    # Compute variance
    y_centered = y - mean
    var = tl.sum(y_centered * y_centered, axis=0) / N
    
    # Compute reciprocal standard deviation
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Normalize
    y_norm = y_centered * rstd
    
    # Apply affine transformation
    weight = tl.load(weight_ptr + cols, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0)
    
    output = y_norm * weight + bias
    
    # Store output
    output_row_ptr = output_ptr + row * stride
    tl.store(output_row_ptr + cols, output, mask=mask)
    
    # Optionally store mean and rstd for backward pass
    if mean_ptr is not None:
        tl.store(mean_ptr + row, mean)
    if rstd_ptr is not None:
        tl.store(rstd_ptr + row, rstd)


def fused_add_layernorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
    return_stats: bool = False,
) -> torch.Tensor:
    """
    Fused residual addition and layer normalization.
    
    Combines x + residual followed by LayerNorm into a single kernel,
    reducing memory bandwidth by 2x.
    
    Args:
        x: Input tensor [..., hidden_size]
        residual: Residual tensor (same shape as x)
        weight: LayerNorm weight [hidden_size]
        bias: LayerNorm bias [hidden_size]
        eps: Epsilon for numerical stability
        return_stats: Whether to return mean and rstd
        
    Returns:
        Normalized output, optionally with mean and rstd
        
    Example:
        >>> x = torch.randn(32, 128, 768, device='cuda')
        >>> residual = torch.randn_like(x)
        >>> weight = torch.ones(768, device='cuda')
        >>> bias = torch.zeros(768, device='cuda')
        >>> out = fused_add_layernorm(x, residual, weight, bias)
    """
    assert x.is_cuda and residual.is_cuda, "Inputs must be on CUDA"
    assert x.shape == residual.shape, "x and residual must have same shape"
    
    # Reshape to 2D
    original_shape = x.shape
    hidden_size = x.shape[-1]
    x_2d = x.contiguous().view(-1, hidden_size)
    residual_2d = residual.contiguous().view(-1, hidden_size)
    
    n_rows = x_2d.shape[0]
    
    # Allocate output
    output = torch.empty_like(x_2d)
    
    # Optionally allocate stats
    mean = torch.empty(n_rows, dtype=x.dtype, device=x.device) if return_stats else None
    rstd = torch.empty(n_rows, dtype=x.dtype, device=x.device) if return_stats else None
    
    # Determine block size
    BLOCK_SIZE = triton.next_power_of_2(hidden_size)
    
    # Launch kernel
    _add_layernorm_kernel[(n_rows,)](
        x_2d,
        residual_2d,
        weight,
        bias,
        output,
        mean,
        rstd,
        x_2d.stride(0),
        hidden_size,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    output = output.view(original_shape)
    
    if return_stats:
        return output, mean, rstd
    return output


# =============================================================================
# Fused Dropout + Add
# =============================================================================

@triton.jit
def _dropout_add_kernel(
    x_ptr,
    residual_ptr,
    output_ptr,
    seed,
    p,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for fused dropout + add."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask)
    residual = tl.load(residual_ptr + offsets, mask=mask)
    
    # Generate random numbers for dropout
    random = tl.rand(seed, offsets)
    
    # Apply dropout: scale by 1/(1-p) when keeping
    keep_mask = random > p
    scale = 1.0 / (1.0 - p)
    x_dropped = tl.where(keep_mask, x * scale, tl.zeros_like(x))
    
    # Add residual
    output = x_dropped + residual
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)


def fused_dropout_add(
    x: torch.Tensor,
    residual: torch.Tensor,
    p: float = 0.1,
    training: bool = True,
) -> torch.Tensor:
    """
    Fused dropout and residual addition.
    
    Combines dropout(x) + residual into a single kernel operation.
    
    Args:
        x: Input tensor to apply dropout to
        residual: Residual tensor to add
        p: Dropout probability
        training: Whether in training mode (if False, no dropout applied)
        
    Returns:
        dropout(x) + residual
        
    Example:
        >>> x = torch.randn(32, 128, 768, device='cuda')
        >>> residual = torch.randn_like(x)
        >>> out = fused_dropout_add(x, residual, p=0.1)
    """
    assert x.is_cuda and residual.is_cuda, "Inputs must be on CUDA"
    
    if not training or p == 0:
        return x + residual
    
    output = torch.empty_like(x)
    n_elements = x.numel()
    
    # Generate random seed
    seed = torch.randint(0, 2**31, (1,), device=x.device).item()
    
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    _dropout_add_kernel[grid](
        x,
        residual,
        output,
        seed,
        p,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


# =============================================================================
# Fused SiLU (Swish) Activation
# =============================================================================

@triton.jit
def _silu_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for SiLU (Swish) activation: x * sigmoid(x)."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # SiLU: x * sigmoid(x) = x * (1 / (1 + exp(-x)))
    sigmoid = 1.0 / (1.0 + tl.exp(-x))
    output = x * sigmoid
    
    tl.store(output_ptr + offsets, output, mask=mask)


def fused_silu(x: torch.Tensor) -> torch.Tensor:
    """
    Fused SiLU (Swish) activation using Triton.
    
    SiLU(x) = x * sigmoid(x)
    
    Args:
        x: Input tensor
        
    Returns:
        SiLU(x)
    """
    assert x.is_cuda, "Input must be on CUDA device"
    
    output = torch.empty_like(x)
    n_elements = x.numel()
    
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    _silu_kernel[grid](
        x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output


# =============================================================================
# Fused RMSNorm
# =============================================================================

@triton.jit
def _rmsnorm_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for RMSNorm."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    
    x = tl.load(x_ptr + row * stride + cols, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + cols, mask=mask, other=1.0)
    
    # Compute RMS
    variance = tl.sum(x * x, axis=0) / N
    rrms = 1.0 / tl.sqrt(variance + eps)
    
    # Normalize and scale
    output = x * rrms * weight
    
    tl.store(output_ptr + row * stride + cols, output, mask=mask)


def fused_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Fused RMSNorm using Triton.
    
    RMSNorm(x) = x / RMS(x) * weight
    where RMS(x) = sqrt(mean(x^2))
    
    Args:
        x: Input tensor [..., hidden_size]
        weight: Scale weight [hidden_size]
        eps: Epsilon for numerical stability
        
    Returns:
        Normalized output
    """
    assert x.is_cuda, "Input must be on CUDA"
    
    original_shape = x.shape
    hidden_size = x.shape[-1]
    x_2d = x.contiguous().view(-1, hidden_size)
    n_rows = x_2d.shape[0]
    
    output = torch.empty_like(x_2d)
    
    BLOCK_SIZE = triton.next_power_of_2(hidden_size)
    
    _rmsnorm_kernel[(n_rows,)](
        x_2d, weight, output,
        x_2d.stride(0), hidden_size, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output.view(original_shape)
