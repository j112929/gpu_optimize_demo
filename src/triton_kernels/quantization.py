"""
Quantization Kernels - INT8/INT4 quantized operations using Triton.

Provides efficient quantized matrix multiplication and activation functions
for inference optimization.
"""

import torch
import triton
import triton.language as tl
from typing import Optional, Tuple


# =============================================================================
# INT8 Quantization
# =============================================================================

@triton.jit
def _quantize_int8_kernel(
    x_ptr, scale_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Quantize FP32/FP16 to INT8."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    scale = tl.load(scale_ptr)
    
    # Quantize: round(x / scale), clamp to [-128, 127]
    x_scaled = x / scale
    x_rounded = tl.where(x_scaled >= 0, x_scaled + 0.5, x_scaled - 0.5)
    x_int = tl.maximum(tl.minimum(x_rounded, 127.0), -128.0)
    
    tl.store(output_ptr + offsets, x_int.to(tl.int8), mask=mask)


@triton.jit
def _dequantize_int8_kernel(
    x_ptr, scale_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Dequantize INT8 to FP32/FP16."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    scale = tl.load(scale_ptr)
    
    output = x * scale
    tl.store(output_ptr + offsets, output, mask=mask)


def quantize_int8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize tensor to INT8 with per-tensor scaling.
    
    Args:
        x: Input tensor (FP32 or FP16)
        
    Returns:
        Tuple of (quantized INT8 tensor, scale factor)
    """
    # Compute scale
    abs_max = x.abs().max()
    scale = abs_max / 127.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    
    output = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    n_elements = x.numel()
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    _quantize_int8_kernel[grid](
        x.contiguous(), scale, output,
        n_elements, BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output, scale


def dequantize_int8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize INT8 tensor back to float."""
    output = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    n_elements = x.numel()
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    _dequantize_int8_kernel[grid](
        x, scale, output, n_elements, BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


# =============================================================================
# INT8 Matrix Multiplication
# =============================================================================

@triton.jit
def _int8_matmul_kernel(
    A, B, C,
    scale_a, scale_b,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """INT8 matrix multiplication with dequantization."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    
    for k in range(0, K, BLOCK_K):
        a_mask = (offs_m[:, None] < M) & ((k + offs_k)[None, :] < K)
        b_mask = ((k + offs_k)[:, None] < K) & (offs_n[None, :] < N)
        
        a = tl.load(a_ptrs, mask=a_mask, other=0)
        b = tl.load(b_ptrs, mask=b_mask, other=0)
        
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # Dequantize result
    sa = tl.load(scale_a)
    sb = tl.load(scale_b)
    c = accumulator.to(tl.float32) * sa * sb
    
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def int8_matmul(
    a: torch.Tensor, scale_a: torch.Tensor,
    b: torch.Tensor, scale_b: torch.Tensor,
) -> torch.Tensor:
    """
    INT8 matrix multiplication with automatic dequantization.
    
    Args:
        a: INT8 matrix [M, K]
        scale_a: Scale for matrix a
        b: INT8 matrix [K, N]
        scale_b: Scale for matrix b
        
    Returns:
        FP32 result [M, N]
    """
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    
    c = torch.empty((M, N), dtype=torch.float32, device=a.device)
    
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    _int8_matmul_kernel[grid](
        a, b, c, scale_a, scale_b,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    
    return c


# =============================================================================
# INT4 Quantization (Packed)
# =============================================================================

@triton.jit
def _quantize_int4_kernel(
    x_ptr, scale_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Quantize to INT4 (packed as 2 values per byte)."""
    pid = tl.program_id(0)
    # Process 2 elements at a time (pack into 1 byte)
    offsets = pid * BLOCK_SIZE * 2 + tl.arange(0, BLOCK_SIZE) * 2
    mask = offsets < n_elements
    
    x0 = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x1 = tl.load(x_ptr + offsets + 1, mask=mask, other=0.0)
    scale = tl.load(scale_ptr)
    
    # Quantize to [-8, 7]
    q0 = tl.maximum(tl.minimum(x0 / scale + 0.5, 7.0), -8.0).to(tl.int8)
    q1 = tl.maximum(tl.minimum(x1 / scale + 0.5, 7.0), -8.0).to(tl.int8)
    
    # Pack: low nibble = q0, high nibble = q1
    packed = (q0 & 0xF) | ((q1 & 0xF) << 4)
    
    out_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    out_mask = out_offsets < (n_elements // 2)
    tl.store(output_ptr + out_offsets, packed, mask=out_mask)


@triton.jit
def _dequantize_int4_kernel(
    x_ptr, scale_ptr, output_ptr,
    n_packed,
    BLOCK_SIZE: tl.constexpr,
):
    """Dequantize INT4 (unpacks 2 values per byte)."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_packed
    
    packed = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.int32)
    scale = tl.load(scale_ptr)
    
    # Unpack
    q0 = (packed & 0xF).to(tl.float32)
    q1 = ((packed >> 4) & 0xF).to(tl.float32)
    
    # Sign extend from 4-bit
    q0 = tl.where(q0 > 7, q0 - 16, q0)
    q1 = tl.where(q1 > 7, q1 - 16, q1)
    
    # Dequantize
    out0 = q0 * scale
    out1 = q1 * scale
    
    out_offsets = pid * BLOCK_SIZE * 2 + tl.arange(0, BLOCK_SIZE) * 2
    out_mask = out_offsets < (n_packed * 2)
    tl.store(output_ptr + out_offsets, out0, mask=out_mask)
    tl.store(output_ptr + out_offsets + 1, out1, mask=out_mask)


def quantize_int4(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize tensor to INT4 (packed, 2 values per byte).
    
    50% memory reduction compared to INT8.
    """
    assert x.numel() % 2 == 0, "Tensor size must be even for INT4 packing"
    
    abs_max = x.abs().max()
    scale = abs_max / 7.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    
    n_packed = x.numel() // 2
    output = torch.empty(n_packed, dtype=torch.int8, device=x.device)
    
    BLOCK_SIZE = 512
    grid = (triton.cdiv(n_packed, BLOCK_SIZE),)
    
    _quantize_int4_kernel[grid](
        x.contiguous(), scale, output,
        x.numel(), BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output, scale


def dequantize_int4(x: torch.Tensor, scale: torch.Tensor, original_numel: int) -> torch.Tensor:
    """Dequantize INT4 packed tensor."""
    output = torch.empty(original_numel, dtype=torch.float32, device=x.device)
    
    BLOCK_SIZE = 512
    grid = (triton.cdiv(x.numel(), BLOCK_SIZE),)
    
    _dequantize_int4_kernel[grid](
        x, scale, output, x.numel(), BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


# =============================================================================
# Quantized Linear Layer
# =============================================================================

class QuantizedLinear:
    """
    Quantized linear layer for inference.
    
    Example:
        >>> linear = QuantizedLinear.from_float(nn.Linear(768, 3072))
        >>> output = linear(input)  # Uses INT8 matmul
    """
    
    def __init__(
        self,
        weight_int8: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        self.weight_int8 = weight_int8
        self.weight_scale = weight_scale
        self.bias = bias
    
    @classmethod
    def from_float(cls, linear: torch.nn.Linear, bits: int = 8) -> "QuantizedLinear":
        """Create quantized layer from float linear."""
        weight = linear.weight.data
        
        if bits == 8:
            weight_int8, scale = quantize_int8(weight)
        else:
            raise ValueError(f"Unsupported bits: {bits}")
        
        return cls(weight_int8, scale, linear.bias)
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize input
        x_int8, x_scale = quantize_int8(x)
        
        # INT8 matmul
        output = int8_matmul(
            x_int8, x_scale,
            self.weight_int8.t(), self.weight_scale,
        )
        
        if self.bias is not None:
            output = output + self.bias
        
        return output
