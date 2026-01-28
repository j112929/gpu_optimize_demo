"""
Triton Attention Kernels - High-performance attention implementations.

Includes Flash Attention V2 style implementation.
"""

import math
import torch
import triton
import triton.language as tl
from typing import Optional


@triton.jit
def _flash_attention_fwd_kernel(
    Q, K, V, Out, Lse,
    sm_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """Flash Attention forward kernel."""
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    off_q = off_z * stride_qz + off_h * stride_qh
    off_k = off_z * stride_kz + off_h * stride_kh
    off_v = off_z * stride_vz + off_h * stride_vh
    
    q_ptrs = Q + off_q + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + off_k + offs_d[:, None] * stride_kk + offs_n[None, :] * stride_kn
    v_ptrs = V + off_v + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    q_mask = offs_m[:, None] < N_CTX
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)
    
    end_n = N_CTX if not IS_CAUSAL else min((start_m + 1) * BLOCK_M, N_CTX)
    
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_mask = (start_n + offs_n)[None, :] < N_CTX
        k = tl.load(k_ptrs + start_n * stride_kn, mask=k_mask, other=0.0)
        v_mask = (start_n + offs_n)[:, None] < N_CTX
        v = tl.load(v_ptrs + start_n * stride_vn, mask=v_mask, other=0.0)
        
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale
        
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], qk, float('-inf'))
        
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        p = tl.exp(qk - m_new[:, None])
        l_new = tl.exp(m_i - m_new) * l_i + tl.sum(p, axis=1)
        acc = acc * (tl.exp(m_i - m_new) * l_i)[:, None] / l_new[:, None]
        acc += tl.dot(p.to(v.dtype), v) / l_new[:, None]
        m_i = m_new
        l_i = l_new
    
    off_o = off_z * stride_oz + off_h * stride_oh
    o_ptrs = Out + off_o + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_CTX)
    
    lse_ptrs = Lse + off_hz * N_CTX + offs_m
    tl.store(lse_ptrs, m_i + tl.log(l_i), mask=offs_m < N_CTX)


def flash_attention_v2(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    causal: bool = False, sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Flash Attention V2 using Triton. O(N) memory complexity.
    
    Args:
        q, k, v: [batch, heads, seq_len, head_dim]
        causal: Apply causal masking
        sm_scale: Softmax scale (default: 1/sqrt(head_dim))
    """
    batch, heads, seq_len, head_dim = q.shape
    sm_scale = sm_scale or 1.0 / math.sqrt(head_dim)
    
    o = torch.empty_like(q)
    lse = torch.empty((batch * heads, seq_len), device=q.device, dtype=torch.float32)
    
    BLOCK_M, BLOCK_N = 128, 64
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * heads)
    
    _flash_attention_fwd_kernel[grid](
        q, k, v, o, lse, sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        batch, heads, seq_len,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=head_dim, IS_CAUSAL=causal,
    )
    return o


def multi_head_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None, scale: Optional[float] = None,
) -> torch.Tensor:
    """Multi-head attention using Flash Attention for long sequences."""
    batch, heads, seq_len, head_dim = q.shape
    scale = scale or 1.0 / math.sqrt(head_dim)
    
    if seq_len >= 512:
        return flash_attention_v2(q, k, v, causal=False, sm_scale=scale)
    
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask
    attn_weights = torch.softmax(attn_weights, dim=-1)
    return torch.matmul(attn_weights, v)


def grouped_query_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    num_kv_heads: int, causal: bool = True,
) -> torch.Tensor:
    """GQA: fewer KV heads than query heads for efficiency."""
    num_q_heads = q.shape[1]
    num_groups = num_q_heads // num_kv_heads
    k = k.repeat_interleave(num_groups, dim=1)
    v = v.repeat_interleave(num_groups, dim=1)
    return flash_attention_v2(q, k, v, causal=causal)
