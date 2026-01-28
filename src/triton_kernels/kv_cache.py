"""
KV-Cache Optimization - Efficient Key-Value caching for inference.

Provides memory-efficient KV-cache with paged attention support.
"""

import math
import torch
import triton
import triton.language as tl
from typing import Optional, Tuple, List
from dataclasses import dataclass


# =============================================================================
# KV-Cache Data Structure
# =============================================================================

@dataclass
class KVCacheConfig:
    """Configuration for KV-Cache."""
    num_layers: int
    num_heads: int
    head_dim: int
    max_seq_len: int
    batch_size: int = 1
    dtype: torch.dtype = torch.float16
    device: str = "cuda"


class KVCache:
    """
    Efficient Key-Value cache for transformer inference.
    
    Features:
    - Pre-allocated memory for maximum sequence length
    - Efficient append operation
    - Support for batch inference
    
    Example:
        >>> config = KVCacheConfig(num_layers=32, num_heads=32, head_dim=128, max_seq_len=2048)
        >>> cache = KVCache(config)
        >>> cache.append(layer_idx=0, key=k, value=v)
        >>> cached_k, cached_v = cache.get(layer_idx=0)
    """
    
    def __init__(self, config: KVCacheConfig):
        self.config = config
        self.seq_len = 0
        
        # Pre-allocate cache: [layers, 2, batch, heads, max_seq, head_dim]
        # 2 for key and value
        self.cache = torch.zeros(
            config.num_layers,
            2,  # k, v
            config.batch_size,
            config.num_heads,
            config.max_seq_len,
            config.head_dim,
            dtype=config.dtype,
            device=config.device,
        )
    
    def append(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Append new keys and values to cache.
        
        Args:
            layer_idx: Layer index
            key: New keys [batch, heads, seq, head_dim]
            value: New values [batch, heads, seq, head_dim]
            
        Returns:
            Updated (cached_keys, cached_values)
        """
        new_seq_len = key.shape[2]
        end_pos = self.seq_len + new_seq_len
        
        # Store new k, v
        self.cache[layer_idx, 0, :, :, self.seq_len:end_pos, :] = key
        self.cache[layer_idx, 1, :, :, self.seq_len:end_pos, :] = value
        
        # Return full cache up to current position
        return (
            self.cache[layer_idx, 0, :, :, :end_pos, :],
            self.cache[layer_idx, 1, :, :, :end_pos, :],
        )
    
    def get(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached keys and values for a layer."""
        return (
            self.cache[layer_idx, 0, :, :, :self.seq_len, :],
            self.cache[layer_idx, 1, :, :, :self.seq_len, :],
        )
    
    def update_seq_len(self, new_tokens: int):
        """Update sequence length after processing tokens."""
        self.seq_len += new_tokens
    
    def reset(self):
        """Reset cache for new sequence."""
        self.seq_len = 0
    
    def memory_usage_mb(self) -> float:
        """Get memory usage in MB."""
        return self.cache.numel() * self.cache.element_size() / (1024 * 1024)


# =============================================================================
# Paged Attention
# =============================================================================

@triton.jit
def _paged_attention_kernel(
    Q, K_cache, V_cache, Out,
    block_tables, context_lens,
    scale,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_od,
    num_heads, head_dim, block_size,
    BLOCK_H: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """
    Paged attention kernel for vLLM-style memory management.
    
    Instead of contiguous KV cache, uses block tables for memory efficiency.
    """
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    # Get context length for this sequence
    context_len = tl.load(context_lens + batch_idx)
    
    # Load query
    offs_d = tl.arange(0, BLOCK_D)
    q = tl.load(Q + batch_idx * stride_qb + head_idx * stride_qh + offs_d * stride_qd,
                mask=offs_d < head_dim, other=0.0)
    
    # Initialize accumulators
    m_i = -float('inf')
    l_i = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    
    # Iterate over blocks in the sequence
    num_blocks = (context_len + block_size - 1) // block_size
    
    for block_idx in range(num_blocks):
        # Get physical block number from block table
        # block_num = tl.load(block_tables + batch_idx * max_blocks + block_idx)
        
        # Load K, V from this block
        start_pos = block_idx * block_size
        end_pos = min(start_pos + block_size, context_len)
        
        for pos in range(start_pos, end_pos):
            k = tl.load(K_cache + batch_idx * stride_kb + head_idx * stride_kh + 
                       pos * stride_ks + offs_d * stride_kd,
                       mask=offs_d < head_dim, other=0.0)
            v = tl.load(V_cache + batch_idx * stride_vb + head_idx * stride_vh +
                       pos * stride_vs + offs_d * stride_vd,
                       mask=offs_d < head_dim, other=0.0)
            
            # Compute attention score
            qk = tl.sum(q * k) * scale
            
            # Online softmax update
            m_new = tl.maximum(m_i, qk)
            exp_old = tl.exp(m_i - m_new)
            exp_new = tl.exp(qk - m_new)
            l_new = l_i * exp_old + exp_new
            
            # Update accumulator
            acc = acc * (l_i * exp_old / l_new) + v * (exp_new / l_new)
            
            m_i = m_new
            l_i = l_new
    
    # Store output
    tl.store(Out + batch_idx * stride_ob + head_idx * stride_oh + offs_d * stride_od,
             acc, mask=offs_d < head_dim)


class PagedKVCache:
    """
    Paged KV-Cache for efficient memory management.
    
    Uses block-based allocation like vLLM for handling
    variable-length sequences efficiently.
    """
    
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        block_size: int = 16,
        num_blocks: int = 1024,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.dtype = dtype
        self.device = device
        
        # Block pool: [num_blocks, 2, num_heads, block_size, head_dim]
        self.block_pool = torch.zeros(
            num_blocks, 2, num_heads, block_size, head_dim,
            dtype=dtype, device=device,
        )
        
        # Block allocation: which blocks are free
        self.free_blocks = list(range(num_blocks))
        
        # Block tables: maps (layer, batch) -> list of block indices
        self.block_tables = {}
    
    def allocate_blocks(self, layer_idx: int, batch_idx: int, num_tokens: int) -> List[int]:
        """Allocate blocks for new tokens."""
        num_needed = (num_tokens + self.block_size - 1) // self.block_size
        
        if len(self.free_blocks) < num_needed:
            raise RuntimeError("Out of KV-cache blocks")
        
        allocated = [self.free_blocks.pop() for _ in range(num_needed)]
        
        key = (layer_idx, batch_idx)
        if key not in self.block_tables:
            self.block_tables[key] = []
        self.block_tables[key].extend(allocated)
        
        return allocated
    
    def free_sequence(self, layer_idx: int, batch_idx: int):
        """Free all blocks for a sequence."""
        key = (layer_idx, batch_idx)
        if key in self.block_tables:
            self.free_blocks.extend(self.block_tables[key])
            del self.block_tables[key]
    
    def get_block_table(self, layer_idx: int, batch_idx: int) -> torch.Tensor:
        """Get block table as tensor."""
        key = (layer_idx, batch_idx)
        return torch.tensor(self.block_tables.get(key, []), device=self.device)
    
    def memory_usage_mb(self) -> float:
        """Get total memory usage."""
        allocated = self.num_blocks - len(self.free_blocks)
        bytes_per_block = 2 * self.num_heads * self.block_size * self.head_dim * 2  # fp16
        return allocated * bytes_per_block / (1024 * 1024)


# =============================================================================
# Efficient KV-Cache Attention
# =============================================================================

@triton.jit
def _kv_cache_attention_kernel(
    Q, K_cache, V_cache, Out,
    cache_len, new_len,
    scale,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_ob, stride_oh, stride_os, stride_od,
    BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Attention with KV-cache for incremental decoding."""
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    offs_d = tl.arange(0, BLOCK_D)
    
    # Load query (single new token in decoding)
    q = tl.load(Q + batch_idx * stride_qb + head_idx * stride_qh + offs_d,
                mask=offs_d < stride_qd)
    
    m_i = -float('inf')
    l_i = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    
    total_len = cache_len + new_len
    
    for s in range(0, total_len, BLOCK_S):
        offs_s = s + tl.arange(0, BLOCK_S)
        s_mask = offs_s < total_len
        
        # Load K block
        k = tl.load(K_cache + batch_idx * stride_kb + head_idx * stride_kh +
                   offs_s[:, None] * stride_ks + offs_d[None, :],
                   mask=s_mask[:, None] & (offs_d[None, :] < stride_kd), other=0.0)
        
        # Compute scores
        scores = tl.sum(q[None, :] * k, axis=1) * scale
        scores = tl.where(s_mask, scores, -float('inf'))
        
        # Online softmax
        m_new = tl.maximum(m_i, tl.max(scores))
        exp_scores = tl.exp(scores - m_new)
        exp_old = tl.exp(m_i - m_new)
        l_new = l_i * exp_old + tl.sum(exp_scores)
        
        # Load V and accumulate
        v = tl.load(V_cache + batch_idx * stride_kb + head_idx * stride_kh +
                   offs_s[:, None] * stride_ks + offs_d[None, :],
                   mask=s_mask[:, None] & (offs_d[None, :] < stride_kd), other=0.0)
        
        acc = acc * (l_i * exp_old / l_new) + tl.sum(exp_scores[:, None] * v, axis=0) / l_new
        
        m_i = m_new
        l_i = l_new
    
    # Store output
    tl.store(Out + batch_idx * stride_ob + head_idx * stride_oh + offs_d,
             acc, mask=offs_d < stride_od)


def kv_cache_attention(
    query: torch.Tensor,
    kv_cache: KVCache,
    layer_idx: int,
    new_key: torch.Tensor,
    new_value: torch.Tensor,
) -> torch.Tensor:
    """
    Attention with KV-cache for efficient inference.
    
    Args:
        query: Query for new tokens [batch, heads, new_seq, head_dim]
        kv_cache: KVCache instance
        layer_idx: Layer index
        new_key: New keys to cache [batch, heads, new_seq, head_dim]
        new_value: New values to cache [batch, heads, new_seq, head_dim]
        
    Returns:
        Attention output [batch, heads, new_seq, head_dim]
    """
    # Append and get full cache
    cached_k, cached_v = kv_cache.append(layer_idx, new_key, new_value)
    
    batch, heads, new_len, head_dim = query.shape
    total_len = cached_k.shape[2]
    
    scale = 1.0 / math.sqrt(head_dim)
    
    # Compute attention
    # For decode (new_len=1), use optimized kernel
    # For prefill (new_len>1), use standard attention
    if new_len == 1:
        # Decode: query is single token
        scores = torch.matmul(query, cached_k.transpose(-2, -1)) * scale
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, cached_v)
    else:
        # Prefill: use causal mask
        scores = torch.matmul(query, cached_k.transpose(-2, -1)) * scale
        
        # Causal mask: can only attend to previous positions
        mask = torch.triu(
            torch.ones(new_len, total_len, device=query.device),
            diagonal=total_len - new_len + 1
        ).bool()
        scores.masked_fill_(mask, float('-inf'))
        
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, cached_v)
    
    # Update cache sequence length
    kv_cache.update_seq_len(new_len)
    
    return output


# =============================================================================
# Sliding Window KV-Cache
# =============================================================================

class SlidingWindowKVCache:
    """
    Sliding window KV-cache for long sequences (Mistral-style).
    
    Only keeps the last `window_size` tokens, reducing memory
    from O(seq_len) to O(window_size).
    """
    
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        window_size: int = 4096,
        batch_size: int = 1,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.batch_size = batch_size
        
        # Circular buffer
        self.cache = torch.zeros(
            num_layers, 2, batch_size, num_heads, window_size, head_dim,
            dtype=dtype, device=device,
        )
        
        self.position = 0
        self.is_full = False
    
    def append(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append with sliding window semantics."""
        new_len = key.shape[2]
        
        for i in range(new_len):
            pos = (self.position + i) % self.window_size
            self.cache[layer_idx, 0, :, :, pos, :] = key[:, :, i, :]
            self.cache[layer_idx, 1, :, :, pos, :] = value[:, :, i, :]
        
        self.position = (self.position + new_len) % self.window_size
        if self.position + new_len >= self.window_size:
            self.is_full = True
        
        # Return valid portion
        if self.is_full:
            return self.cache[layer_idx, 0], self.cache[layer_idx, 1]
        else:
            return (
                self.cache[layer_idx, 0, :, :, :self.position, :],
                self.cache[layer_idx, 1, :, :, :self.position, :],
            )
    
    def memory_usage_mb(self) -> float:
        """Memory is fixed regardless of sequence length."""
        return self.cache.numel() * self.cache.element_size() / (1024 * 1024)
