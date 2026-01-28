"""
KV-Cache Optimization - Efficient key-value caching for LLMs.

Provides:
- Standard KV-Cache management
- Paged Attention (vLLM-style)
- Sliding Window Cache
- Prefix Caching
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import math


@dataclass
class KVCacheConfig:
    """Configuration for KV-Cache."""
    # Dimensions
    num_layers: int = 32
    num_heads: int = 32
    head_dim: int = 128
    
    # Size
    max_batch_size: int = 32
    max_sequence_length: int = 4096
    
    # Optimization
    dtype: torch.dtype = torch.float16
    
    # Paging (for PagedKVCache)
    page_size: int = 16
    num_pages: int = 1024


class KVCache:
    """
    Standard KV-Cache for transformer inference.
    
    Stores key-value pairs from previous tokens to avoid
    recomputation during autoregressive generation.
    
    Example:
        >>> cache = KVCache(config)
        >>> 
        >>> # First token
        >>> k, v = attention(x)
        >>> cache.update(layer_idx=0, key=k, value=v)
        >>> 
        >>> # Next tokens
        >>> past_k, past_v = cache.get(layer_idx=0)
        >>> full_k = torch.cat([past_k, k], dim=2)
    """
    
    def __init__(self, config: KVCacheConfig):
        self.config = config
        
        # Initialize cache tensors
        # Shape: [batch, num_heads, seq_len, head_dim]
        cache_shape = (
            config.max_batch_size,
            config.num_heads,
            config.max_sequence_length,
            config.head_dim,
        )
        
        self.key_cache = [
            torch.zeros(cache_shape, dtype=config.dtype)
            for _ in range(config.num_layers)
        ]
        self.value_cache = [
            torch.zeros(cache_shape, dtype=config.dtype)
            for _ in range(config.num_layers)
        ]
        
        # Track sequence lengths per batch
        self.seq_lengths = torch.zeros(config.max_batch_size, dtype=torch.long)
    
    def to(self, device: torch.device) -> "KVCache":
        """Move cache to device."""
        for i in range(len(self.key_cache)):
            self.key_cache[i] = self.key_cache[i].to(device)
            self.value_cache[i] = self.value_cache[i].to(device)
        self.seq_lengths = self.seq_lengths.to(device)
        return self
    
    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        batch_indices: Optional[torch.Tensor] = None,
    ):
        """
        Update cache with new key-value pairs.
        
        Args:
            layer_idx: Layer index
            key: New keys [batch, heads, new_seq, dim]
            value: New values [batch, heads, new_seq, dim]
            batch_indices: Which batch elements to update
        """
        batch_size = key.size(0)
        new_seq_len = key.size(2)
        
        if batch_indices is None:
            batch_indices = torch.arange(batch_size, device=key.device)
        
        for i, batch_idx in enumerate(batch_indices):
            seq_pos = self.seq_lengths[batch_idx].item()
            end_pos = seq_pos + new_seq_len
            
            self.key_cache[layer_idx][batch_idx, :, seq_pos:end_pos] = key[i]
            self.value_cache[layer_idx][batch_idx, :, seq_pos:end_pos] = value[i]
        
        # Update sequence lengths
        self.seq_lengths[batch_indices] += new_seq_len
    
    def get(
        self,
        layer_idx: int,
        batch_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cached key-value pairs.
        
        Returns:
            Tuple of (keys, values) for specified batch elements
        """
        if batch_indices is None:
            batch_indices = torch.arange(
                self.config.max_batch_size,
                device=self.key_cache[layer_idx].device,
            )
        
        keys = []
        values = []
        
        for batch_idx in batch_indices:
            seq_len = self.seq_lengths[batch_idx].item()
            keys.append(self.key_cache[layer_idx][batch_idx, :, :seq_len])
            values.append(self.value_cache[layer_idx][batch_idx, :, :seq_len])
        
        # Pad to same length
        max_len = max(k.size(1) for k in keys)
        
        padded_keys = torch.zeros(
            len(keys), self.config.num_heads, max_len, self.config.head_dim,
            dtype=self.config.dtype, device=keys[0].device,
        )
        padded_values = torch.zeros_like(padded_keys)
        
        for i, (k, v) in enumerate(zip(keys, values)):
            padded_keys[i, :, :k.size(1)] = k
            padded_values[i, :, :v.size(1)] = v
        
        return padded_keys, padded_values
    
    def clear(self, batch_indices: Optional[torch.Tensor] = None):
        """Clear cache for specified batch elements."""
        if batch_indices is None:
            for i in range(len(self.key_cache)):
                self.key_cache[i].zero_()
                self.value_cache[i].zero_()
            self.seq_lengths.zero_()
        else:
            for batch_idx in batch_indices:
                for i in range(len(self.key_cache)):
                    self.key_cache[i][batch_idx].zero_()
                    self.value_cache[i][batch_idx].zero_()
                self.seq_lengths[batch_idx] = 0
    
    def memory_usage_mb(self) -> float:
        """Get memory usage in MB."""
        bytes_per_element = 2 if self.config.dtype == torch.float16 else 4
        total_elements = (
            2 * self.config.num_layers *
            self.config.max_batch_size *
            self.config.num_heads *
            self.config.max_sequence_length *
            self.config.head_dim
        )
        return total_elements * bytes_per_element / (1024 ** 2)


class PagedKVCache:
    """
    Paged KV-Cache (vLLM-style) for memory efficiency.
    
    Uses paging to:
    - Reduce memory fragmentation
    - Enable dynamic memory allocation
    - Support longer sequences
    
    Example:
        >>> cache = PagedKVCache(config)
        >>> 
        >>> # Allocate pages for a request
        >>> block_ids = cache.allocate(request_id, num_tokens)
        >>> 
        >>> # Store KV
        >>> cache.update(layer_idx, block_ids, positions, keys, values)
    """
    
    def __init__(self, config: KVCacheConfig):
        self.config = config
        
        # Physical pages
        # Shape: [num_pages, 2 (k/v), num_heads, page_size, head_dim]
        page_shape = (
            config.num_pages,
            config.num_layers,
            2,  # Key and Value
            config.num_heads,
            config.page_size,
            config.head_dim,
        )
        
        self.pages = torch.zeros(page_shape, dtype=config.dtype)
        
        # Page table: maps request -> list of page indices
        self.page_table: Dict[str, List[int]] = {}
        
        # Free pages
        self.free_pages = list(range(config.num_pages))
    
    def to(self, device: torch.device) -> "PagedKVCache":
        """Move cache to device."""
        self.pages = self.pages.to(device)
        return self
    
    def allocate(
        self,
        request_id: str,
        num_tokens: int,
    ) -> List[int]:
        """
        Allocate pages for a request.
        
        Returns:
            List of allocated page indices
        """
        num_pages_needed = math.ceil(num_tokens / self.config.page_size)
        
        if len(self.free_pages) < num_pages_needed:
            raise RuntimeError(f"Not enough free pages: need {num_pages_needed}, have {len(self.free_pages)}")
        
        allocated = []
        for _ in range(num_pages_needed):
            page_idx = self.free_pages.pop(0)
            allocated.append(page_idx)
        
        self.page_table[request_id] = allocated
        
        return allocated
    
    def free(self, request_id: str):
        """Free pages for a request."""
        if request_id in self.page_table:
            pages = self.page_table.pop(request_id)
            self.free_pages.extend(pages)
    
    def update(
        self,
        layer_idx: int,
        request_id: str,
        positions: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ):
        """Update cache at specified positions."""
        if request_id not in self.page_table:
            raise KeyError(f"Request {request_id} not found in page table")
        
        page_indices = self.page_table[request_id]
        
        for i, pos in enumerate(positions):
            page_idx = page_indices[pos // self.config.page_size]
            offset = pos % self.config.page_size
            
            self.pages[page_idx, layer_idx, 0, :, offset] = keys[i]
            self.pages[page_idx, layer_idx, 1, :, offset] = values[i]
    
    def get(
        self,
        layer_idx: int,
        request_id: str,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached values at positions."""
        page_indices = self.page_table[request_id]
        
        keys = []
        values = []
        
        for pos in positions:
            page_idx = page_indices[pos // self.config.page_size]
            offset = pos % self.config.page_size
            
            keys.append(self.pages[page_idx, layer_idx, 0, :, offset])
            values.append(self.pages[page_idx, layer_idx, 1, :, offset])
        
        return torch.stack(keys), torch.stack(values)
    
    def utilization(self) -> float:
        """Get page utilization."""
        used = self.config.num_pages - len(self.free_pages)
        return used / self.config.num_pages


# =============================================================================
# Sliding Window Cache
# =============================================================================

class SlidingWindowCache:
    """
    Sliding Window KV-Cache for long sequences.
    
    Only keeps the most recent tokens in cache,
    reducing memory for very long sequences.
    """
    
    def __init__(
        self,
        window_size: int = 4096,
        config: Optional[KVCacheConfig] = None,
    ):
        self.window_size = window_size
        self.config = config or KVCacheConfig()
        
        # Circular buffer
        cache_shape = (
            self.config.max_batch_size,
            self.config.num_heads,
            window_size,
            self.config.head_dim,
        )
        
        self.key_cache = [
            torch.zeros(cache_shape, dtype=self.config.dtype)
            for _ in range(self.config.num_layers)
        ]
        self.value_cache = [
            torch.zeros(cache_shape, dtype=self.config.dtype)
            for _ in range(self.config.num_layers)
        ]
        
        # Current position in circular buffer
        self.positions = torch.zeros(self.config.max_batch_size, dtype=torch.long)
        self.total_tokens = torch.zeros(self.config.max_batch_size, dtype=torch.long)
    
    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        batch_idx: int = 0,
    ):
        """Update cache with new token (circular)."""
        pos = self.positions[batch_idx].item() % self.window_size
        
        self.key_cache[layer_idx][batch_idx, :, pos] = key.squeeze()
        self.value_cache[layer_idx][batch_idx, :, pos] = value.squeeze()
        
        self.positions[batch_idx] += 1
        self.total_tokens[batch_idx] += 1
    
    def get(
        self,
        layer_idx: int,
        batch_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached values in correct order."""
        total = min(self.total_tokens[batch_idx].item(), self.window_size)
        
        if total < self.window_size:
            # Not wrapped yet
            return (
                self.key_cache[layer_idx][batch_idx, :, :total],
                self.value_cache[layer_idx][batch_idx, :, :total],
            )
        
        # Need to reorder for wrapped buffer
        pos = self.positions[batch_idx].item() % self.window_size
        
        # Concatenate: [pos:end] + [0:pos]
        key = torch.cat([
            self.key_cache[layer_idx][batch_idx, :, pos:],
            self.key_cache[layer_idx][batch_idx, :, :pos],
        ], dim=1)
        
        value = torch.cat([
            self.value_cache[layer_idx][batch_idx, :, pos:],
            self.value_cache[layer_idx][batch_idx, :, :pos],
        ], dim=1)
        
        return key, value


# =============================================================================
# Prefix Cache
# =============================================================================

class PrefixCache:
    """
    Prefix Cache for sharing KV-cache across requests.
    
    Caches common prefixes (system prompts, few-shot examples)
    to avoid recomputation.
    """
    
    def __init__(self, config: KVCacheConfig):
        self.config = config
        
        # Cached prefixes: hash -> (keys, values)
        self._cache: Dict[str, Tuple[List[torch.Tensor], List[torch.Tensor]]] = {}
    
    def cache_prefix(
        self,
        prefix_tokens: torch.Tensor,
        keys: List[torch.Tensor],
        values: List[torch.Tensor],
    ) -> str:
        """Cache a prefix's KV values."""
        # Create hash from tokens
        prefix_hash = hash(tuple(prefix_tokens.tolist()))
        prefix_key = f"prefix_{prefix_hash}"
        
        self._cache[prefix_key] = (
            [k.clone() for k in keys],
            [v.clone() for v in values],
        )
        
        return prefix_key
    
    def get_prefix(
        self,
        prefix_key: str,
    ) -> Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        """Get cached prefix KV values."""
        return self._cache.get(prefix_key)
    
    def clear(self):
        """Clear all cached prefixes."""
        self._cache.clear()
