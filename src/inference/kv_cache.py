"""
KV Cache Management with Prefix Caching (Radix Attention).

Provides:
- KVCacheManager: Manages KV cache allocation and eviction
- RadixTree: Data structure for efficient prefix matching and reuse
- PagedMemory: Simulates paged memory management for KV blocks

Key benefits:
- Reuses KV cache for shared prefixes (System prompts, few-shot examples)
- Reduces latency for multi-turn chat and agentic workflows
"""

import torch
import collections
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
import time

@dataclass
class CacheBlock:
    """A single block of KV cache memory."""
    block_id: int
    size: int
    is_full: bool = False
    ref_count: int = 0
    last_accessed: float = 0.0

class RadixNode:
    """Node in the Radix Tree for Prefix Caching."""
    def __init__(self, key: Tuple[int, ...]):
        self.key = key  # Tuple of token IDs
        self.children: Dict[int, 'RadixNode'] = {} # Map from first token of key to child
        self.block_indices: List[int] = [] # Indices of KV blocks storing this sequence
        self.parent: Optional['RadixNode'] = None
        self.last_accessed: float = time.time()
        self.value: Any = None # Optional payload

class RadixCache:
    """
    Radix Tree for managing Prefix Caching.
    Allows mapping token sequences to cached KV blocks.
    """
    def __init__(self):
        self.root = RadixNode(())
        self.nodes_pool = {} # Track all nodes for eviction policies
    
    def insert(self, tokens: List[int], value: Any = None):
        """Insert a token sequence into the tree."""
        node = self.root
        
        # Simplified insertion (Trie style for demo)
        for token in tokens:
            if token not in node.children:
                child = RadixNode((token,))
                child.parent = node
                node.children[token] = child
            node = node.children[token]
            node.last_accessed = time.time()
            
        node.value = value
        return node

    def match_prefix(self, tokens: List[int]) -> Tuple[List[int], Any]:
        """
        Find the longest cached prefix for the given tokens.
        Returns: (cached_tokens, cached_value)
        """
        node = self.root
        matched_tokens = []
        last_value = None
        
        for token in tokens:
            if token in node.children:
                node = node.children[token]
                node.last_accessed = time.time()
                matched_tokens.append(token)
                if node.value is not None:
                    last_value = node.value
            else:
                break
                
        return matched_tokens, last_value

class PagedKVCache:
    """
    Simulates Paged KV Cache memory management.
    """
    def __init__(self, num_blocks: int, block_size: int, hidden_dim: int, num_layers: int):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.free_blocks = set(range(num_blocks))
        self.used_blocks: Dict[int, CacheBlock] = {}
        
        # Simulate GPU memory (just metadata here)
        self.memory_usage = 0
    
    def allocate(self, num_needed: int) -> List[int]:
        """Allocate 'num_needed' blocks."""
        if len(self.free_blocks) < num_needed:
            raise MemoryError("Out of KV cache memory")
            
        allocated = []
        for _ in range(num_needed):
            bid = self.free_blocks.pop()
            block = CacheBlock(block_id=bid, size=self.block_size)
            block.ref_count = 1
            block.last_accessed = time.time()
            self.used_blocks[bid] = block
            allocated.append(bid)
            
        return allocated

    def free(self, block_ids: List[int]):
        """Free blocks, decrementing ref count."""
        for bid in block_ids:
            if bid in self.used_blocks:
                block = self.used_blocks[bid]
                block.ref_count -= 1
                if block.ref_count <= 0:
                    del self.used_blocks[bid]
                    self.free_blocks.add(bid)
                    
    def get_token_capacity(self):
        return self.num_blocks * self.block_size


class PrefixCacheManager:
    """
    Manager that combines Radix Tree and Paged KV Cache.
    """
    def __init__(self, block_size=16, max_blocks=1000):
        self.radix_tree = RadixCache()
        self.paged_cache = PagedKVCache(max_blocks, block_size, 4096, 32)
        self.block_size = block_size
        
        # Stats
        self.hits = 0
        self.misses = 0
        
    def check_cache(self, input_ids: List[int]):
        """Check for existing prefix in cache."""
        cached_tokens, blocks = self.radix_tree.match_prefix(input_ids)
        
        if not cached_tokens:
            self.misses += 1
            return 0, []
        
        self.hits += 1
        return len(cached_tokens), blocks

    def cache_request(self, input_ids: List[int], blocks: List[int]):
        """Cache a processed request."""
        # Insert into radix tree
        self.radix_tree.insert(input_ids, blocks)
        
        # Increment ref counts for blocks
        # (In a real system, we'd need complex ref-counting logic)
        for bid in blocks:
            if bid in self.paged_cache.used_blocks:
                self.paged_cache.used_blocks[bid].ref_count += 1

                
class FP8KVCache(PagedKVCache):
    """
    FP8 Quantized KV Cache.
    
    Reduces memory usage by 2x-4x compared to FP16/BF16.
    Simulates E4M3/E5M2 encoding.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantization_scale = 1.0
        
    def quantize_block(self, key_block: torch.Tensor, value_block: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize K/V blocks to FP8.
        
        Args:
            key_block: [num_tokens, num_heads, head_dim]
            value_block: [num_tokens, num_heads, head_dim]
        """
        # Simulation of FP8 casting
        # In real scenario: usage of torch.float8_e4m3fn
        if hasattr(torch, 'float8_e4m3fn'):
            k_fp8 = key_block.to(torch.float8_e4m3fn)
            v_fp8 = value_block.to(torch.float8_e4m3fn)
            return k_fp8, v_fp8
        else:
            # Fallback simulation: reduced range clamping
            # e4m3 has range [-448, 448] roughly
            k_clamped = torch.clamp(key_block, -448, 448)
            v_clamped = torch.clamp(value_block, -448, 448)
            # Add quantization noise or just return clamped
            return k_clamped, v_clamped
            
    def get_memory_usage(self) -> float:
        """Get memory usage in MB (assuming FP8)."""
        num_used = len(self.used_blocks)
        # 1 byte per element for FP8
        bytes_per_block = self.block_size * 4096 * 2  # simplified: hidden_dim * 2 (K+V)
        return (num_used * bytes_per_block) / (1024**2)

