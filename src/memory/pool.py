"""
Memory Pool - Efficient GPU memory allocation.

Provides memory pooling to reduce allocation overhead
and memory fragmentation.
"""

import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import threading


@dataclass  
class PoolConfig:
    """Configuration for memory pool."""
    # Pool sizes
    small_tensor_threshold: int = 1024 * 1024  # 1MB
    large_tensor_threshold: int = 256 * 1024 * 1024  # 256MB
    
    # Caching
    max_cached_tensors: int = 128
    max_cached_memory: int = 2 * 1024 * 1024 * 1024  # 2GB
    
    # Behavior
    enable_caching: bool = True
    round_up_power_of_2: bool = True


class MemoryPool:
    """
    GPU Memory Pool for efficient tensor allocation.
    
    Reduces allocation overhead by reusing tensor buffers.
    Particularly useful for inference with fixed shapes.
    
    Example:
        >>> pool = MemoryPool()
        >>> 
        >>> # Allocate from pool
        >>> tensor = pool.allocate((1024, 1024), dtype=torch.float32)
        >>> 
        >>> # Return to pool
        >>> pool.release(tensor)
        >>> 
        >>> # Next allocation reuses buffer
        >>> tensor2 = pool.allocate((1024, 1024), dtype=torch.float32)
    """
    
    def __init__(self, config: Optional[PoolConfig] = None):
        self.config = config or PoolConfig()
        self._pools: Dict[Tuple, List[torch.Tensor]] = {}
        self._lock = threading.Lock()
        self._total_cached = 0
        self._hits = 0
        self._misses = 0
    
    def _get_key(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple:
        """Get pool key for shape/dtype/device."""
        if self.config.round_up_power_of_2:
            # Round up to power of 2 for better reuse
            shape = tuple(self._round_up_power_2(s) for s in shape)
        return (shape, dtype, device)
    
    @staticmethod
    def _round_up_power_2(n: int) -> int:
        """Round up to next power of 2."""
        if n <= 0:
            return 1
        n -= 1
        n |= n >> 1
        n |= n >> 2
        n |= n >> 4
        n |= n >> 8
        n |= n >> 16
        return n + 1
    
    def allocate(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Allocate tensor from pool.
        
        Args:
            shape: Tensor shape
            dtype: Data type
            device: Device (default: cuda)
            
        Returns:
            Tensor from pool or newly allocated
        """
        device = device or torch.device("cuda")
        key = self._get_key(shape, dtype, device)
        
        with self._lock:
            if key in self._pools and self._pools[key]:
                tensor = self._pools[key].pop()
                self._total_cached -= tensor.numel() * tensor.element_size()
                self._hits += 1
                
                # Resize if needed (when using power-of-2 rounding)
                if tensor.shape != shape:
                    tensor = tensor.view(-1)[:torch.tensor(shape).prod().item()].view(shape)
                
                return tensor
            
            self._misses += 1
        
        return torch.empty(shape, dtype=dtype, device=device)
    
    def release(self, tensor: torch.Tensor):
        """
        Return tensor to pool.
        
        Args:
            tensor: Tensor to return
        """
        if not self.config.enable_caching:
            return
        
        key = self._get_key(
            tuple(tensor.shape),
            tensor.dtype,
            tensor.device,
        )
        
        tensor_size = tensor.numel() * tensor.element_size()
        
        with self._lock:
            # Check if we have room
            if self._total_cached + tensor_size > self.config.max_cached_memory:
                return  # Don't cache, let it be freed
            
            if key not in self._pools:
                self._pools[key] = []
            
            if len(self._pools[key]) < self.config.max_cached_tensors:
                self._pools[key].append(tensor.detach())
                self._total_cached += tensor_size
    
    def clear(self):
        """Clear all cached tensors."""
        with self._lock:
            self._pools.clear()
            self._total_cached = 0
    
    @property
    def stats(self) -> Dict[str, float]:
        """Get pool statistics."""
        total_requests = self._hits + self._misses
        return {
            "cached_memory_mb": self._total_cached / (1024 ** 2),
            "num_cached_tensors": sum(len(p) for p in self._pools.values()),
            "hit_rate": self._hits / max(1, total_requests),
            "hits": self._hits,
            "misses": self._misses,
        }


class CachingAllocator:
    """
    Context manager for PyTorch's caching allocator.
    
    Provides fine-grained control over memory allocation behavior.
    
    Example:
        >>> with CachingAllocator(max_split_size_mb=256):
        ...     # Allocations here use the configured settings
        ...     tensor = torch.randn(10000, 10000, device="cuda")
    """
    
    def __init__(
        self,
        max_split_size_mb: Optional[int] = None,
        garbage_collection_threshold: Optional[float] = None,
    ):
        self.max_split_size_mb = max_split_size_mb
        self.gc_threshold = garbage_collection_threshold
        self._original_settings = {}
    
    def __enter__(self):
        # Save original settings
        if self.max_split_size_mb is not None:
            # Set via environment variable (must be set before CUDA init)
            import os
            self._original_settings["PYTORCH_CUDA_ALLOC_CONF"] = os.environ.get(
                "PYTORCH_CUDA_ALLOC_CONF", ""
            )
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
                f"max_split_size_mb:{self.max_split_size_mb}"
            )
        
        return self
    
    def __exit__(self, *args):
        # Restore original settings
        import os
        if "PYTORCH_CUDA_ALLOC_CONF" in self._original_settings:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = self._original_settings[
                "PYTORCH_CUDA_ALLOC_CONF"
            ]


def clear_memory_cache():
    """
    Clear CUDA memory cache.
    
    Frees cached memory but may cause fragmentation if called frequently.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def reset_peak_memory():
    """Reset peak memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def memory_efficient_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """
    Memory-efficient attention using chunking.
    
    Processes attention in chunks to reduce peak memory.
    O(N) memory instead of O(N²).
    """
    batch_size, num_heads, seq_len, head_dim = query.shape
    output = torch.zeros_like(query)
    
    for i in range(0, seq_len, chunk_size):
        chunk_end = min(i + chunk_size, seq_len)
        q_chunk = query[:, :, i:chunk_end]
        
        # Compute attention for this chunk
        attn = torch.matmul(q_chunk, key.transpose(-2, -1)) / (head_dim ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        output[:, :, i:chunk_end] = torch.matmul(attn, value)
    
    return output


# =============================================================================
# Memory Pinning
# =============================================================================

def pin_memory(tensor: torch.Tensor) -> torch.Tensor:
    """Pin tensor memory for faster CPU-GPU transfer."""
    if tensor.device.type == "cpu" and not tensor.is_pinned():
        return tensor.pin_memory()
    return tensor


def create_pinned_buffer(
    shape: Tuple[int, ...],
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a pinned memory buffer."""
    return torch.empty(shape, dtype=dtype, pin_memory=True)
