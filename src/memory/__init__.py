"""
Memory Optimization Module - Efficient GPU memory management.

Provides:
- Memory pool management
- CPU/NVMe offloading
- Activation checkpointing
- Memory profiling
"""

from src.memory.pool import (
    MemoryPool,
    CachingAllocator,
    clear_memory_cache,
)
from src.memory.offload import (
    CPUOffloader,
    NVMeOffloader,
    OffloadConfig,
)
from src.memory.profiling import (
    MemoryProfiler,
    MemorySnapshot,
    get_memory_stats,
    print_memory_summary,
)

__all__ = [
    # Pool
    "MemoryPool",
    "CachingAllocator",
    "clear_memory_cache",
    # Offload
    "CPUOffloader",
    "NVMeOffloader",
    "OffloadConfig",
    # Profiling
    "MemoryProfiler",
    "MemorySnapshot",
    "get_memory_stats",
    "print_memory_summary",
]
