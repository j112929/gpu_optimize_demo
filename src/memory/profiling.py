"""
Memory Profiling - Track and analyze GPU memory usage.

Provides detailed memory profiling, snapshots, and visualization.
"""

import torch
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path


@dataclass
class MemorySnapshot:
    """Snapshot of GPU memory at a point in time."""
    timestamp: str
    name: str
    
    # Memory usage
    allocated_mb: float
    reserved_mb: float
    free_mb: float
    
    # Peak memory
    peak_allocated_mb: float
    peak_reserved_mb: float
    
    # Details
    num_tensors: int = 0
    largest_tensor_mb: float = 0.0
    
    def __repr__(self) -> str:
        return (
            f"MemorySnapshot(name='{self.name}', "
            f"allocated={self.allocated_mb:.1f}MB, "
            f"reserved={self.reserved_mb:.1f}MB, "
            f"peak={self.peak_allocated_mb:.1f}MB)"
        )


class MemoryProfiler:
    """
    GPU Memory Profiler for tracking memory usage over time.
    
    Features:
    - Named snapshots
    - Delta tracking
    - Peak detection
    - Export to JSON
    
    Example:
        >>> profiler = MemoryProfiler()
        >>> 
        >>> profiler.snapshot("before_forward")
        >>> output = model(input)
        >>> profiler.snapshot("after_forward")
        >>> 
        >>> print(profiler.delta("before_forward", "after_forward"))
    """
    
    def __init__(self, device: int = 0):
        self.device = device
        self.snapshots: Dict[str, MemorySnapshot] = {}
        self._snapshot_order: List[str] = []
    
    def snapshot(self, name: str) -> MemorySnapshot:
        """
        Take a memory snapshot.
        
        Args:
            name: Name for this snapshot
            
        Returns:
            MemorySnapshot object
        """
        if not torch.cuda.is_available():
            return MemorySnapshot(
                timestamp=datetime.now().isoformat(),
                name=name,
                allocated_mb=0,
                reserved_mb=0,
                free_mb=0,
                peak_allocated_mb=0,
                peak_reserved_mb=0,
            )
        
        torch.cuda.synchronize(self.device)
        
        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        
        total = torch.cuda.get_device_properties(self.device).total_memory
        free = total - reserved
        
        peak_allocated = torch.cuda.max_memory_allocated(self.device)
        peak_reserved = torch.cuda.max_memory_reserved(self.device)
        
        snap = MemorySnapshot(
            timestamp=datetime.now().isoformat(),
            name=name,
            allocated_mb=allocated / (1024 ** 2),
            reserved_mb=reserved / (1024 ** 2),
            free_mb=free / (1024 ** 2),
            peak_allocated_mb=peak_allocated / (1024 ** 2),
            peak_reserved_mb=peak_reserved / (1024 ** 2),
        )
        
        self.snapshots[name] = snap
        if name not in self._snapshot_order:
            self._snapshot_order.append(name)
        
        return snap
    
    def delta(
        self,
        before: str,
        after: str,
    ) -> Dict[str, float]:
        """
        Get memory delta between two snapshots.
        
        Args:
            before: Name of earlier snapshot
            after: Name of later snapshot
            
        Returns:
            Dict with delta values in MB
        """
        if before not in self.snapshots or after not in self.snapshots:
            return {"error": "Snapshot not found"}
        
        s1 = self.snapshots[before]
        s2 = self.snapshots[after]
        
        return {
            "allocated_delta_mb": s2.allocated_mb - s1.allocated_mb,
            "reserved_delta_mb": s2.reserved_mb - s1.reserved_mb,
            "peak_delta_mb": s2.peak_allocated_mb - s1.peak_allocated_mb,
        }
    
    def summary(self) -> Dict[str, Any]:
        """Get profiling summary."""
        if not self.snapshots:
            return {"message": "No snapshots taken"}
        
        all_allocated = [s.allocated_mb for s in self.snapshots.values()]
        all_peaks = [s.peak_allocated_mb for s in self.snapshots.values()]
        
        return {
            "num_snapshots": len(self.snapshots),
            "min_allocated_mb": min(all_allocated),
            "max_allocated_mb": max(all_allocated),
            "peak_allocated_mb": max(all_peaks),
            "snapshots": list(self.snapshots.keys()),
        }
    
    def reset(self):
        """Reset profiler and CUDA peak stats."""
        self.snapshots.clear()
        self._snapshot_order.clear()
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
    
    def export_json(self, path: str):
        """Export snapshots to JSON file."""
        data = {
            name: {
                "timestamp": snap.timestamp,
                "allocated_mb": snap.allocated_mb,
                "reserved_mb": snap.reserved_mb,
                "peak_allocated_mb": snap.peak_allocated_mb,
            }
            for name, snap in self.snapshots.items()
        }
        
        Path(path).write_text(json.dumps(data, indent=2))
    
    def print_summary(self):
        """Print formatted summary."""
        print("\n" + "=" * 60)
        print("MEMORY PROFILER SUMMARY")
        print("=" * 60)
        
        if not self.snapshots:
            print("No snapshots taken")
            return
        
        print(f"{'Snapshot':<20} {'Allocated':<12} {'Reserved':<12} {'Peak':<12}")
        print("-" * 60)
        
        for name in self._snapshot_order:
            snap = self.snapshots[name]
            print(
                f"{name:<20} "
                f"{snap.allocated_mb:>8.1f} MB "
                f"{snap.reserved_mb:>8.1f} MB "
                f"{snap.peak_allocated_mb:>8.1f} MB"
            )
        
        print("=" * 60)


def get_memory_stats(device: int = 0) -> Dict[str, float]:
    """
    Get current GPU memory statistics.
    
    Returns:
        Dict with memory values in MB
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    return {
        "allocated_mb": torch.cuda.memory_allocated(device) / (1024 ** 2),
        "reserved_mb": torch.cuda.memory_reserved(device) / (1024 ** 2),
        "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
        "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024 ** 2),
        "total_mb": torch.cuda.get_device_properties(device).total_memory / (1024 ** 2),
    }


def print_memory_summary(device: int = 0):
    """Print current memory summary."""
    stats = get_memory_stats(device)
    
    if "error" in stats:
        print(stats["error"])
        return
    
    print("\n" + "=" * 40)
    print("GPU MEMORY SUMMARY")
    print("=" * 40)
    print(f"Allocated:      {stats['allocated_mb']:>8.1f} MB")
    print(f"Reserved:       {stats['reserved_mb']:>8.1f} MB")
    print(f"Peak Allocated: {stats['peak_allocated_mb']:>8.1f} MB")
    print(f"Total:          {stats['total_mb']:>8.1f} MB")
    print(f"Utilization:    {100 * stats['allocated_mb'] / stats['total_mb']:>7.1f} %")
    print("=" * 40)


# =============================================================================
# Tensor Memory Analysis
# =============================================================================

def get_tensor_memory_stats() -> Dict[str, Any]:
    """
    Get detailed tensor memory statistics.
    
    Shows which tensors are using memory.
    """
    import gc
    
    tensor_info = []
    
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                tensor_info.append({
                    "shape": tuple(obj.shape),
                    "dtype": str(obj.dtype),
                    "size_mb": obj.numel() * obj.element_size() / (1024 ** 2),
                    "device": str(obj.device),
                })
        except:
            pass
    
    # Sort by size
    tensor_info.sort(key=lambda x: x["size_mb"], reverse=True)
    
    total_mb = sum(t["size_mb"] for t in tensor_info)
    
    return {
        "num_tensors": len(tensor_info),
        "total_mb": total_mb,
        "largest_tensors": tensor_info[:10],
    }


def find_memory_leaks() -> List[Dict]:
    """
    Find potential memory leaks.
    
    Returns list of tensors that might be leaked.
    """
    import gc
    
    gc.collect()
    torch.cuda.synchronize()
    
    orphan_tensors = []
    
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                # Check if tensor has no Python references
                if len(gc.get_referrers(obj)) <= 2:  # gc list + our check
                    orphan_tensors.append({
                        "shape": tuple(obj.shape),
                        "dtype": str(obj.dtype),
                        "size_mb": obj.numel() * obj.element_size() / (1024 ** 2),
                    })
        except:
            pass
    
    return orphan_tensors


# =============================================================================
# Context Managers
# =============================================================================

class track_memory:
    """
    Context manager to track memory usage of a code block.
    
    Example:
        >>> with track_memory("forward_pass") as tracker:
        ...     output = model(input)
        >>> print(f"Used {tracker.delta_mb:.1f} MB")
    """
    
    def __init__(self, name: str = "block"):
        self.name = name
        self.start_mb = 0.0
        self.end_mb = 0.0
        self.delta_mb = 0.0
        self.peak_mb = 0.0
    
    def __enter__(self):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.start_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        return self
    
    def __exit__(self, *args):
        torch.cuda.synchronize()
        self.end_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        self.delta_mb = self.end_mb - self.start_mb
        self.peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    def __repr__(self) -> str:
        return (
            f"MemoryTrack(name='{self.name}', "
            f"delta={self.delta_mb:.1f}MB, "
            f"peak={self.peak_mb:.1f}MB)"
        )
