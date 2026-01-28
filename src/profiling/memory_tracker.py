"""
Memory Tracker - GPU memory profiling and analysis.

This module provides tools for tracking GPU memory usage, detecting memory leaks,
and analyzing allocation patterns.
"""

import gc
import functools
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch


@dataclass
class MemorySnapshot:
    """Snapshot of GPU memory state at a point in time."""
    
    timestamp: str
    label: str
    
    # Current memory
    allocated_mb: float
    reserved_mb: float
    
    # Peak memory
    peak_allocated_mb: float
    peak_reserved_mb: float
    
    # Free memory
    free_mb: float
    
    # Tensor info
    num_tensors: int = 0
    tensor_sizes: List[Tuple[str, float]] = field(default_factory=list)
    
    def __repr__(self) -> str:
        return (
            f"MemorySnapshot({self.label}, "
            f"allocated={self.allocated_mb:.1f}MB, "
            f"reserved={self.reserved_mb:.1f}MB, "
            f"peak={self.peak_allocated_mb:.1f}MB)"
        )


@dataclass
class MemoryDelta:
    """Change in memory between two snapshots."""
    
    label: str
    allocated_delta_mb: float
    reserved_delta_mb: float
    
    @property
    def is_leak(self) -> bool:
        """Check if this delta indicates a potential memory leak."""
        return self.allocated_delta_mb > 0.1  # More than 100KB increase
    
    def __repr__(self) -> str:
        sign = "+" if self.allocated_delta_mb >= 0 else ""
        return f"MemoryDelta({self.label}, {sign}{self.allocated_delta_mb:.2f}MB)"


class MemoryTracker:
    """
    Track GPU memory usage over time.
    
    Example:
        >>> tracker = MemoryTracker()
        >>> tracker.snapshot("before_forward")
        >>> output = model(input)
        >>> tracker.snapshot("after_forward")
        >>> tracker.print_summary()
    """
    
    def __init__(self, device: Optional[int] = None):
        """
        Initialize memory tracker.
        
        Args:
            device: CUDA device to track. Defaults to current device.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for MemoryTracker")
        
        self.device = device or torch.cuda.current_device()
        self._snapshots: List[MemorySnapshot] = []
        self._start_time = datetime.now()
    
    def snapshot(
        self,
        label: str = "snapshot",
        include_tensors: bool = False
    ) -> MemorySnapshot:
        """
        Take a memory snapshot.
        
        Args:
            label: Label for this snapshot
            include_tensors: Whether to include tensor breakdown
            
        Returns:
            MemorySnapshot with current memory state
        """
        torch.cuda.synchronize(self.device)
        
        # Get memory stats
        allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 2)
        peak_allocated = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
        peak_reserved = torch.cuda.max_memory_reserved(self.device) / (1024 ** 2)
        
        # Get free memory
        total = torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 2)
        free = total - reserved
        
        # Get tensor info if requested
        num_tensors = 0
        tensor_sizes = []
        
        if include_tensors:
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj) and obj.is_cuda:
                        num_tensors += 1
                        size_mb = obj.element_size() * obj.nelement() / (1024 ** 2)
                        tensor_sizes.append((str(obj.shape), size_mb))
                except Exception:
                    pass
            
            # Sort by size
            tensor_sizes.sort(key=lambda x: x[1], reverse=True)
            tensor_sizes = tensor_sizes[:20]  # Keep top 20
        
        snapshot = MemorySnapshot(
            timestamp=datetime.now().isoformat(),
            label=label,
            allocated_mb=allocated,
            reserved_mb=reserved,
            peak_allocated_mb=peak_allocated,
            peak_reserved_mb=peak_reserved,
            free_mb=free,
            num_tensors=num_tensors,
            tensor_sizes=tensor_sizes,
        )
        
        self._snapshots.append(snapshot)
        return snapshot
    
    def delta(self, label1: str, label2: str) -> Optional[MemoryDelta]:
        """
        Calculate memory delta between two snapshots.
        
        Args:
            label1: Label of first snapshot
            label2: Label of second snapshot
            
        Returns:
            MemoryDelta or None if snapshots not found
        """
        snap1 = None
        snap2 = None
        
        for snap in self._snapshots:
            if snap.label == label1:
                snap1 = snap
            if snap.label == label2:
                snap2 = snap
        
        if snap1 is None or snap2 is None:
            return None
        
        return MemoryDelta(
            label=f"{label1} -> {label2}",
            allocated_delta_mb=snap2.allocated_mb - snap1.allocated_mb,
            reserved_delta_mb=snap2.reserved_mb - snap1.reserved_mb,
        )
    
    def reset_peak_stats(self) -> None:
        """Reset peak memory statistics."""
        torch.cuda.reset_peak_memory_stats(self.device)
    
    def reset_snapshots(self) -> None:
        """Clear all snapshots."""
        self._snapshots.clear()
    
    def empty_cache(self) -> float:
        """
        Empty CUDA cache and return freed memory.
        
        Returns:
            Amount of memory freed in MB
        """
        before = torch.cuda.memory_reserved(self.device) / (1024 ** 2)
        torch.cuda.empty_cache()
        after = torch.cuda.memory_reserved(self.device) / (1024 ** 2)
        return before - after
    
    @property
    def snapshots(self) -> List[MemorySnapshot]:
        """Get all snapshots."""
        return self._snapshots.copy()
    
    def summary(self) -> str:
        """Generate memory tracking summary."""
        lines = [
            "=" * 70,
            "GPU MEMORY TRACKING SUMMARY",
            "=" * 70,
            f"Device: {torch.cuda.get_device_name(self.device)}",
            f"Snapshots: {len(self._snapshots)}",
            "",
        ]
        
        if self._snapshots:
            # Current state
            latest = self._snapshots[-1]
            lines.extend([
                "Current Memory State:",
                f"  Allocated: {latest.allocated_mb:.2f} MB",
                f"  Reserved:  {latest.reserved_mb:.2f} MB",
                f"  Peak:      {latest.peak_allocated_mb:.2f} MB",
                f"  Free:      {latest.free_mb:.2f} MB",
                "",
                "Snapshot Timeline:",
            ])
            
            for i, snap in enumerate(self._snapshots):
                lines.append(
                    f"  [{i+1}] {snap.label}: "
                    f"{snap.allocated_mb:.2f} MB allocated"
                )
            
            # Deltas
            if len(self._snapshots) > 1:
                lines.extend(["", "Memory Deltas:"])
                for i in range(1, len(self._snapshots)):
                    delta = self.delta(
                        self._snapshots[i-1].label,
                        self._snapshots[i].label
                    )
                    if delta:
                        sign = "+" if delta.allocated_delta_mb >= 0 else ""
                        leak_indicator = " ⚠️ POTENTIAL LEAK" if delta.is_leak else ""
                        lines.append(
                            f"  {delta.label}: "
                            f"{sign}{delta.allocated_delta_mb:.2f} MB{leak_indicator}"
                        )
        
        lines.append("=" * 70)
        return "\n".join(lines)
    
    def print_summary(self) -> None:
        """Print memory tracking summary."""
        print(self.summary())


@contextmanager
def memory_snapshot(label: str = "operation", print_result: bool = True):
    """
    Context manager for tracking memory usage of a code block.
    
    Example:
        >>> with memory_snapshot("forward_pass") as tracker:
        ...     output = model(input)
        Memory Usage [forward_pass]: 128.50 MB (delta: +64.25 MB)
    """
    if not torch.cuda.is_available():
        yield None
        return
    
    tracker = MemoryTracker()
    tracker.snapshot("before")
    
    yield tracker
    
    tracker.snapshot("after")
    
    if print_result:
        delta = tracker.delta("before", "after")
        after = tracker.snapshots[-1]
        
        sign = "+" if delta.allocated_delta_mb >= 0 else ""
        print(
            f"Memory Usage [{label}]: {after.allocated_mb:.2f} MB "
            f"(delta: {sign}{delta.allocated_delta_mb:.2f} MB)"
        )


def track_memory(func: Callable) -> Callable:
    """
    Decorator to track memory usage of a function.
    
    Example:
        >>> @track_memory
        ... def train_step(model, data):
        ...     return model(data)
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        name = func.__name__
        with memory_snapshot(name):
            return func(*args, **kwargs)
    
    return wrapper


class MemoryProfiler:
    """
    Advanced memory profiler with allocation tracking.
    
    Uses PyTorch's memory snapshot feature for detailed allocation analysis.
    
    Example:
        >>> profiler = MemoryProfiler()
        >>> profiler.start()
        >>> output = model(input)
        >>> profiler.stop()
        >>> profiler.export_snapshot("memory_profile.pickle")
    """
    
    def __init__(self, device: Optional[int] = None):
        """
        Initialize memory profiler.
        
        Args:
            device: CUDA device to profile
        """
        self.device = device or torch.cuda.current_device()
        self._recording = False
    
    def start(self) -> None:
        """Start recording memory allocations."""
        if hasattr(torch.cuda.memory, '_record_memory_history'):
            torch.cuda.memory._record_memory_history(
                max_entries=100000
            )
        self._recording = True
    
    def stop(self) -> None:
        """Stop recording memory allocations."""
        if hasattr(torch.cuda.memory, '_record_memory_history'):
            torch.cuda.memory._record_memory_history(enabled=None)
        self._recording = False
    
    def export_snapshot(self, path: str) -> None:
        """
        Export memory snapshot to file.
        
        Args:
            path: Path to save snapshot (should end in .pickle)
        """
        if hasattr(torch.cuda.memory, '_dump_snapshot'):
            torch.cuda.memory._dump_snapshot(path)
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """
        Get detailed memory summary.
        
        Returns:
            Dictionary with memory statistics
        """
        stats = torch.cuda.memory_stats(self.device)
        
        return {
            "allocated_bytes": stats.get("allocated_bytes.all.current", 0),
            "reserved_bytes": stats.get("reserved_bytes.all.current", 0),
            "active_blocks": stats.get("active.all.current", 0),
            "inactive_split_ratio": self._get_fragmentation_ratio(stats),
            "oom_count": stats.get("num_ooms", 0),
            "allocation_retries": stats.get("num_alloc_retries", 0),
        }
    
    def _get_fragmentation_ratio(self, stats: Dict) -> float:
        """Calculate memory fragmentation ratio."""
        active = stats.get("active_bytes.all.current", 0)
        reserved = stats.get("reserved_bytes.all.current", 0)
        
        if reserved == 0:
            return 0.0
        
        return 1 - (active / reserved)


def find_memory_leaks(
    func: Callable,
    iterations: int = 10,
    threshold_mb: float = 1.0
) -> List[MemoryDelta]:
    """
    Run function multiple times to detect memory leaks.
    
    Args:
        func: Function to test (should take no arguments)
        iterations: Number of iterations to run
        threshold_mb: Minimum delta to consider as leak
        
    Returns:
        List of MemoryDelta objects indicating potential leaks
    """
    tracker = MemoryTracker()
    leaks = []
    
    # Initial measurement
    gc.collect()
    torch.cuda.empty_cache()
    tracker.snapshot("iteration_0")
    
    for i in range(iterations):
        func()
        gc.collect()
        torch.cuda.empty_cache()
        tracker.snapshot(f"iteration_{i+1}")
        
        delta = tracker.delta(f"iteration_{i}", f"iteration_{i+1}")
        if delta and abs(delta.allocated_delta_mb) > threshold_mb:
            leaks.append(delta)
    
    return leaks
