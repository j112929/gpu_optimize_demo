"""
Communication Profiler - Profile NCCL collective operations.

This module provides tools for profiling distributed communication
operations like AllReduce, AllGather, ReduceScatter, and Broadcast.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist


@dataclass
class CollectiveStats:
    """Statistics for a collective operation."""
    
    name: str
    count: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    total_bytes: int
    
    # Derived metrics
    @property
    def bandwidth_gbps(self) -> float:
        """Calculate bandwidth in Gbps."""
        if self.total_time_ms <= 0:
            return 0.0
        bytes_per_second = self.total_bytes / (self.total_time_ms / 1000)
        return bytes_per_second * 8 / 1e9


@dataclass
class CommProfile:
    """Profile of communication operations."""
    
    world_size: int
    local_rank: int
    
    # Collective stats
    all_reduce_stats: Optional[CollectiveStats] = None
    all_gather_stats: Optional[CollectiveStats] = None
    reduce_scatter_stats: Optional[CollectiveStats] = None
    broadcast_stats: Optional[CollectiveStats] = None
    
    # Overall metrics
    total_comm_time_ms: float = 0.0
    total_comp_time_ms: float = 0.0
    
    # Event log
    events: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def comm_compute_ratio(self) -> float:
        """Ratio of communication to computation time."""
        if self.total_comp_time_ms <= 0:
            return float('inf')
        return self.total_comm_time_ms / self.total_comp_time_ms
    
    def summary(self) -> str:
        """Generate profile summary."""
        lines = [
            "=" * 70,
            "NCCL COMMUNICATION PROFILE",
            "=" * 70,
            f"World Size: {self.world_size}",
            f"Local Rank: {self.local_rank}",
            "",
            f"Total Communication Time: {self.total_comm_time_ms:.2f} ms",
            f"Total Computation Time: {self.total_comp_time_ms:.2f} ms",
            f"Comm/Compute Ratio: {self.comm_compute_ratio:.2%}",
            "",
            "Collective Operations:",
        ]
        
        for stats in [self.all_reduce_stats, self.all_gather_stats, 
                      self.reduce_scatter_stats, self.broadcast_stats]:
            if stats is not None:
                lines.append(
                    f"  {stats.name}:"
                    f"\n    Count: {stats.count}"
                    f"\n    Total: {stats.total_time_ms:.2f} ms"
                    f"\n    Avg: {stats.avg_time_ms:.3f} ms"
                    f"\n    Bandwidth: {stats.bandwidth_gbps:.2f} Gbps"
                )
        
        lines.append("=" * 70)
        return "\n".join(lines)


class CommProfiler:
    """
    Profiler for NCCL/distributed communication operations.
    
    Example:
        >>> profiler = CommProfiler()
        >>> with profiler:
        ...     dist.all_reduce(tensor)
        ...     output = model(input)
        >>> print(profiler.profile.summary())
    """
    
    def __init__(
        self,
        enabled: bool = True,
        record_events: bool = True,
    ):
        """
        Initialize communication profiler.
        
        Args:
            enabled: Whether profiling is enabled
            record_events: Whether to record individual events
        """
        self.enabled = enabled
        self.record_events = record_events
        
        self._events: List[Dict[str, Any]] = []
        self._collective_times: Dict[str, List[Tuple[float, int]]] = {
            'all_reduce': [],
            'all_gather': [],
            'reduce_scatter': [],
            'broadcast': [],
        }
        
        self._comp_start: Optional[float] = None
        self._total_comp_time: float = 0.0
        self._profile: Optional[CommProfile] = None
        
        # Hooks for intercepting collectives
        self._original_all_reduce = None
        self._original_all_gather = None
    
    @property
    def profile(self) -> Optional[CommProfile]:
        """Get profiling results."""
        return self._profile
    
    def __enter__(self) -> "CommProfiler":
        """Enter profiling context."""
        if self.enabled:
            self._install_hooks()
            self._comp_start = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit profiling context."""
        if self.enabled:
            if self._comp_start:
                self._total_comp_time = (time.perf_counter() - self._comp_start) * 1000
            self._remove_hooks()
            self._build_profile()
        return False
    
    def _install_hooks(self) -> None:
        """Install hooks to intercept collective operations."""
        if not dist.is_initialized():
            return
        
        # Store original functions
        self._original_all_reduce = dist.all_reduce
        self._original_all_gather = dist.all_gather
        self._original_reduce_scatter = getattr(dist, 'reduce_scatter', None)
        self._original_broadcast = dist.broadcast
        
        # Create wrapped versions
        profiler = self
        
        def wrapped_all_reduce(tensor, *args, **kwargs):
            return profiler._profile_collective(
                'all_reduce',
                profiler._original_all_reduce,
                tensor,
                *args,
                **kwargs
            )
        
        def wrapped_all_gather(output_list, tensor, *args, **kwargs):
            return profiler._profile_collective(
                'all_gather',
                profiler._original_all_gather,
                output_list,
                tensor,
                *args,
                **kwargs
            )
        
        def wrapped_broadcast(tensor, *args, **kwargs):
            return profiler._profile_collective(
                'broadcast',
                profiler._original_broadcast,
                tensor,
                *args,
                **kwargs
            )
        
        # Install hooks
        dist.all_reduce = wrapped_all_reduce
        dist.all_gather = wrapped_all_gather
        dist.broadcast = wrapped_broadcast
    
    def _remove_hooks(self) -> None:
        """Remove installed hooks."""
        if self._original_all_reduce:
            dist.all_reduce = self._original_all_reduce
        if self._original_all_gather:
            dist.all_gather = self._original_all_gather
        if self._original_broadcast:
            dist.broadcast = self._original_broadcast
    
    def _profile_collective(
        self,
        name: str,
        func,
        *args,
        **kwargs
    ) -> Any:
        """Profile a collective operation."""
        # Get tensor size
        tensor_size = 0
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensor_size += arg.element_size() * arg.nelement()
        
        # Time the operation
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        result = func(*args, **kwargs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        # Record
        self._collective_times[name].append((elapsed_ms, tensor_size))
        
        if self.record_events:
            self._events.append({
                'type': name,
                'time_ms': elapsed_ms,
                'bytes': tensor_size,
                'timestamp': time.time(),
            })
        
        return result
    
    def _build_profile(self) -> None:
        """Build profile from collected data."""
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        local_rank = dist.get_rank() if dist.is_initialized() else 0
        
        total_comm_time = 0.0
        
        def build_stats(name: str, times: List[Tuple[float, int]]) -> Optional[CollectiveStats]:
            if not times:
                return None
            
            durations = [t[0] for t in times]
            bytes_list = [t[1] for t in times]
            
            return CollectiveStats(
                name=name,
                count=len(times),
                total_time_ms=sum(durations),
                avg_time_ms=sum(durations) / len(durations),
                min_time_ms=min(durations),
                max_time_ms=max(durations),
                total_bytes=sum(bytes_list),
            )
        
        all_reduce_stats = build_stats('AllReduce', self._collective_times['all_reduce'])
        all_gather_stats = build_stats('AllGather', self._collective_times['all_gather'])
        reduce_scatter_stats = build_stats('ReduceScatter', self._collective_times['reduce_scatter'])
        broadcast_stats = build_stats('Broadcast', self._collective_times['broadcast'])
        
        # Sum up communication time
        for stats in [all_reduce_stats, all_gather_stats, reduce_scatter_stats, broadcast_stats]:
            if stats:
                total_comm_time += stats.total_time_ms
        
        self._profile = CommProfile(
            world_size=world_size,
            local_rank=local_rank,
            all_reduce_stats=all_reduce_stats,
            all_gather_stats=all_gather_stats,
            reduce_scatter_stats=reduce_scatter_stats,
            broadcast_stats=broadcast_stats,
            total_comm_time_ms=total_comm_time,
            total_comp_time_ms=max(0, self._total_comp_time - total_comm_time),
            events=self._events,
        )
    
    def print_summary(self) -> None:
        """Print profile summary."""
        if self._profile:
            print(self._profile.summary())


@contextmanager
def profile_collective(name: str = "collective"):
    """
    Context manager for profiling a single collective operation.
    
    Example:
        >>> with profile_collective("gradient_sync"):
        ...     dist.all_reduce(gradients)
    """
    start_event = None
    end_event = None
    
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        start_time = time.perf_counter()
    
    yield
    
    if torch.cuda.is_available() and start_event and end_event:
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
    else:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
    
    print(f"Collective [{name}]: {elapsed_ms:.3f} ms")


def measure_all_reduce_time(
    tensor: torch.Tensor,
    iterations: int = 100,
    warmup: int = 10,
) -> Dict[str, float]:
    """
    Measure all-reduce time for a tensor.
    
    Args:
        tensor: Tensor to reduce
        iterations: Number of iterations
        warmup: Warmup iterations
        
    Returns:
        Dictionary with timing statistics
    """
    if not dist.is_initialized():
        raise RuntimeError("Distributed not initialized")
    
    # Warmup
    for _ in range(warmup):
        dist.all_reduce(tensor.clone())
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    # Measure
    times = []
    for _ in range(iterations):
        t = tensor.clone()
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        dist.all_reduce(t)
        end.record()
        
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    return {
        'mean_ms': sum(times) / len(times),
        'std_ms': (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
        'min_ms': min(times),
        'max_ms': max(times),
        'p95_ms': sorted(times)[int(len(times) * 0.95)],
    }
