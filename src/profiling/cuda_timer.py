"""
CUDA Timer - Precise GPU timing using CUDA events.

This module provides microsecond-precision timing for CUDA operations using
CUDA events, which is more accurate than CPU-based timing for GPU operations.
"""

import functools
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch


@dataclass
class TimingResult:
    """Container for timing results."""
    name: str
    elapsed_ms: float
    iterations: int = 1
    
    @property
    def avg_ms(self) -> float:
        """Average time per iteration in milliseconds."""
        return self.elapsed_ms / self.iterations
    
    @property
    def throughput(self) -> float:
        """Iterations per second."""
        return self.iterations / (self.elapsed_ms / 1000) if self.elapsed_ms > 0 else 0
    
    def __repr__(self) -> str:
        return f"TimingResult(name='{self.name}', elapsed_ms={self.elapsed_ms:.3f}, iterations={self.iterations})"


class CUDATimer:
    """
    CUDA Event-based timer for precise GPU timing.
    
    CUDA events are synchronized with the GPU stream, providing accurate
    timing that accounts for asynchronous kernel launches.
    
    Example:
        >>> timer = CUDATimer()
        >>> timer.start("forward")
        >>> output = model(input)
        >>> timer.stop("forward")
        >>> print(timer.elapsed("forward"))
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize CUDA timer.
        
        Args:
            device: CUDA device to use. Defaults to current device.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for CUDATimer")
        
        self.device = device or torch.cuda.current_device()
        self._events: Dict[str, Tuple[torch.cuda.Event, torch.cuda.Event]] = {}
        self._results: Dict[str, TimingResult] = {}
    
    def start(self, name: str) -> None:
        """
        Start timing a region.
        
        Args:
            name: Name for this timing region
        """
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        self._events[name] = (start_event, end_event)
    
    def stop(self, name: str) -> float:
        """
        Stop timing a region and return elapsed time.
        
        Args:
            name: Name of the timing region
            
        Returns:
            Elapsed time in milliseconds
        """
        if name not in self._events:
            raise ValueError(f"Timer '{name}' was not started")
        
        start_event, end_event = self._events[name]
        end_event.record()
        
        # Synchronize to get accurate timing
        torch.cuda.synchronize()
        
        elapsed_ms = start_event.elapsed_time(end_event)
        self._results[name] = TimingResult(name=name, elapsed_ms=elapsed_ms)
        
        return elapsed_ms
    
    def elapsed(self, name: str) -> float:
        """
        Get elapsed time for a completed timing region.
        
        Args:
            name: Name of the timing region
            
        Returns:
            Elapsed time in milliseconds
        """
        if name in self._results:
            return self._results[name].elapsed_ms
        
        if name not in self._events:
            raise ValueError(f"Timer '{name}' does not exist")
        
        return self.stop(name)
    
    def reset(self, name: Optional[str] = None) -> None:
        """
        Reset timer(s).
        
        Args:
            name: Specific timer to reset, or None to reset all
        """
        if name is None:
            self._events.clear()
            self._results.clear()
        else:
            self._events.pop(name, None)
            self._results.pop(name, None)
    
    def summary(self) -> str:
        """Generate summary of all timing results."""
        lines = ["CUDA Timing Summary", "=" * 50]
        
        for name, result in sorted(self._results.items()):
            lines.append(f"{name}: {result.elapsed_ms:.3f} ms")
        
        lines.append("=" * 50)
        return "\n".join(lines)
    
    @property
    def results(self) -> Dict[str, TimingResult]:
        """Get all timing results."""
        return self._results.copy()


class CUDATimerContext:
    """
    Context manager for CUDA timing.
    
    Example:
        >>> with CUDATimerContext("forward") as timer:
        ...     output = model(input)
        >>> print(f"Elapsed: {timer.elapsed_ms:.3f} ms")
    """
    
    def __init__(self, name: str = "default"):
        """
        Initialize timing context.
        
        Args:
            name: Name for this timing region
        """
        self.name = name
        self._start_event: Optional[torch.cuda.Event] = None
        self._end_event: Optional[torch.cuda.Event] = None
        self._elapsed_ms: Optional[float] = None
    
    def __enter__(self) -> "CUDATimerContext":
        """Start timing."""
        if torch.cuda.is_available():
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Stop timing and calculate elapsed time."""
        if self._end_event is not None:
            self._end_event.record()
            torch.cuda.synchronize()
            self._elapsed_ms = self._start_event.elapsed_time(self._end_event)
        return False
    
    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return self._elapsed_ms or 0.0


@contextmanager
def cuda_time(name: str = "operation"):
    """
    Context manager for quick CUDA timing.
    
    Example:
        >>> with cuda_time("matmul"):
        ...     result = torch.mm(a, b)
        CUDA Time [matmul]: 0.123 ms
    """
    ctx = CUDATimerContext(name)
    with ctx:
        yield ctx
    
    print(f"CUDA Time [{name}]: {ctx.elapsed_ms:.3f} ms")


def cuda_timed(func: Callable) -> Callable:
    """
    Decorator to time CUDA operations in a function.
    
    Example:
        >>> @cuda_timed
        ... def forward(model, input):
        ...     return model(input)
        >>> output = forward(model, x)  # Prints timing info
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        name = func.__name__
        with CUDATimerContext(name) as timer:
            result = func(*args, **kwargs)
        print(f"CUDA Time [{name}]: {timer.elapsed_ms:.3f} ms")
        return result
    
    return wrapper


class CUDABenchmark:
    """
    Benchmark utility for measuring GPU operation performance.
    
    Handles warmup, multiple iterations, and statistical analysis.
    
    Example:
        >>> benchmark = CUDABenchmark(warmup=10, iterations=100)
        >>> result = benchmark.run(lambda: model(input))
        >>> print(f"Mean: {result.mean_ms:.3f} ms, Std: {result.std_ms:.3f} ms")
    """
    
    def __init__(
        self,
        warmup: int = 10,
        iterations: int = 100,
        device: Optional[torch.device] = None
    ):
        """
        Initialize benchmark.
        
        Args:
            warmup: Number of warmup iterations
            iterations: Number of timed iterations
            device: CUDA device to use
        """
        self.warmup = warmup
        self.iterations = iterations
        self.device = device or torch.cuda.current_device()
    
    def run(
        self,
        func: Callable[[], Any],
        name: str = "benchmark"
    ) -> "BenchmarkResult":
        """
        Run benchmark on a function.
        
        Args:
            func: Function to benchmark (should take no arguments)
            name: Name for this benchmark
            
        Returns:
            BenchmarkResult with timing statistics
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for benchmarking")
        
        # Warmup
        for _ in range(self.warmup):
            func()
        
        torch.cuda.synchronize()
        
        # Timed iterations
        times = []
        for _ in range(self.iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            func()
            end.record()
            
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        
        return BenchmarkResult(name=name, times=times)


@dataclass
class BenchmarkResult:
    """Container for benchmark results with statistics."""
    
    name: str
    times: List[float]
    
    @property
    def mean_ms(self) -> float:
        """Mean time in milliseconds."""
        return sum(self.times) / len(self.times) if self.times else 0
    
    @property
    def std_ms(self) -> float:
        """Standard deviation in milliseconds."""
        if len(self.times) < 2:
            return 0
        mean = self.mean_ms
        variance = sum((t - mean) ** 2 for t in self.times) / (len(self.times) - 1)
        return variance ** 0.5
    
    @property
    def min_ms(self) -> float:
        """Minimum time in milliseconds."""
        return min(self.times) if self.times else 0
    
    @property
    def max_ms(self) -> float:
        """Maximum time in milliseconds."""
        return max(self.times) if self.times else 0
    
    @property
    def median_ms(self) -> float:
        """Median time in milliseconds."""
        if not self.times:
            return 0
        sorted_times = sorted(self.times)
        n = len(sorted_times)
        if n % 2 == 0:
            return (sorted_times[n // 2 - 1] + sorted_times[n // 2]) / 2
        return sorted_times[n // 2]
    
    def percentile(self, p: float) -> float:
        """
        Get p-th percentile time.
        
        Args:
            p: Percentile (0-100)
        """
        if not self.times:
            return 0
        sorted_times = sorted(self.times)
        idx = int(len(sorted_times) * p / 100)
        return sorted_times[min(idx, len(sorted_times) - 1)]
    
    def summary(self) -> str:
        """Generate summary string."""
        return (
            f"Benchmark [{self.name}] ({len(self.times)} iterations)\n"
            f"  Mean:   {self.mean_ms:.3f} ms (±{self.std_ms:.3f})\n"
            f"  Median: {self.median_ms:.3f} ms\n"
            f"  Min:    {self.min_ms:.3f} ms\n"
            f"  Max:    {self.max_ms:.3f} ms\n"
            f"  P95:    {self.percentile(95):.3f} ms\n"
            f"  P99:    {self.percentile(99):.3f} ms"
        )
