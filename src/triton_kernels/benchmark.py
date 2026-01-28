"""
Triton Kernel Benchmarking - Tools for measuring and comparing kernel performance.
"""

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import triton


@dataclass
class BenchmarkResult:
    """Result of a kernel benchmark."""
    name: str
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    tflops: float = 0.0
    bandwidth_gbps: float = 0.0
    
    def __repr__(self) -> str:
        return f"{self.name}: {self.mean_ms:.3f}ms (±{self.std_ms:.3f})"


class TritonBenchmark:
    """
    Benchmark Triton kernels against PyTorch implementations.
    
    Example:
        >>> bench = TritonBenchmark(warmup=10, iterations=100)
        >>> result = bench.run(triton_fn, x, y)
    """
    
    def __init__(self, warmup: int = 10, iterations: int = 100):
        self.warmup = warmup
        self.iterations = iterations
    
    def run(
        self,
        fn: Callable,
        *args,
        name: str = "kernel",
        flops: Optional[int] = None,
        bytes_accessed: Optional[int] = None,
        **kwargs
    ) -> BenchmarkResult:
        """Run benchmark on a function."""
        # Warmup
        for _ in range(self.warmup):
            fn(*args, **kwargs)
        torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(self.iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            fn(*args, **kwargs)
            end.record()
            
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        
        mean_ms = sum(times) / len(times)
        std_ms = (sum((t - mean_ms)**2 for t in times) / len(times)) ** 0.5
        
        tflops = (flops / mean_ms / 1e9) if flops else 0.0
        bandwidth = (bytes_accessed / mean_ms / 1e6) if bytes_accessed else 0.0
        
        return BenchmarkResult(
            name=name,
            mean_ms=mean_ms,
            std_ms=std_ms,
            min_ms=min(times),
            max_ms=max(times),
            tflops=tflops,
            bandwidth_gbps=bandwidth,
        )


def compare_with_pytorch(
    triton_fn: Callable,
    pytorch_fn: Callable,
    *args,
    warmup: int = 10,
    iterations: int = 100,
    **kwargs
) -> Dict[str, Any]:
    """
    Compare Triton kernel with PyTorch equivalent.
    
    Returns:
        Dict with results and speedup
    """
    bench = TritonBenchmark(warmup=warmup, iterations=iterations)
    
    triton_result = bench.run(triton_fn, *args, name="triton", **kwargs)
    pytorch_result = bench.run(pytorch_fn, *args, name="pytorch", **kwargs)
    
    speedup = pytorch_result.mean_ms / triton_result.mean_ms
    
    return {
        "triton": triton_result,
        "pytorch": pytorch_result,
        "speedup": speedup,
        "summary": f"Triton is {speedup:.2f}x faster",
    }


def kernel_autotuner(
    kernel_fn: Callable,
    configs: List[Dict],
    *args,
    iterations: int = 20,
) -> Tuple[Dict, List[BenchmarkResult]]:
    """
    Automatically find best config for a kernel.
    
    Args:
        kernel_fn: Kernel function that accepts config kwargs
        configs: List of config dicts to try
        args: Input arguments
        iterations: Iterations per config
    """
    bench = TritonBenchmark(warmup=5, iterations=iterations)
    results = []
    
    for config in configs:
        try:
            result = bench.run(
                lambda: kernel_fn(*args, **config),
                name=str(config),
            )
            results.append((config, result))
        except Exception as e:
            print(f"Config {config} failed: {e}")
    
    if not results:
        raise RuntimeError("All configs failed")
    
    best_config, best_result = min(results, key=lambda x: x[1].mean_ms)
    
    return {
        "best_config": best_config,
        "best_time_ms": best_result.mean_ms,
        "all_results": results,
    }


def benchmark_memory_bandwidth(
    size_mb: float = 64.0,
    iterations: int = 50,
) -> Dict[str, float]:
    """Benchmark GPU memory bandwidth."""
    num_elements = int(size_mb * 1024 * 1024 / 4)
    x = torch.randn(num_elements, device='cuda', dtype=torch.float32)
    y = torch.empty_like(x)
    
    bench = TritonBenchmark(warmup=10, iterations=iterations)
    
    # Copy bandwidth
    result = bench.run(
        lambda: y.copy_(x),
        name="copy",
        bytes_accessed=int(2 * size_mb * 1024 * 1024),
    )
    
    return {
        "size_mb": size_mb,
        "copy_time_ms": result.mean_ms,
        "bandwidth_gbps": result.bandwidth_gbps,
    }
