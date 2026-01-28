"""
IO Benchmark - Measure and compare data loading performance.

This module provides tools for benchmarking different data loading
configurations to find optimal settings.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class BenchmarkResult:
    """Results from IO benchmark."""
    
    name: str
    samples_per_second: float
    batches_per_second: float
    avg_batch_time_ms: float
    total_time_s: float
    
    # Configuration
    batch_size: int
    num_workers: int
    pin_memory: bool
    
    # Additional metrics
    gpu_transfer_time_ms: float = 0.0
    cpu_preprocessing_time_ms: float = 0.0
    
    def __repr__(self) -> str:
        return (
            f"BenchmarkResult({self.name}, "
            f"{self.samples_per_second:.1f} samples/s, "
            f"{self.avg_batch_time_ms:.2f} ms/batch)"
        )
    
    def summary(self) -> str:
        """Generate summary string."""
        return (
            f"Benchmark: {self.name}\n"
            f"  Configuration:\n"
            f"    Batch size: {self.batch_size}\n"
            f"    Workers: {self.num_workers}\n"
            f"    Pin memory: {self.pin_memory}\n"
            f"  Performance:\n"
            f"    Samples/second: {self.samples_per_second:.1f}\n"
            f"    Batches/second: {self.batches_per_second:.2f}\n"
            f"    Avg batch time: {self.avg_batch_time_ms:.2f} ms\n"
            f"    Total time: {self.total_time_s:.2f} s\n"
        )


@dataclass
class ComparisonResult:
    """Comparison between multiple benchmark results."""
    
    baseline: BenchmarkResult
    comparisons: List[BenchmarkResult]
    
    def summary(self) -> str:
        """Generate comparison summary."""
        lines = [
            "=" * 70,
            "IO BENCHMARK COMPARISON",
            "=" * 70,
            "",
            f"Baseline: {self.baseline.name}",
            f"  Throughput: {self.baseline.samples_per_second:.1f} samples/s",
            f"  Batch time: {self.baseline.avg_batch_time_ms:.2f} ms",
            "",
            "Comparisons:",
        ]
        
        for result in self.comparisons:
            speedup = result.samples_per_second / self.baseline.samples_per_second
            time_reduction = (1 - result.avg_batch_time_ms / self.baseline.avg_batch_time_ms) * 100
            
            lines.append(f"  {result.name}:")
            lines.append(f"    Throughput: {result.samples_per_second:.1f} samples/s ({speedup:.2f}x)")
            lines.append(f"    Batch time: {result.avg_batch_time_ms:.2f} ms ({time_reduction:+.1f}%)")
        
        lines.append("=" * 70)
        return "\n".join(lines)


class IOBenchmark:
    """
    Benchmark data loading performance.
    
    Example:
        >>> benchmark = IOBenchmark(dataset)
        >>> result = benchmark.run(batch_size=64, num_workers=4)
        >>> print(result.summary())
    """
    
    def __init__(
        self,
        dataset: Dataset,
        num_batches: int = 100,
        warmup_batches: int = 10,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize benchmark.
        
        Args:
            dataset: PyTorch dataset to benchmark
            num_batches: Number of batches to measure
            warmup_batches: Number of warmup batches
            device: GPU device for transfer benchmarks
        """
        self.dataset = dataset
        self.num_batches = num_batches
        self.warmup_batches = warmup_batches
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def run(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        name: str = "benchmark",
        **loader_kwargs
    ) -> BenchmarkResult:
        """
        Run benchmark with specified configuration.
        
        Args:
            batch_size: Batch size
            num_workers: Number of data loading workers
            pin_memory: Whether to use pinned memory
            prefetch_factor: Prefetch factor for workers
            name: Name for this benchmark configuration
            **loader_kwargs: Additional DataLoader arguments
            
        Returns:
            BenchmarkResult with performance metrics
        """
        # Create DataLoader
        loader_kwargs.update({
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory and torch.cuda.is_available(),
            "shuffle": True,
        })
        
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = prefetch_factor
            loader_kwargs["persistent_workers"] = True
        
        loader = DataLoader(self.dataset, **loader_kwargs)
        
        # Warmup
        batch_iter = iter(loader)
        for _ in range(min(self.warmup_batches, len(loader))):
            try:
                batch = next(batch_iter)
                if torch.cuda.is_available():
                    self._to_device(batch)
                    torch.cuda.synchronize()
            except StopIteration:
                break
        
        # Reset iterator for measurement
        batch_iter = iter(loader)
        
        # Measure
        batch_times = []
        gpu_transfer_times = []
        samples_count = 0
        
        start_time = time.perf_counter()
        
        for i in range(min(self.num_batches, len(loader))):
            try:
                batch_start = time.perf_counter()
                batch = next(batch_iter)
                
                cpu_time = time.perf_counter() - batch_start
                
                # Measure GPU transfer
                if torch.cuda.is_available():
                    transfer_start = time.perf_counter()
                    batch = self._to_device(batch)
                    torch.cuda.synchronize()
                    gpu_time = time.perf_counter() - transfer_start
                    gpu_transfer_times.append(gpu_time * 1000)
                
                batch_end = time.perf_counter()
                batch_times.append((batch_end - batch_start) * 1000)
                
                # Count samples
                if isinstance(batch, (list, tuple)):
                    samples_count += len(batch[0])
                elif isinstance(batch, torch.Tensor):
                    samples_count += len(batch)
                elif isinstance(batch, dict):
                    first_value = next(iter(batch.values()))
                    samples_count += len(first_value)
                
            except StopIteration:
                break
        
        total_time = time.perf_counter() - start_time
        
        # Calculate metrics
        avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
        avg_gpu_transfer = sum(gpu_transfer_times) / len(gpu_transfer_times) if gpu_transfer_times else 0
        
        return BenchmarkResult(
            name=name,
            samples_per_second=samples_count / total_time if total_time > 0 else 0,
            batches_per_second=len(batch_times) / total_time if total_time > 0 else 0,
            avg_batch_time_ms=avg_batch_time,
            total_time_s=total_time,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            gpu_transfer_time_ms=avg_gpu_transfer,
        )
    
    def _to_device(self, batch: Any) -> Any:
        """Move batch to device."""
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device, non_blocking=True)
        elif isinstance(batch, (list, tuple)):
            return type(batch)([self._to_device(item) for item in batch])
        elif isinstance(batch, dict):
            return {key: self._to_device(value) for key, value in batch.items()}
        return batch
    
    def compare_workers(
        self,
        batch_size: int = 32,
        worker_counts: Optional[List[int]] = None,
    ) -> ComparisonResult:
        """
        Compare performance across different worker counts.
        
        Args:
            batch_size: Fixed batch size
            worker_counts: List of worker counts to test
            
        Returns:
            ComparisonResult with baseline and comparisons
        """
        if worker_counts is None:
            import os
            max_workers = os.cpu_count() or 8
            worker_counts = [0, 2, 4, 8, min(16, max_workers)]
        
        results = []
        for count in worker_counts:
            result = self.run(
                batch_size=batch_size,
                num_workers=count,
                name=f"{count}_workers"
            )
            results.append(result)
        
        return ComparisonResult(
            baseline=results[0],
            comparisons=results[1:],
        )
    
    def find_optimal_config(
        self,
        batch_sizes: Optional[List[int]] = None,
        worker_counts: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Find optimal DataLoader configuration.
        
        Args:
            batch_sizes: List of batch sizes to test
            worker_counts: List of worker counts to test
            
        Returns:
            Dictionary with optimal configuration
        """
        if batch_sizes is None:
            batch_sizes = [16, 32, 64, 128]
        
        if worker_counts is None:
            import os
            max_workers = os.cpu_count() or 8
            worker_counts = [0, 2, 4, 8, min(16, max_workers)]
        
        best_result: Optional[BenchmarkResult] = None
        all_results: List[BenchmarkResult] = []
        
        for batch_size in batch_sizes:
            for num_workers in worker_counts:
                result = self.run(
                    batch_size=batch_size,
                    num_workers=num_workers,
                    name=f"bs{batch_size}_w{num_workers}"
                )
                all_results.append(result)
                
                if best_result is None or result.samples_per_second > best_result.samples_per_second:
                    best_result = result
        
        return {
            "optimal": {
                "batch_size": best_result.batch_size if best_result else 32,
                "num_workers": best_result.num_workers if best_result else 4,
                "pin_memory": True,
            },
            "best_throughput": best_result.samples_per_second if best_result else 0,
            "all_results": all_results,
        }


def benchmark_dataloader(
    dataset: Dataset,
    num_batches: int = 100,
    **loader_kwargs
) -> BenchmarkResult:
    """
    Quick benchmark function.
    
    Args:
        dataset: PyTorch dataset
        num_batches: Number of batches to measure
        **loader_kwargs: DataLoader configuration
        
    Returns:
        BenchmarkResult
    """
    benchmark = IOBenchmark(dataset, num_batches=num_batches)
    return benchmark.run(**loader_kwargs)
