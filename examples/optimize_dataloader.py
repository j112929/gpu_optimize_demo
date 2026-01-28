#!/usr/bin/env python3
"""
Example: Optimize DataLoader configuration.

This example demonstrates:
1. How to find optimal DataLoader settings
2. Impact of different num_workers
3. Benefits of pin_memory and prefetching
4. GPU prefetcher for hiding data transfer latency

Usage:
    python examples/optimize_dataloader.py [--batch-size 64]
"""

import argparse
import time
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.io_optimize import FastDataLoader, GPUPrefetcher, IOBenchmark
from src.utils.logger import setup_logger


class SyntheticImageDataset(Dataset):
    """Synthetic dataset that simulates image loading."""
    
    def __init__(
        self,
        size: int = 10000,
        image_size: Tuple[int, int, int] = (3, 224, 224),
        num_classes: int = 1000,
        simulate_disk_io: bool = True,
    ):
        """
        Initialize synthetic dataset.
        
        Args:
            size: Number of samples
            image_size: Image dimensions (C, H, W)
            num_classes: Number of classes
            simulate_disk_io: Add artificial delay to simulate disk I/O
        """
        self.size = size
        self.image_size = image_size
        self.num_classes = num_classes
        self.simulate_disk_io = simulate_disk_io
    
    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Simulate image loading from disk
        if self.simulate_disk_io:
            time.sleep(0.001)  # 1ms delay per sample
        
        # Generate random image and label
        image = torch.randn(*self.image_size)
        label = idx % self.num_classes
        
        return image, label


def benchmark_configuration(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
    num_batches: int = 50,
    use_prefetcher: bool = False,
) -> dict:
    """Benchmark a specific DataLoader configuration."""
    
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "pin_memory": pin_memory,
        "num_workers": num_workers,
    }
    
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = True
    
    loader = DataLoader(dataset, **loader_kwargs)
    
    # Optionally wrap with GPU prefetcher
    if use_prefetcher and torch.cuda.is_available():
        data_source = GPUPrefetcher(loader)
    else:
        data_source = loader
    
    # Warmup
    data_iter = iter(data_source)
    for _ in range(min(5, len(loader))):
        try:
            batch = next(data_iter)
            if torch.cuda.is_available() and not use_prefetcher:
                if isinstance(batch, (list, tuple)):
                    batch = tuple(b.cuda(non_blocking=True) if isinstance(b, torch.Tensor) else b for b in batch)
                torch.cuda.synchronize()
        except StopIteration:
            break
    
    # Benchmark
    samples = 0
    start_time = time.perf_counter()
    
    data_iter = iter(data_source)
    for i in range(min(num_batches, len(loader))):
        try:
            batch = next(data_iter)
            
            if torch.cuda.is_available() and not use_prefetcher:
                if isinstance(batch, (list, tuple)):
                    batch = tuple(b.cuda(non_blocking=True) if isinstance(b, torch.Tensor) else b for b in batch)
                torch.cuda.synchronize()
            
            # Count samples
            if isinstance(batch, (list, tuple)):
                samples += len(batch[0])
            elif isinstance(batch, torch.Tensor):
                samples += len(batch)
                
        except StopIteration:
            break
    
    elapsed = time.perf_counter() - start_time
    
    return {
        "samples_per_second": samples / elapsed if elapsed > 0 else 0,
        "batch_time_ms": (elapsed / (i + 1)) * 1000 if (i + 1) > 0 else 0,
        "total_samples": samples,
        "elapsed_seconds": elapsed,
    }


def compare_configurations(dataset: Dataset, batch_size: int, num_batches: int = 50):
    """Compare different DataLoader configurations."""
    
    configurations = [
        # (name, num_workers, pin_memory, prefetch_factor, use_prefetcher)
        ("Baseline (0 workers)", 0, False, 2, False),
        ("2 workers", 2, False, 2, False),
        ("4 workers", 4, False, 2, False),
        ("4 workers + pin_memory", 4, True, 2, False),
        ("4 workers + pin + prefetch=4", 4, True, 4, False),
        ("8 workers + pin_memory", 8, True, 2, False),
    ]
    
    if torch.cuda.is_available():
        configurations.append(
            ("4 workers + GPU prefetcher", 4, True, 2, True)
        )
    
    results = []
    
    print("\n" + "=" * 70)
    print("DATALOADER CONFIGURATION COMPARISON")
    print("=" * 70)
    print(f"\nDataset size: {len(dataset)} samples")
    print(f"Batch size: {batch_size}")
    print(f"Benchmarking {num_batches} batches per configuration...\n")
    
    for name, workers, pin, prefetch, use_pf in configurations:
        print(f"Testing: {name}...", end=" ", flush=True)
        
        result = benchmark_configuration(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=workers,
            pin_memory=pin,
            prefetch_factor=prefetch,
            num_batches=num_batches,
            use_prefetcher=use_pf,
        )
        
        result["name"] = name
        results.append(result)
        
        print(f"{result['samples_per_second']:.1f} samples/s")
    
    # Sort by throughput
    results.sort(key=lambda x: x["samples_per_second"], reverse=True)
    
    # Print summary
    print("\n" + "-" * 70)
    print("RESULTS (sorted by throughput)")
    print("-" * 70)
    print(f"{'Configuration':<40} {'Samples/s':>12} {'Batch Time':>12} {'Speedup':>10}")
    print("-" * 70)
    
    baseline_throughput = None
    for result in reversed(results):
        if "Baseline" in result["name"]:
            baseline_throughput = result["samples_per_second"]
            break
    
    for result in results:
        speedup = result["samples_per_second"] / baseline_throughput if baseline_throughput else 1.0
        print(f"{result['name']:<40} {result['samples_per_second']:>12.1f} "
              f"{result['batch_time_ms']:>10.2f}ms {speedup:>9.2f}x")
    
    print("-" * 70)
    
    # Best configuration
    best = results[0]
    print(f"\n✅ Best Configuration: {best['name']}")
    print(f"   Throughput: {best['samples_per_second']:.1f} samples/second")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Optimize DataLoader configuration")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--dataset-size", type=int, default=5000, help="Dataset size")
    parser.add_argument("--num-batches", type=int, default=50, help="Batches to test")
    parser.add_argument("--no-simulate-io", action="store_true", help="Disable I/O simulation")
    args = parser.parse_args()
    
    logger = setup_logger("optimize_dataloader")
    
    # Device info
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
    else:
        logger.info("Running on CPU (for best results, use a GPU)")
    
    # Create dataset
    logger.info("Creating synthetic dataset...")
    dataset = SyntheticImageDataset(
        size=args.dataset_size,
        simulate_disk_io=not args.no_simulate_io,
    )
    
    # Run comparison
    results = compare_configurations(
        dataset=dataset,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
    )
    
    # Recommendations
    print("\n📋 RECOMMENDATIONS:")
    print("-" * 70)
    print("1. Use num_workers = CPU_cores // 2 (typically 4-8)")
    print("2. Always enable pin_memory=True when using GPU")
    print("3. Use persistent_workers=True to avoid worker startup overhead")
    print("4. Consider GPU prefetcher for maximum throughput")
    print("5. For large datasets, use memory-mapped files (MMapDataset)")
    print("-" * 70)


if __name__ == "__main__":
    main()
