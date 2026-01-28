#!/usr/bin/env python3
"""
Comprehensive GPU Optimization Benchmark Suite.

This script runs all benchmarks and generates a complete performance report.

Usage:
    python benchmarks/run_all.py [--output-dir ./benchmark_results]
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.profiling import TorchProfiler, CUDATimer, CUDABenchmark, MemoryTracker
from src.io_optimize import IOBenchmark
from src.utils.logger import setup_logger


def get_system_info() -> Dict[str, Any]:
    """Gather system information."""
    info = {
        "timestamp": datetime.now().isoformat(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info.update({
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_count": torch.cuda.device_count(),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
        })
    
    return info


def benchmark_memory_operations(device: torch.device) -> Dict[str, float]:
    """Benchmark memory operations."""
    print("\n📊 Benchmarking memory operations...")
    
    results = {}
    sizes = [1, 8, 64, 256, 1024]  # MB
    
    for size_mb in sizes:
        num_elements = (size_mb * 1024 * 1024) // 4  # float32
        
        # CPU to GPU transfer
        cpu_tensor = torch.randn(num_elements)
        
        benchmark = CUDABenchmark(warmup=5, iterations=20)
        result = benchmark.run(
            lambda: cpu_tensor.to(device),
            name=f"cpu_to_gpu_{size_mb}MB"
        )
        results[f"cpu_to_gpu_{size_mb}MB_ms"] = result.mean_ms
        
        # GPU to CPU transfer
        gpu_tensor = cpu_tensor.to(device)
        result = benchmark.run(
            lambda: gpu_tensor.cpu(),
            name=f"gpu_to_cpu_{size_mb}MB"
        )
        results[f"gpu_to_cpu_{size_mb}MB_ms"] = result.mean_ms
        
        del cpu_tensor, gpu_tensor
        torch.cuda.empty_cache()
    
    return results


def benchmark_compute_operations(device: torch.device) -> Dict[str, float]:
    """Benchmark compute operations."""
    print("\n📊 Benchmarking compute operations...")
    
    results = {}
    sizes = [512, 1024, 2048, 4096]
    
    benchmark = CUDABenchmark(warmup=10, iterations=50)
    
    for size in sizes:
        # Matrix multiplication
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        
        result = benchmark.run(
            lambda: torch.mm(a, b),
            name=f"matmul_{size}x{size}"
        )
        
        # Calculate TFLOPS
        flops = 2 * size * size * size  # 2n^3 for matmul
        tflops = flops / (result.mean_ms / 1000) / 1e12
        
        results[f"matmul_{size}_ms"] = result.mean_ms
        results[f"matmul_{size}_tflops"] = tflops
        
        # Convolution (if applicable)
        if size <= 1024:
            x = torch.randn(32, 64, size // 8, size // 8, device=device)
            conv = torch.nn.Conv2d(64, 128, 3, padding=1).to(device)
            
            result = benchmark.run(
                lambda: conv(x),
                name=f"conv2d_{size//8}x{size//8}"
            )
            results[f"conv2d_{size//8}_ms"] = result.mean_ms
            
            del x, conv
        
        del a, b
        torch.cuda.empty_cache()
    
    return results


def benchmark_memory_patterns(device: torch.device) -> Dict[str, Any]:
    """Analyze memory usage patterns."""
    print("\n📊 Analyzing memory patterns...")
    
    tracker = MemoryTracker()
    results = {}
    
    # Initial state
    tracker.snapshot("initial")
    
    # Allocate progressively
    tensors = []
    for i, size_mb in enumerate([64, 128, 256, 512]):
        num_elements = (size_mb * 1024 * 1024) // 4
        tensors.append(torch.randn(num_elements, device=device))
        tracker.snapshot(f"after_{size_mb}MB")
    
    # Peak memory
    peak = torch.cuda.max_memory_allocated(device) / (1024**2)
    results["peak_memory_mb"] = peak
    
    # Free memory
    for t in tensors:
        del t
    torch.cuda.empty_cache()
    tracker.snapshot("after_cleanup")
    
    # Memory freed
    final = torch.cuda.memory_allocated(device) / (1024**2)
    results["final_memory_mb"] = final
    
    # Snapshot progression
    snapshots = []
    for snap in tracker.snapshots:
        snapshots.append({
            "label": snap.label,
            "allocated_mb": snap.allocated_mb,
            "reserved_mb": snap.reserved_mb,
        })
    results["snapshots"] = snapshots
    
    return results


def run_resnet_benchmark(device: torch.device) -> Dict[str, float]:
    """Benchmark ResNet50 training iteration."""
    print("\n📊 Benchmarking ResNet50...")
    
    import torchvision.models as models
    
    model = models.resnet50(weights=None).to(device)
    model.train()
    
    batch_sizes = [8, 16, 32]
    if torch.cuda.get_device_properties(device).total_memory > 16e9:
        batch_sizes.append(64)
    
    results = {}
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    for batch_size in batch_sizes:
        try:
            input_tensor = torch.randn(batch_size, 3, 224, 224, device=device)
            labels = torch.randint(0, 1000, (batch_size,), device=device)
            
            # Warmup
            for _ in range(5):
                output = model(input_tensor)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            torch.cuda.synchronize()
            
            # Benchmark
            timer = CUDATimer()
            iterations = 20
            
            timer.start("total")
            for _ in range(iterations):
                output = model(input_tensor)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            torch.cuda.synchronize()
            total_time = timer.stop("total")
            
            avg_time = total_time / iterations
            throughput = batch_size / (avg_time / 1000)
            
            results[f"resnet50_bs{batch_size}_ms"] = avg_time
            results[f"resnet50_bs{batch_size}_imgs_per_sec"] = throughput
            
            del input_tensor, labels
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  Skipping batch_size={batch_size} (OOM)")
                torch.cuda.empty_cache()
            else:
                raise
    
    del model
    torch.cuda.empty_cache()
    
    return results


def generate_report(
    system_info: Dict[str, Any],
    memory_results: Dict[str, float],
    compute_results: Dict[str, float],
    memory_patterns: Dict[str, Any],
    resnet_results: Dict[str, float],
    output_path: str,
) -> str:
    """Generate comprehensive report."""
    
    lines = [
        "=" * 70,
        "GPU OPTIMIZATION BENCHMARK REPORT",
        "=" * 70,
        "",
        "System Information:",
        f"  Timestamp: {system_info['timestamp']}",
        f"  PyTorch: {system_info['pytorch_version']}",
    ]
    
    if system_info['cuda_available']:
        lines.extend([
            f"  CUDA: {system_info['cuda_version']}",
            f"  cuDNN: {system_info['cudnn_version']}",
            f"  GPU: {system_info['gpu_name']}",
            f"  GPU Memory: {system_info['gpu_memory_gb']:.1f} GB",
            f"  GPU Count: {system_info['gpu_count']}",
        ])
    
    lines.extend([
        "",
        "-" * 70,
        "Memory Transfer Performance:",
        "-" * 70,
    ])
    
    for key, value in memory_results.items():
        lines.append(f"  {key}: {value:.3f} ms")
    
    lines.extend([
        "",
        "-" * 70,
        "Compute Performance:",
        "-" * 70,
    ])
    
    for key, value in compute_results.items():
        if "tflops" in key:
            lines.append(f"  {key}: {value:.2f} TFLOPS")
        else:
            lines.append(f"  {key}: {value:.3f} ms")
    
    lines.extend([
        "",
        "-" * 70,
        "ResNet50 Training Performance:",
        "-" * 70,
    ])
    
    for key, value in resnet_results.items():
        if "imgs_per_sec" in key:
            lines.append(f"  {key}: {value:.1f} images/second")
        else:
            lines.append(f"  {key}: {value:.2f} ms")
    
    lines.extend([
        "",
        "-" * 70,
        "Memory Usage:",
        "-" * 70,
        f"  Peak memory: {memory_patterns['peak_memory_mb']:.1f} MB",
        f"  Final memory: {memory_patterns['final_memory_mb']:.1f} MB",
        "",
        "=" * 70,
    ])
    
    report = "\n".join(lines)
    
    # Save report
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)
    
    return report


def main():
    parser = argparse.ArgumentParser(description="GPU Optimization Benchmark Suite")
    parser.add_argument("--output-dir", type=str, default="./benchmark_results",
                        help="Output directory for results")
    args = parser.parse_args()
    
    logger = setup_logger("benchmark")
    
    print("=" * 70)
    print("GPU OPTIMIZATION BENCHMARK SUITE")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. Exiting.")
        return
    
    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Gather system info
    print("\n📊 Gathering system information...")
    system_info = get_system_info()
    print(f"  GPU: {system_info['gpu_name']}")
    print(f"  Memory: {system_info['gpu_memory_gb']:.1f} GB")
    
    # Run benchmarks
    memory_results = benchmark_memory_operations(device)
    compute_results = benchmark_compute_operations(device)
    memory_patterns = benchmark_memory_patterns(device)
    resnet_results = run_resnet_benchmark(device)
    
    # Generate report
    report_path = output_dir / "benchmark_report.txt"
    report = generate_report(
        system_info=system_info,
        memory_results=memory_results,
        compute_results=compute_results,
        memory_patterns=memory_patterns,
        resnet_results=resnet_results,
        output_path=str(report_path),
    )
    
    print("\n" + report)
    
    # Save JSON results
    all_results = {
        "system_info": system_info,
        "memory_transfer": memory_results,
        "compute": compute_results,
        "memory_patterns": memory_patterns,
        "resnet50": resnet_results,
    }
    
    json_path = output_dir / "benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Results saved to {output_dir}")
    print(f"  Report: {report_path}")
    print(f"  JSON: {json_path}")


if __name__ == "__main__":
    main()
