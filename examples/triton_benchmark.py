#!/usr/bin/env python3
"""
Example: Triton Kernels Benchmark

Demonstrates Triton kernels and compares with PyTorch implementations.

Usage:
    python examples/triton_benchmark.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F


def benchmark_fused_ops():
    """Benchmark fused operations."""
    from src.triton_kernels.fused_ops import fused_gelu, fused_softmax, fused_silu
    from src.triton_kernels.benchmark import compare_with_pytorch
    
    print("\n" + "=" * 60)
    print("FUSED OPERATIONS BENCHMARK")
    print("=" * 60)
    
    # GELU
    x = torch.randn(4096, 4096, device='cuda', dtype=torch.float32)
    
    result = compare_with_pytorch(
        triton_fn=lambda: fused_gelu(x),
        pytorch_fn=lambda: F.gelu(x),
    )
    print(f"\nGELU (4096x4096): {result['summary']}")
    print(f"  Triton: {result['triton'].mean_ms:.3f}ms")
    print(f"  PyTorch: {result['pytorch'].mean_ms:.3f}ms")
    
    # Softmax
    x = torch.randn(1024, 8192, device='cuda', dtype=torch.float32)
    
    result = compare_with_pytorch(
        triton_fn=lambda: fused_softmax(x),
        pytorch_fn=lambda: F.softmax(x, dim=-1),
    )
    print(f"\nSoftmax (1024x8192): {result['summary']}")
    print(f"  Triton: {result['triton'].mean_ms:.3f}ms")
    print(f"  PyTorch: {result['pytorch'].mean_ms:.3f}ms")
    
    # SiLU
    x = torch.randn(4096, 4096, device='cuda', dtype=torch.float32)
    
    result = compare_with_pytorch(
        triton_fn=lambda: fused_silu(x),
        pytorch_fn=lambda: F.silu(x),
    )
    print(f"\nSiLU (4096x4096): {result['summary']}")


def benchmark_attention():
    """Benchmark attention implementations."""
    from src.triton_kernels.attention import flash_attention_v2
    from src.triton_kernels.benchmark import TritonBenchmark
    
    print("\n" + "=" * 60)
    print("ATTENTION BENCHMARK")
    print("=" * 60)
    
    bench = TritonBenchmark(warmup=10, iterations=50)
    
    for seq_len in [512, 1024, 2048, 4096]:
        batch, heads, head_dim = 2, 8, 64
        
        q = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
        k = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
        v = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
        
        # Flash Attention
        flash_result = bench.run(
            lambda: flash_attention_v2(q, k, v, causal=True),
            name=f"flash_attn_{seq_len}",
        )
        
        # Standard attention (only for shorter sequences)
        if seq_len <= 1024:
            def standard_attn():
                scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
                mask = torch.triu(torch.ones(seq_len, seq_len, device='cuda'), diagonal=1).bool()
                scores.masked_fill_(mask, float('-inf'))
                attn = F.softmax(scores, dim=-1)
                return torch.matmul(attn, v)
            
            std_result = bench.run(standard_attn, name=f"std_attn_{seq_len}")
            speedup = std_result.mean_ms / flash_result.mean_ms
            print(f"\nSeq={seq_len}: Flash {flash_result.mean_ms:.3f}ms vs Standard {std_result.mean_ms:.3f}ms ({speedup:.2f}x)")
        else:
            print(f"\nSeq={seq_len}: Flash Attention {flash_result.mean_ms:.3f}ms")


def benchmark_matmul():
    """Benchmark matrix multiplication."""
    from src.triton_kernels.matmul import triton_matmul
    from src.triton_kernels.benchmark import compare_with_pytorch
    
    print("\n" + "=" * 60)
    print("MATRIX MULTIPLICATION BENCHMARK")
    print("=" * 60)
    
    for size in [1024, 2048, 4096]:
        a = torch.randn(size, size, device='cuda', dtype=torch.float32)
        b = torch.randn(size, size, device='cuda', dtype=torch.float32)
        
        flops = 2 * size * size * size
        
        result = compare_with_pytorch(
            triton_fn=lambda: triton_matmul(a, b),
            pytorch_fn=lambda: torch.matmul(a, b),
        )
        
        tflops = flops / result['triton'].mean_ms / 1e9
        print(f"\nMatmul {size}x{size}: {result['summary']}")
        print(f"  Triton: {result['triton'].mean_ms:.3f}ms ({tflops:.2f} TFLOPS)")


def main():
    print("=" * 60)
    print("TRITON KERNELS BENCHMARK")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA required for Triton benchmarks")
        return
    
    print(f"GPU: {torch.cuda.get_device_name()}")
    
    try:
        import triton
        print(f"Triton version: {triton.__version__}")
    except ImportError:
        print("Triton not installed. Install with: pip install triton")
        return
    
    benchmark_fused_ops()
    benchmark_attention()
    benchmark_matmul()
    
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
