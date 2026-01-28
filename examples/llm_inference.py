#!/usr/bin/env python3
"""
Example: LLM Inference Optimization

Demonstrates quantization, LoRA, and KV-cache optimization for
efficient large language model inference.

Usage:
    python examples/llm_inference.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn


def demo_quantization():
    """Demonstrate INT8/INT4 quantization."""
    from src.triton_kernels.quantization import (
        quantize_int8, dequantize_int8,
        quantize_int4, dequantize_int4,
        int8_matmul, QuantizedLinear,
    )
    
    print("\n" + "=" * 60)
    print("QUANTIZATION DEMO")
    print("=" * 60)
    
    # Create a test tensor
    x = torch.randn(1024, 1024, device='cuda')
    
    # INT8 Quantization
    print("\n📊 INT8 Quantization:")
    x_int8, scale = quantize_int8(x)
    x_recovered = dequantize_int8(x_int8, scale)
    
    mse = ((x - x_recovered) ** 2).mean().item()
    compression = x.numel() * 4 / (x_int8.numel() * 1)  # FP32 -> INT8
    print(f"   Original: {x.shape}, {x.dtype}")
    print(f"   Quantized: {x_int8.shape}, {x_int8.dtype}")
    print(f"   Compression: {compression:.1f}x")
    print(f"   MSE: {mse:.6f}")
    
    # INT4 Quantization
    print("\n📊 INT4 Quantization:")
    x_int4, scale4 = quantize_int4(x)
    x_recovered4 = dequantize_int4(x_int4, scale4, x.numel()).view(x.shape)
    
    mse4 = ((x - x_recovered4) ** 2).mean().item()
    compression4 = x.numel() * 4 / (x_int4.numel() * 1)  # FP32 -> INT4 packed
    print(f"   Quantized shape: {x_int4.shape} (packed, 2 values/byte)")
    print(f"   Compression: {compression4:.1f}x")
    print(f"   MSE: {mse4:.6f}")
    
    # Quantized Matrix Multiplication
    print("\n📊 INT8 MatMul:")
    a = torch.randn(512, 512, device='cuda')
    b = torch.randn(512, 512, device='cuda')
    
    a_int8, sa = quantize_int8(a)
    b_int8, sb = quantize_int8(b)
    
    # Benchmark
    import time
    
    # FP32 matmul
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        c_fp32 = torch.mm(a, b)
    torch.cuda.synchronize()
    fp32_time = time.perf_counter() - start
    
    # INT8 matmul
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        c_int8 = int8_matmul(a_int8, sa, b_int8, sb)
    torch.cuda.synchronize()
    int8_time = time.perf_counter() - start
    
    print(f"   FP32 time: {fp32_time*10:.2f}ms")
    print(f"   INT8 time: {int8_time*10:.2f}ms")
    print(f"   Speedup: {fp32_time/int8_time:.2f}x")


def demo_lora():
    """Demonstrate LoRA fused kernels."""
    from src.triton_kernels.lora import LoRALinear, QLoRALinear
    
    print("\n" + "=" * 60)
    print("LORA DEMO")
    print("=" * 60)
    
    # Create base linear layer
    in_features, out_features = 768, 3072
    base_linear = nn.Linear(in_features, out_features).cuda()
    
    # Wrap with LoRA
    lora_linear = LoRALinear(base_linear, rank=8, alpha=16.0)
    
    print("\n📊 LoRA Linear Layer:")
    print(f"   Base parameters: {sum(p.numel() for p in base_linear.parameters()):,}")
    print(f"   LoRA A shape: {lora_linear.lora_a.shape}")
    print(f"   LoRA B shape: {lora_linear.lora_b.shape}")
    lora_params = lora_linear.lora_a.numel() + lora_linear.lora_b.numel()
    print(f"   LoRA parameters: {lora_params:,} ({100*lora_params/(in_features*out_features):.2f}% of base)")
    
    # Test forward pass
    x = torch.randn(32, 128, in_features, device='cuda')
    output = lora_linear(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    
    # Merge weights for inference
    print("\n📊 Merging LoRA weights for inference:")
    merged = lora_linear.merge_weights().cuda()
    
    output_merged = merged(x)
    diff = (output - output_merged).abs().max().item()
    print(f"   Max difference after merge: {diff:.8f}")
    print(f"   ✅ Merge successful!" if diff < 1e-5 else "   ⚠️ Merge has numerical differences")


def demo_kv_cache():
    """Demonstrate KV-cache optimization."""
    from src.triton_kernels.kv_cache import (
        KVCache, KVCacheConfig,
        PagedKVCache, SlidingWindowKVCache,
        kv_cache_attention,
    )
    
    print("\n" + "=" * 60)
    print("KV-CACHE DEMO")
    print("=" * 60)
    
    # Standard KV-Cache
    print("\n📊 Standard KV-Cache:")
    config = KVCacheConfig(
        num_layers=32,
        num_heads=32,
        head_dim=128,
        max_seq_len=2048,
        batch_size=1,
    )
    cache = KVCache(config)
    print(f"   Layers: {config.num_layers}")
    print(f"   Heads: {config.num_heads}")
    print(f"   Head dim: {config.head_dim}")
    print(f"   Max seq len: {config.max_seq_len}")
    print(f"   Memory usage: {cache.memory_usage_mb():.1f} MB")
    
    # Simulate prefill + decode
    batch, heads, head_dim = 1, 32, 128
    
    # Prefill: 512 tokens
    prefill_len = 512
    q = torch.randn(batch, heads, prefill_len, head_dim, device='cuda', dtype=torch.float16)
    k = torch.randn(batch, heads, prefill_len, head_dim, device='cuda', dtype=torch.float16)
    v = torch.randn(batch, heads, prefill_len, head_dim, device='cuda', dtype=torch.float16)
    
    output = kv_cache_attention(q, cache, layer_idx=0, new_key=k, new_value=v)
    print(f"\n   Prefill ({prefill_len} tokens):")
    print(f"   Output shape: {output.shape}")
    print(f"   Cache seq_len: {cache.seq_len}")
    
    # Decode: 1 token at a time
    for i in range(3):
        q_new = torch.randn(batch, heads, 1, head_dim, device='cuda', dtype=torch.float16)
        k_new = torch.randn(batch, heads, 1, head_dim, device='cuda', dtype=torch.float16)
        v_new = torch.randn(batch, heads, 1, head_dim, device='cuda', dtype=torch.float16)
        
        output = kv_cache_attention(q_new, cache, layer_idx=0, new_key=k_new, new_value=v_new)
    
    print(f"   After 3 decode steps: cache seq_len = {cache.seq_len}")
    
    # Sliding Window Cache
    print("\n📊 Sliding Window KV-Cache (Mistral-style):")
    sw_cache = SlidingWindowKVCache(
        num_layers=32,
        num_heads=32,
        head_dim=128,
        window_size=4096,
    )
    print(f"   Window size: 4096")
    print(f"   Memory (fixed): {sw_cache.memory_usage_mb():.1f} MB")
    print(f"   vs Standard (8K seq): {cache.memory_usage_mb() * 4:.1f} MB")
    print(f"   Memory savings: {(1 - sw_cache.memory_usage_mb()/(cache.memory_usage_mb()*4))*100:.1f}%")
    
    # Paged KV-Cache
    print("\n📊 Paged KV-Cache (vLLM-style):")
    paged_cache = PagedKVCache(
        num_layers=32,
        num_heads=32,
        head_dim=128,
        block_size=16,
        num_blocks=1024,
    )
    print(f"   Block size: 16 tokens")
    print(f"   Total blocks: 1024")
    print(f"   Max tokens: {16 * 1024 // 32} per layer (shared pool)")


def main():
    print("=" * 60)
    print("LLM INFERENCE OPTIMIZATION DEMO")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA required")
        return
    
    print(f"GPU: {torch.cuda.get_device_name()}")
    
    try:
        import triton
        print(f"Triton: {triton.__version__}")
    except ImportError:
        print("Triton not installed")
        return
    
    demo_quantization()
    demo_lora()
    demo_kv_cache()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
✅ Quantization: 2-4x memory reduction
✅ LoRA: <1% trainable parameters  
✅ KV-Cache: Efficient inference memory
✅ Sliding Window: Fixed memory for long sequences
✅ Paged Attention: Dynamic memory allocation
    """)


if __name__ == "__main__":
    main()
