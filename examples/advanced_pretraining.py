"""
Advanced Pre-Training Optimizations Demo

Demonstrates the new pre-training optimization techniques:
- Flash Attention (v2/v3/SDPA)
- Gradient Compression
- Optimized DataLoader with CUDA prefetching
- Fused Operations

Usage:
    python examples/advanced_pretraining.py --mode attention
    python examples/advanced_pretraining.py --mode compression
    python examples/advanced_pretraining.py --mode dataloader
    python examples/advanced_pretraining.py --mode fused
    python examples/advanced_pretraining.py --mode all
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import time


# =============================================================================
# Flash Attention Demo
# =============================================================================

def attention_example():
    """Demonstrate Flash Attention optimization."""
    print("\n" + "="*60)
    print("⚡ Flash Attention Example")
    print("="*60)
    
    from src.training import (
        FlashAttention,
        FlashAttentionConfig,
        detect_flash_attention_version,
        create_attention_layer,
        MemoryEfficientAttentionContext,
    )
    
    # Detect available backend
    backend = detect_flash_attention_version()
    print(f"\nDetected attention backend: {backend}")
    
    # Configuration
    hidden_size = 2048
    num_heads = 32
    batch_size = 4
    seq_len = 2048
    
    # Create attention layer
    config = FlashAttentionConfig(
        use_flash_attn=True,
        num_heads=num_heads,
        num_kv_heads=8,  # GQA: 4 groups
        causal=True,
    )
    attn = FlashAttention(config, hidden_size)
    
    print(f"\nConfiguration:")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Num heads: {num_heads}")
    print(f"  Num KV heads: 8 (GQA)")
    print(f"  Sequence length: {seq_len}")
    
    # Create test tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.float16)
    k = torch.randn(batch_size, seq_len, hidden_size // 4, device=device, dtype=torch.float16)
    v = torch.randn(batch_size, seq_len, hidden_size // 4, device=device, dtype=torch.float16)
    
    attn = attn.to(device).half()
    
    # Warmup
    for _ in range(5):
        _ = attn(q, k, v)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    num_iters = 50
    for _ in range(num_iters):
        output = attn(q, k, v)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    elapsed = (time.perf_counter() - start) / num_iters * 1000
    print(f"\n⏱️ Attention Time: {elapsed:.2f} ms per forward pass")
    print(f"   Output shape: {output.shape}")
    
    # Memory efficient context
    print("\n🔧 Memory Efficient Attention Context:")
    with MemoryEfficientAttentionContext(enable_flash=True):
        output2 = attn(q, k, v)
    print("   Using memory-efficient attention path")


# =============================================================================
# Gradient Compression Demo
# =============================================================================

def compression_example():
    """Demonstrate gradient compression."""
    print("\n" + "="*60)
    print("📦 Gradient Compression Example")
    print("="*60)
    
    from src.training import (
        GradientCompressionConfig,
        TopKCompressor,
        RandomCompressor,
        QuantizationCompressor,
        PowerSGDCompressor,
        estimate_compression_ratio,
        create_gradient_compressor,
    )
    
    # Create test gradient
    gradient = torch.randn(1024, 1024)
    
    print("\n--- Top-K Compression (1% kept) ---")
    config = GradientCompressionConfig(method="topk", topk_ratio=0.01, error_feedback=True)
    compressor = TopKCompressor(config)
    
    compressed, context = compressor.compress(gradient, "test_layer")
    values, indices = compressed
    
    original_size = gradient.numel() * 4  # float32 = 4 bytes
    compressed_size = (values.numel() + indices.numel()) * 4
    ratio = original_size / compressed_size
    
    print(f"  Original size: {original_size / 1024:.1f} KB")
    print(f"  Compressed size: {compressed_size / 1024:.1f} KB")
    print(f"  Compression ratio: {ratio:.1f}×")
    
    # Decompress and check error
    decompressed = compressor.decompress(compressed, context)
    error = (gradient - decompressed).abs().mean()
    print(f"  Reconstruction error: {error:.4f}")
    
    print("\n--- Quantization (8-bit) ---")
    config = GradientCompressionConfig(method="quantize", quantize_bits=8)
    compressor = QuantizationCompressor(config)
    
    compressed, context = compressor.compress(gradient, "test_layer")
    compressed_size = compressed.numel()  # int8 = 1 byte
    ratio = original_size / compressed_size
    
    print(f"  Compression ratio: {ratio:.1f}×")
    
    decompressed = compressor.decompress(compressed, context)
    error = (gradient - decompressed).abs().mean()
    print(f"  Reconstruction error: {error:.4f}")
    
    print("\n--- 1-Bit SGD (Sign Only) ---")
    config = GradientCompressionConfig(method="quantize", quantize_bits=1)
    compressor = QuantizationCompressor(config)
    
    compressed, context = compressor.compress(gradient, "test_layer")
    print(f"  Compression ratio: 32× (theoretical)")
    
    print("\n--- PowerSGD (Low-Rank, rank=4) ---")
    config = GradientCompressionConfig(method="powersgd", powersgd_rank=4)
    compressor = PowerSGDCompressor(config)
    
    compressed, context = compressor.compress(gradient, "test_layer")
    P, Q = compressed
    
    compressed_size = (P.numel() + Q.numel()) * 4
    ratio = original_size / compressed_size
    print(f"  P shape: {P.shape}, Q shape: {Q.shape}")
    print(f"  Compression ratio: {ratio:.1f}×")
    
    print("\n📊 Theoretical Compression Ratios:")
    for method in ["topk", "random", "quantize", "powersgd"]:
        cfg = GradientCompressionConfig(method=method)
        ratio = estimate_compression_ratio(cfg)
        print(f"  {method}: {ratio:.1f}×")



# =============================================================================
# Helper Classes
# =============================================================================

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, size=10000, dim=512):
        self.size = size
        self.dim = dim
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return torch.randn(self.dim), torch.randint(0, 10, (1,))[0]

# =============================================================================
# Optimized DataLoader Demo
# =============================================================================

def dataloader_example():
    """Demonstrate optimized DataLoader."""
    print("\n" + "="*60)
    print("🚀 Optimized DataLoader Example")
    print("="*60)
    
    from src.training import (
        DataLoaderConfig,
        CUDAPrefetcher,
        create_dataloader,
        create_prefetched_loader,
        StreamingDataset,
        benchmark_dataloader,
    )
    import torch.utils.data as data
    
    dataset = DummyDataset()
    
    # Standard DataLoader
    print("\n--- Standard DataLoader ---")
    config = DataLoaderConfig(
        batch_size=64,
        num_workers=0,  # For demo purposes
        pin_memory=True,
    )
    loader = create_dataloader(dataset, config)
    
    print(f"  Batch size: {config.batch_size}")
    print(f"  Num workers: {config.num_workers}")
    print(f"  Pin memory: {config.pin_memory}")
    
    # Test iteration
    start = time.perf_counter()
    for i, (x, y) in enumerate(loader):
        if i >= 50:
            break
    standard_time = time.perf_counter() - start
    print(f"  Time for 50 batches: {standard_time*1000:.1f} ms")
    
    # Optimized config
    print("\n--- Optimized DataLoader (with prefetching) ---")
    config = DataLoaderConfig(
        batch_size=64,
        num_workers=2,
        prefetch_factor=4,
        pin_memory=True,
        persistent_workers=True,
    )
    loader = create_dataloader(dataset, config)
    
    print(f"  Prefetch factor: {config.prefetch_factor}")
    print(f"  Persistent workers: {config.persistent_workers}")
    
    start = time.perf_counter()
    for i, (x, y) in enumerate(loader):
        if i >= 50:
            break
    optimized_time = time.perf_counter() - start
    print(f"  Time for 50 batches: {optimized_time*1000:.1f} ms")
    
    if standard_time > 0:
        speedup = standard_time / optimized_time
        print(f"  Speedup: {speedup:.2f}×")
    
    # CUDA Prefetcher
    if torch.cuda.is_available():
        print("\n--- CUDA Prefetcher ---")
        print("  Overlaps data transfer with computation")
        print("  Uses CUDA streams for async transfer")
        print("  GPU never waits for data!")
    
    # Streaming dataset info
    print("\n--- Streaming Dataset ---")
    print("  Use for datasets that don't fit in memory")
    print("  Supports: .parquet, .jsonl, text files")
    print("  Automatic sharding across workers")


# =============================================================================
# Fused Operations Demo
# =============================================================================

def fused_example():
    """Demonstrate fused operations."""
    print("\n" + "="*60)
    print("🔥 Fused Operations Example")
    print("="*60)
    
    from src.training import (
        FusedLayerNorm,
        RMSNorm,
        FusedCrossEntropyLoss,
        FusedAdamW,
        FusedRoPE,
        FusedSwiGLU,
        benchmark_fused_ops,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_size = 4096
    batch_size = 4
    seq_len = 2048
    
    x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.float16)
    
    # RMSNorm vs LayerNorm
    print("\n--- RMSNorm vs LayerNorm ---")
    
    ln = nn.LayerNorm(hidden_size).to(device).half()
    rms = RMSNorm(hidden_size).to(device).half()
    
    # Warmup
    for _ in range(10):
        _ = ln(x)
        _ = rms(x)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark LayerNorm
    start = time.perf_counter()
    for _ in range(100):
        _ = ln(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ln_time = (time.perf_counter() - start) / 100 * 1000
    
    # Benchmark RMSNorm
    start = time.perf_counter()
    for _ in range(100):
        _ = rms(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    rms_time = (time.perf_counter() - start) / 100 * 1000
    
    print(f"  LayerNorm: {ln_time:.3f} ms")
    print(f"  RMSNorm: {rms_time:.3f} ms")
    print(f"  Speedup: {ln_time/rms_time:.2f}×")
    
    # Fused SwiGLU
    print("\n--- Fused SwiGLU ---")
    intermediate_size = int(hidden_size * 2.67)  # LLaMA-style
    swiglu = FusedSwiGLU(hidden_size, intermediate_size).to(device).half()
    
    # Compare to separate gate and up projection
    class SeparateSwiGLU(nn.Module):
        def __init__(self, h, i):
            super().__init__()
            self.gate = nn.Linear(h, i, bias=False)
            self.up = nn.Linear(h, i, bias=False)
            self.down = nn.Linear(i, h, bias=False)
        
        def forward(self, x):
            return self.down(F.silu(self.gate(x)) * self.up(x))
    
    separate = SeparateSwiGLU(hidden_size, intermediate_size).to(device).half()
    
    # Warmup
    for _ in range(10):
        _ = swiglu(x)
        _ = separate(x)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(100):
        _ = separate(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    separate_time = (time.perf_counter() - start) / 100 * 1000
    
    start = time.perf_counter()
    for _ in range(100):
        _ = swiglu(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    fused_time = (time.perf_counter() - start) / 100 * 1000
    
    print(f"  Separate projections: {separate_time:.3f} ms")
    print(f"  Fused SwiGLU: {fused_time:.3f} ms")
    print(f"  Speedup: {separate_time/fused_time:.2f}×")
    
    # Fused RoPE
    print("\n--- Fused RoPE ---")
    head_dim = 128
    rope = FusedRoPE(dim=head_dim, max_seq_len=8192).to(device)
    
    query = torch.randn(batch_size, seq_len, 32, head_dim, device=device, dtype=torch.float16)
    
    start = time.perf_counter()
    for _ in range(100):
        _ = rope(query)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    rope_time = (time.perf_counter() - start) / 100 * 1000
    
    print(f"  RoPE application: {rope_time:.3f} ms")
    
    # Fused CrossEntropy
    print("\n--- Fused CrossEntropy ---")
    vocab_size = 32000
    logits = torch.randn(batch_size * seq_len, vocab_size, device=device, dtype=torch.float16)
    labels = torch.randint(0, vocab_size, (batch_size * seq_len,), device=device)
    
    standard_ce = nn.CrossEntropyLoss()
    fused_ce = FusedCrossEntropyLoss()
    
    # Move to device
    logits_f32 = logits.float()
    
    start = time.perf_counter()
    for _ in range(100):
        _ = standard_ce(logits_f32, labels)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    standard_time = (time.perf_counter() - start) / 100 * 1000
    
    start = time.perf_counter()
    for _ in range(100):
        _ = fused_ce(logits_f32, labels)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    fused_time = (time.perf_counter() - start) / 100 * 1000
    
    print(f"  Standard CrossEntropy: {standard_time:.3f} ms")
    print(f"  Fused CrossEntropy: {fused_time:.3f} ms")
    
    # Fused AdamW
    print("\n--- Fused AdamW ---")
    model = nn.Linear(hidden_size, hidden_size).to(device)
    optimizer = FusedAdamW(model.parameters(), lr=1e-4)
    print("  FusedAdamW created with decoupled weight decay")


# =============================================================================
# ZeRO++ Demo
# =============================================================================

def zeroplus_example():
    """Demonstrate ZeRO++ features."""
    print("\n" + "="*60)
    print("💎 ZeRO++ Enhancements Example")
    print("="*60)
    
    from src.training import (
        ZeroPlusConfig,
        QuantizedCommunicator,
    )
    
    # Configuration
    config = ZeroPlusConfig(
        zero_quantized_weights=True,
        zero_quantized_gradients=True,
    )
    ds_config = config.to_deepspeed_config()
    print("\nGenerated DeepSpeed Configuration:")
    print(ds_config)
    
    # Simulate Quantized Communication
    print("\n--- Quantized Communication Simulation ---")
    
    # Create a large tensor
    tensor = torch.randn(1024, 1024)
    print(f"Original Tensor: {tensor.numel() * 4 / 1024:.2f} KB (FP32)")
    
    # Quantize
    q_tensor, scale, zero_point = QuantizedCommunicator.quantize(tensor, num_bits=8)
    print(f"Quantized Tensor: {q_tensor.numel() * 1 / 1024:.2f} KB (INT8)")
    
    compression_ratio = (tensor.numel() * 4) / (q_tensor.numel() * 1)
    print(f"Compression Ratio: {compression_ratio:.1f}x")
    
    # Dequantize
    recon = QuantizedCommunicator.dequantize(q_tensor, scale, zero_point)
    error = (tensor - recon).abs().mean()
    print(f"Reconstruction Error: {error:.4f}")


# =============================================================================
# 3D Parallelism Demo
# =============================================================================

def parallelism_example():
    """Demonstrate Tensor, Sequence, and Pipeline Parallelism."""
    print("\n" + "="*60)
    print("🧩 3D Parallelism Example (TP / SP / PP)")
    print("="*60)
    
    from src.training import (
        ColumnParallelLinear,
        RowParallelLinear,
        SequenceParallelWrapper,
        split_model_into_stages,
    )
    
    # 1. Tensor Parallelism
    print("\n--- Tensor Parallelism (TP) ---")
    print("Simulating TP World Size = 2")
    hidden_size = 64
    
    # Standard Linear: [H, H] weight
    standard = nn.Linear(hidden_size, hidden_size)
    print(f"Standard Linear Params: {sum(p.numel() for p in standard.parameters())}")
    
    # Column Parallel: [H/2, H] weight per GPU
    col_p = ColumnParallelLinear(hidden_size, hidden_size)
    print(f"Column Parallel Params (per GPU): {sum(p.numel() for p in col_p.parameters())}")
    
    # Row Parallel: [H, H/2] weight per GPU
    row_p = RowParallelLinear(hidden_size, hidden_size)
    print(f"Row Parallel Params (per GPU): {sum(p.numel() for p in row_p.parameters())}")
    
    # 2. Sequence Parallelism
    print("\n--- Sequence Parallelism (SP) ---")
    batch, seq, dim = 2, 1024, 64
    x = torch.randn(batch, seq, dim)
    
    sp_layer = SequenceParallelWrapper(nn.Linear(dim, dim))
    # In a real distributed setting, this would split 'seq' dimension
    print(f"Input Shape: {x.shape}")
    print(f"Module wrapped with SP: {sp_layer}")
    
    # 3. Pipeline Parallelism
    print("\n--- Pipeline Parallelism (PP) ---")
    model = nn.Sequential(
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 5)
    )
    
    stages = split_model_into_stages(model, num_stages=2)
    print(f"Split model into {len(stages)} stages:")
    for i, stage in enumerate(stages):
        print(f"  Stage {i}: {stage}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Advanced Pre-Training Optimizations Demo")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["attention", "compression", "dataloader", "fused", "zeroplus", "parallelism", "all"],
        help="Which example to run",
    )
    args = parser.parse_args()
    
    print("="*60)
    print("🚀 Advanced Pre-Training Optimization Demo")
    print("="*60)
    
    if torch.cuda.is_available():
        print(f"🖥️  GPU: {torch.cuda.get_device_name()}")
        print(f"📊 Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  Running on CPU (limited functionality)")
    
    examples = {
        "attention": attention_example,
        "compression": compression_example,
        "dataloader": dataloader_example,
        "fused": fused_example,
        "zeroplus": zeroplus_example,
        "parallelism": parallelism_example,
    }
    
    if args.mode == "all":
        for name, func in examples.items():
            try:
                func()
            except Exception as e:
                print(f"\n❌ Error in {name}: {e}")
    else:
        examples[args.mode]()
    
    print("\n" + "="*60)
    print("✅ Demo Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
