#!/usr/bin/env python3
"""
Example: Training Optimization

Demonstrates training optimizations including:
- torch.compile for 30-200% speedup
- Mixed precision (AMP) for 2x speed, 50% memory
- CUDA Graphs for low-latency inference
- Gradient checkpointing for memory savings

Usage:
    python examples/training_optimization.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import time


# =============================================================================
# Demo Model
# =============================================================================

class TransformerBlock(nn.Module):
    """Simple transformer block for demo."""
    
    def __init__(self, hidden_size: int = 768, num_heads: int = 12):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
    
    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class DemoModel(nn.Module):
    """Demo model with transformer blocks."""
    
    def __init__(self, num_layers: int = 6, hidden_size: int = 768):
        super().__init__()
        self.embed = nn.Linear(hidden_size, hidden_size)
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size) for _ in range(num_layers)
        ])
        self.head = nn.Linear(hidden_size, 1000)
    
    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x.mean(dim=1))


# =============================================================================
# Demos
# =============================================================================

def demo_torch_compile():
    """Demonstrate torch.compile speedup."""
    print("\n" + "=" * 60)
    print("TORCH.COMPILE DEMO")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        return
    
    from src.compile import compile_model, benchmark_compile
    
    # Create model
    model = DemoModel(num_layers=4).cuda().eval()
    x = torch.randn(8, 128, 768, device="cuda")
    
    print("\n📊 Benchmarking compilation modes...")
    
    # Benchmark different modes
    results = benchmark_compile(
        model,
        x,
        modes=["eager", "default", "reduce-overhead"],
        warmup=5,
        iterations=50,
    )
    
    print("\n✅ torch.compile provides significant speedups!")
    print("   Use 'reduce-overhead' for inference")
    print("   Use 'max-autotune' for maximum throughput")


def demo_mixed_precision():
    """Demonstrate mixed precision training."""
    print("\n" + "=" * 60)
    print("MIXED PRECISION (AMP) DEMO")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        return
    
    from src.training.amp import AMPTrainer, estimate_memory_savings
    
    # Create model and optimizer
    model = DemoModel(num_layers=6).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Estimate memory savings
    savings = estimate_memory_savings(model)
    print(f"\n📊 Memory Savings Estimate:")
    print(f"   FP32 total:    {savings['fp32_total_gb']:.2f} GB")
    print(f"   FP16 total:    {savings['fp16_total_gb']:.2f} GB")
    print(f"   Savings:       {savings['savings_percent']:.0f}%")
    
    # Training with AMP
    amp_trainer = AMPTrainer()
    
    x = torch.randn(16, 128, 768, device="cuda")
    y = torch.randint(0, 1000, (16,), device="cuda")
    
    print("\n📊 Training with AMP:")
    
    # Warmup
    for _ in range(3):
        with amp_trainer.autocast():
            output = model(x)
            loss = criterion(output, y)
        amp_trainer.backward(loss)
        amp_trainer.step(optimizer)
        optimizer.zero_grad()
    
    torch.cuda.synchronize()
    
    # Benchmark AMP
    start = time.perf_counter()
    for _ in range(10):
        with amp_trainer.autocast():
            output = model(x)
            loss = criterion(output, y)
        amp_trainer.backward(loss)
        amp_trainer.step(optimizer)
        optimizer.zero_grad()
    torch.cuda.synchronize()
    amp_time = (time.perf_counter() - start) / 10 * 1000
    
    print(f"   AMP step time: {amp_time:.2f}ms")
    print(f"   Grad scale:    {amp_trainer.scale:.0f}")
    
    print("\n✅ Mixed precision provides ~2x speedup and 50% memory savings!")


def demo_cuda_graphs():
    """Demonstrate CUDA graphs for inference."""
    print("\n" + "=" * 60)
    print("CUDA GRAPHS DEMO")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        return
    
    from src.compile.cuda_graphs import CUDAGraphWrapper, benchmark_cuda_graphs
    
    # Create model
    model = DemoModel(num_layers=4).cuda().eval()
    x = torch.randn(1, 64, 768, device="cuda")
    
    print("\n📊 Benchmarking CUDA Graphs vs Eager...")
    
    results = benchmark_cuda_graphs(
        model,
        input_shape=(1, 64, 768),
        iterations=500,
        warmup=50,
    )
    
    print(f"\n✅ CUDA Graphs provide {results['speedup']:.2f}x speedup!")
    print("   Best for small batch inference")


def demo_gradient_checkpointing():
    """Demonstrate gradient checkpointing."""
    print("\n" + "=" * 60)
    print("GRADIENT CHECKPOINTING DEMO")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        return
    
    from src.training.gradient import gradient_checkpoint_model
    from src.memory.profiling import track_memory
    
    # Model without checkpointing
    model_no_ckpt = DemoModel(num_layers=8).cuda()
    x = torch.randn(8, 256, 768, device="cuda")
    
    torch.cuda.reset_peak_memory_stats()
    
    print("\n📊 Memory Usage Comparison:")
    
    # Without checkpointing
    with track_memory("no_checkpoint") as no_ckpt:
        output = model_no_ckpt(x)
        loss = output.sum()
        loss.backward()
    
    print(f"\n   Without checkpointing:")
    print(f"     Peak memory: {no_ckpt.peak_mb:.1f} MB")
    
    # With checkpointing
    del model_no_ckpt, output, loss
    torch.cuda.empty_cache()
    
    model_ckpt = DemoModel(num_layers=8).cuda()
    
    # Apply checkpointing
    from torch.utils.checkpoint import checkpoint
    
    original_forward = model_ckpt.blocks[0].forward
    
    # Checkpoint half the blocks
    for i in range(0, len(model_ckpt.blocks), 2):
        block = model_ckpt.blocks[i]
        orig_forward = block.forward
        def make_ckpt_forward(orig):
            def ckpt_forward(x):
                return checkpoint(orig, x, use_reentrant=False)
            return ckpt_forward
        block.forward = make_ckpt_forward(orig_forward)
    
    with track_memory("with_checkpoint") as with_ckpt:
        output = model_ckpt(x)
        loss = output.sum()
        loss.backward()
    
    print(f"\n   With checkpointing (50% blocks):")
    print(f"     Peak memory: {with_ckpt.peak_mb:.1f} MB")
    
    savings = (no_ckpt.peak_mb - with_ckpt.peak_mb) / no_ckpt.peak_mb * 100
    print(f"\n   Memory savings: {savings:.1f}%")
    
    print("\n✅ Gradient checkpointing trades compute for memory!")


def demo_fsdp():
    """Demonstrate FSDP configuration."""
    print("\n" + "=" * 60)
    print("FSDP (Fully Sharded Data Parallel) DEMO")
    print("=" * 60)
    
    from src.training.distributed import FSDPConfig
    
    print("\n📊 FSDP Configuration Options:")
    
    # Full sharding
    full_shard = FSDPConfig(
        sharding_strategy="full_shard",
        mixed_precision=True,
        precision="bf16",
        cpu_offload=False,
    )
    print(f"\n   Full Shard (max memory savings):")
    print(f"     Strategy: {full_shard.sharding_strategy}")
    print(f"     Mixed precision: {full_shard.precision}")
    
    # Hybrid sharding
    hybrid = FSDPConfig(
        sharding_strategy="hybrid_shard",
        mixed_precision=True,
        activation_checkpointing=True,
    )
    print(f"\n   Hybrid Shard (balance of speed/memory):")
    print(f"     Strategy: {hybrid.sharding_strategy}")
    print(f"     Activation checkpointing: {hybrid.activation_checkpointing}")
    
    # CPU offload
    offload = FSDPConfig(
        sharding_strategy="full_shard",
        cpu_offload=True,
    )
    print(f"\n   With CPU Offload (train huge models):")
    print(f"     CPU offload: {offload.cpu_offload}")
    
    print("\n✅ FSDP enables training models of any size!")
    print("   Use with: torchrun --nproc_per_node=8 train.py")


def main():
    print("=" * 60)
    print("TRAINING OPTIMIZATION DEMO")
    print("=" * 60)
    
    print("\nThis demo showcases various training optimizations:")
    print("  1. torch.compile - 30-200% speedup")
    print("  2. Mixed Precision - 2x speed, 50% memory")
    print("  3. CUDA Graphs - Low-latency inference")
    print("  4. Gradient Checkpointing - Trade compute for memory")
    print("  5. FSDP - Distributed training for large models")
    
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    demo_torch_compile()
    demo_mixed_precision()
    demo_cuda_graphs()
    demo_gradient_checkpointing()
    demo_fsdp()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
✅ torch.compile     → 30-200% speedup (one line change!)
✅ Mixed Precision   → 2x speed, 50% memory
✅ CUDA Graphs       → 10-30% latency reduction  
✅ Grad Checkpoint   → 50-70% memory savings
✅ FSDP/DeepSpeed    → Train 10x larger models
    """)


if __name__ == "__main__":
    main()
