#!/usr/bin/env python3
"""
Example: Distributed training with NCCL profiling.

This example demonstrates:
1. Setting up distributed training with PyTorch
2. Profiling NCCL communication operations
3. Testing GPU-to-GPU bandwidth
4. Analyzing GPU topology

Usage (single-node multi-GPU):
    torchrun --nproc_per_node=2 examples/distributed_training.py

Usage (multi-node):
    # On node 0:
    torchrun --nnodes=2 --node_rank=0 --master_addr=<MASTER_IP> --master_port=29500 \\
        --nproc_per_node=4 examples/distributed_training.py
    
    # On node 1:
    torchrun --nnodes=2 --node_rank=1 --master_addr=<MASTER_IP> --master_port=29500 \\
        --nproc_per_node=4 examples/distributed_training.py
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nccl import CommProfiler, BandwidthTest, TopologyAnalyzer
from src.utils.logger import setup_logger


class SimpleModel(nn.Module):
    """Simple model for demonstration."""
    
    def __init__(self, hidden_size: int = 1024, num_layers: int = 4):
        super().__init__()
        
        layers = []
        for i in range(num_layers):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.LayerNorm(hidden_size),
            ])
        
        self.layers = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_size, 1000)
    
    def forward(self, x):
        x = self.layers(x)
        return self.output(x)


def setup_distributed():
    """Initialize distributed training."""
    if not dist.is_initialized():
        # Check if launched with torchrun
        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            rank = int(os.environ.get("RANK", 0))
            
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
            )
            
            torch.cuda.set_device(local_rank)
            
            return local_rank, world_size, rank
        else:
            # Single GPU fallback
            return 0, 1, 0
    
    return (
        dist.get_rank() % torch.cuda.device_count(),
        dist.get_world_size(),
        dist.get_rank(),
    )


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def print_rank0(msg: str, rank: int):
    """Print only on rank 0."""
    if rank == 0:
        print(msg)


def run_topology_analysis(rank: int):
    """Analyze and print GPU topology."""
    if rank != 0:
        return
    
    print("\n" + "=" * 70)
    print("GPU TOPOLOGY ANALYSIS")
    print("=" * 70)
    
    try:
        analyzer = TopologyAnalyzer()
        topology = analyzer.analyze()
        print(topology.summary())
        
        # Get optimal ring order
        ring_order = analyzer.get_optimal_ring_order()
        print(f"\nOptimal ring order for AllReduce: {ring_order}")
    except Exception as e:
        print(f"Topology analysis failed: {e}")


def run_bandwidth_test(rank: int, world_size: int):
    """Run NCCL bandwidth tests."""
    print_rank0("\n" + "=" * 70, rank)
    print_rank0("NCCL BANDWIDTH TEST", rank)
    print_rank0("=" * 70, rank)
    
    if world_size < 2:
        print_rank0("Skipping bandwidth test (requires 2+ GPUs)", rank)
        return
    
    try:
        test = BandwidthTest(
            min_size_mb=1.0,
            max_size_mb=128.0,
            size_steps=5,
            iterations=20,
            warmup=5,
        )
        
        results = test.run_all(operations=['all_reduce', 'broadcast'])
        
        if rank == 0:
            print(results.summary())
    except Exception as e:
        print_rank0(f"Bandwidth test failed: {e}", rank)


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    input_tensor: torch.Tensor,
    criterion: nn.Module,
) -> float:
    """Perform a single training step."""
    optimizer.zero_grad()
    
    output = model(input_tensor)
    labels = torch.randint(0, 1000, (input_tensor.shape[0],), device=input_tensor.device)
    loss = criterion(output, labels)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()


def run_training_benchmark(
    rank: int,
    local_rank: int,
    world_size: int,
    batch_size: int = 64,
    hidden_size: int = 1024,
    num_iterations: int = 100,
):
    """Run training with NCCL profiling."""
    print_rank0("\n" + "=" * 70, rank)
    print_rank0("DISTRIBUTED TRAINING BENCHMARK", rank)
    print_rank0("=" * 70, rank)
    
    device = torch.device(f"cuda:{local_rank}")
    
    # Create model
    model = SimpleModel(hidden_size=hidden_size).to(device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Create input
    input_tensor = torch.randn(batch_size, hidden_size, device=device)
    
    # Warmup
    for _ in range(10):
        train_step(model, optimizer, input_tensor, criterion)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Profile training
    print_rank0(f"\nRunning {num_iterations} training iterations...", rank)
    
    with CommProfiler() as profiler:
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        
        for i in range(num_iterations):
            loss = train_step(model, optimizer, input_tensor, criterion)
        
        end_time.record()
        torch.cuda.synchronize()
        
        total_time = start_time.elapsed_time(end_time)
    
    # Print results
    if rank == 0:
        avg_time = total_time / num_iterations
        samples_per_second = (batch_size * world_size) / (avg_time / 1000)
        
        print(f"\nResults:")
        print(f"  World size: {world_size}")
        print(f"  Batch size per GPU: {batch_size}")
        print(f"  Effective batch size: {batch_size * world_size}")
        print(f"  Total time: {total_time:.2f} ms")
        print(f"  Avg iteration time: {avg_time:.2f} ms")
        print(f"  Throughput: {samples_per_second:.1f} samples/second")
        
        # Print communication profile
        if profiler.profile:
            print(profiler.profile.summary())
    
    # Synchronize before cleanup
    if world_size > 1:
        dist.barrier()


def main():
    parser = argparse.ArgumentParser(description="Distributed training with NCCL profiling")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per GPU")
    parser.add_argument("--hidden-size", type=int, default=1024, help="Model hidden size")
    parser.add_argument("--iterations", type=int, default=100, help="Training iterations")
    parser.add_argument("--skip-topology", action="store_true", help="Skip topology analysis")
    parser.add_argument("--skip-bandwidth", action="store_true", help="Skip bandwidth test")
    args = parser.parse_args()
    
    # Setup distributed
    local_rank, world_size, rank = setup_distributed()
    
    logger = setup_logger("distributed_training")
    
    if rank == 0:
        print("=" * 70)
        print("DISTRIBUTED TRAINING WITH NCCL PROFILING")
        print("=" * 70)
        print(f"World size: {world_size}")
        print(f"Local rank: {local_rank}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"CUDA version: {torch.version.cuda}")
    
    try:
        # Topology analysis (rank 0 only)
        if not args.skip_topology:
            run_topology_analysis(rank)
        
        # Synchronize before bandwidth test
        if world_size > 1:
            dist.barrier()
        
        # Bandwidth test
        if not args.skip_bandwidth:
            run_bandwidth_test(rank, world_size)
        
        # Synchronize before training
        if world_size > 1:
            dist.barrier()
        
        # Training benchmark
        run_training_benchmark(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            batch_size=args.batch_size,
            hidden_size=args.hidden_size,
            num_iterations=args.iterations,
        )
        
        print_rank0("\n✅ All benchmarks completed successfully!", rank)
        
    except Exception as e:
        print(f"[Rank {rank}] Error: {e}")
        raise
    
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
