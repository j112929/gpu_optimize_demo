#!/usr/bin/env python3
"""
Example: Profile ResNet model forward and backward passes.

This example demonstrates:
1. How to use TorchProfiler for comprehensive GPU profiling
2. How to analyze CUDA kernel execution
3. How to track memory usage
4. How to export traces for visualization

Usage:
    python examples/profile_resnet.py [--batch-size 32] [--output-dir ./traces]
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.profiling import TorchProfiler, CUDATimer, MemoryTracker
from src.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Profile ResNet model")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--image-size", type=int, default=224, help="Image size")
    parser.add_argument("--model", type=str, default="resnet50", 
                        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
                        help="Model architecture")
    parser.add_argument("--output-dir", type=str, default="./profiler_traces",
                        help="Output directory for traces")
    parser.add_argument("--num-iterations", type=int, default=10,
                        help="Number of iterations to profile")
    return parser.parse_args()


def create_model(model_name: str, device: torch.device) -> nn.Module:
    """Create and return the specified model."""
    model_fn = getattr(models, model_name)
    model = model_fn(weights=None)  # No pretrained weights for speed
    return model.to(device)


def profile_forward_backward(
    model: nn.Module,
    input_tensor: torch.Tensor,
    profiler: TorchProfiler,
    timer: CUDATimer,
    memory_tracker: MemoryTracker,
    num_iterations: int = 10,
):
    """Profile forward and backward passes."""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Warmup
    for _ in range(3):
        output = model(input_tensor)
        loss = criterion(output, torch.randint(0, 1000, (input_tensor.shape[0],), device=input_tensor.device))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    torch.cuda.synchronize()
    memory_tracker.reset_peak_stats()
    memory_tracker.snapshot("after_warmup")
    
    # Profiled iterations
    for i in range(num_iterations):
        timer.start(f"iteration_{i}")
        
        # Forward
        timer.start(f"forward_{i}")
        memory_tracker.snapshot(f"before_forward_{i}")
        
        output = model(input_tensor)
        
        torch.cuda.synchronize()
        timer.stop(f"forward_{i}")
        memory_tracker.snapshot(f"after_forward_{i}")
        
        # Loss computation
        labels = torch.randint(0, 1000, (input_tensor.shape[0],), device=input_tensor.device)
        loss = criterion(output, labels)
        
        # Backward
        timer.start(f"backward_{i}")
        memory_tracker.snapshot(f"before_backward_{i}")
        
        loss.backward()
        
        torch.cuda.synchronize()
        timer.stop(f"backward_{i}")
        memory_tracker.snapshot(f"after_backward_{i}")
        
        # Optimizer step
        timer.start(f"optimizer_{i}")
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        timer.stop(f"optimizer_{i}")
        
        timer.stop(f"iteration_{i}")
        
        profiler.step()
    
    memory_tracker.snapshot("final")


def main():
    args = parse_args()
    logger = setup_logger("profile_resnet")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This example requires a GPU.")
        return
    
    device = torch.device("cuda")
    logger.info(f"Using device: {torch.cuda.get_device_name()}")
    
    # Create model
    logger.info(f"Creating {args.model} model...")
    model = create_model(args.model, device)
    model.train()
    
    # Create input tensor
    input_tensor = torch.randn(
        args.batch_size, 3, args.image_size, args.image_size,
        device=device
    )
    
    logger.info(f"Input shape: {input_tensor.shape}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize profiling tools
    timer = CUDATimer()
    memory_tracker = MemoryTracker()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting profiling...")
    
    # Profile with TorchProfiler
    with TorchProfiler(
        output_dir=str(output_dir),
        trace_name=f"{args.model}_bs{args.batch_size}",
        wait_steps=0,
        warmup_steps=0,
        active_steps=args.num_iterations,
    ) as profiler:
        profile_forward_backward(
            model=model,
            input_tensor=input_tensor,
            profiler=profiler,
            timer=timer,
            memory_tracker=memory_tracker,
            num_iterations=args.num_iterations,
        )
    
    # Print results
    print("\n" + "=" * 70)
    print("PROFILING RESULTS")
    print("=" * 70)
    
    # Timing summary
    print("\nTiming Summary:")
    total_time = 0
    for i in range(args.num_iterations):
        iter_time = timer.elapsed(f"iteration_{i}")
        total_time += iter_time
        
        if i < 3 or i == args.num_iterations - 1:  # Show first 3 and last
            fwd_time = timer.elapsed(f"forward_{i}")
            bwd_time = timer.elapsed(f"backward_{i}")
            opt_time = timer.elapsed(f"optimizer_{i}")
            print(f"  Iteration {i}: {iter_time:.2f}ms "
                  f"(fwd: {fwd_time:.2f}ms, bwd: {bwd_time:.2f}ms, opt: {opt_time:.2f}ms)")
    
    avg_time = total_time / args.num_iterations
    throughput = args.batch_size / (avg_time / 1000)
    
    print(f"\nAverage iteration time: {avg_time:.2f}ms")
    print(f"Throughput: {throughput:.1f} images/second")
    
    # Memory summary
    print("\n" + memory_tracker.summary())
    
    # Profiler summary
    if profiler.result:
        print("\n" + profiler.result.summary())
    
    logger.info(f"Traces saved to {output_dir}")
    logger.info("Open Chrome and go to chrome://tracing to visualize the trace file")


if __name__ == "__main__":
    main()
