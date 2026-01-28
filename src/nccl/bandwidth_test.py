"""
Bandwidth Test - Measure GPU-to-GPU communication bandwidth.

This module provides tools for measuring NCCL bandwidth between GPUs,
testing different collective operations and message sizes.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist


@dataclass
class BandwidthResult:
    """Result of a bandwidth test."""
    
    operation: str
    message_size_mb: float
    time_ms: float
    bandwidth_gbps: float
    
    # For multiple iterations
    iterations: int = 1
    std_ms: float = 0.0
    
    def __repr__(self) -> str:
        return (
            f"BandwidthResult({self.operation}, "
            f"{self.message_size_mb:.1f}MB, "
            f"{self.bandwidth_gbps:.2f} Gbps)"
        )


@dataclass
class BandwidthTestResults:
    """Collection of bandwidth test results."""
    
    world_size: int
    local_rank: int
    device_name: str
    
    # Results per operation
    all_reduce: List[BandwidthResult] = field(default_factory=list)
    all_gather: List[BandwidthResult] = field(default_factory=list)
    reduce_scatter: List[BandwidthResult] = field(default_factory=list)
    broadcast: List[BandwidthResult] = field(default_factory=list)
    
    # Peak bandwidths
    peak_all_reduce_gbps: float = 0.0
    peak_all_gather_gbps: float = 0.0
    peak_reduce_scatter_gbps: float = 0.0
    peak_broadcast_gbps: float = 0.0
    
    def summary(self) -> str:
        """Generate test summary."""
        lines = [
            "=" * 70,
            "NCCL BANDWIDTH TEST RESULTS",
            "=" * 70,
            f"World Size: {self.world_size}",
            f"Device: {self.device_name}",
            "",
            "Peak Bandwidths:",
            f"  AllReduce:     {self.peak_all_reduce_gbps:.2f} Gbps",
            f"  AllGather:     {self.peak_all_gather_gbps:.2f} Gbps",
            f"  ReduceScatter: {self.peak_reduce_scatter_gbps:.2f} Gbps",
            f"  Broadcast:     {self.peak_broadcast_gbps:.2f} Gbps",
            "",
            "Detailed Results:",
        ]
        
        # AllReduce results
        if self.all_reduce:
            lines.append("\nAllReduce:")
            lines.append(f"  {'Size (MB)':>10} {'Time (ms)':>10} {'BW (Gbps)':>10}")
            for r in self.all_reduce:
                lines.append(f"  {r.message_size_mb:>10.1f} {r.time_ms:>10.3f} {r.bandwidth_gbps:>10.2f}")
        
        lines.append("=" * 70)
        return "\n".join(lines)


class BandwidthTest:
    """
    Test NCCL bandwidth between GPUs.
    
    Example:
        >>> test = BandwidthTest()
        >>> results = test.run_all()
        >>> print(results.summary())
    """
    
    def __init__(
        self,
        min_size_mb: float = 1.0,
        max_size_mb: float = 256.0,
        size_steps: int = 8,
        iterations: int = 50,
        warmup: int = 10,
    ):
        """
        Initialize bandwidth test.
        
        Args:
            min_size_mb: Minimum message size in MB
            max_size_mb: Maximum message size in MB
            size_steps: Number of size steps
            iterations: Number of iterations per test
            warmup: Number of warmup iterations
        """
        self.min_size_mb = min_size_mb
        self.max_size_mb = max_size_mb
        self.size_steps = size_steps
        self.iterations = iterations
        self.warmup = warmup
        
        # Generate size sequence (logarithmic)
        import math
        log_min = math.log2(min_size_mb)
        log_max = math.log2(max_size_mb)
        step = (log_max - log_min) / (size_steps - 1)
        self.sizes_mb = [2 ** (log_min + i * step) for i in range(size_steps)]
    
    def run_all(
        self,
        operations: Optional[List[str]] = None,
    ) -> BandwidthTestResults:
        """
        Run all bandwidth tests.
        
        Args:
            operations: List of operations to test.
                       Options: 'all_reduce', 'all_gather', 'reduce_scatter', 'broadcast'
                       
        Returns:
            BandwidthTestResults with all measurements
        """
        if not dist.is_initialized():
            raise RuntimeError("Distributed not initialized. Call dist.init_process_group first.")
        
        if operations is None:
            operations = ['all_reduce', 'all_gather', 'broadcast']
        
        world_size = dist.get_world_size()
        local_rank = dist.get_rank()
        device_name = torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"
        
        results = BandwidthTestResults(
            world_size=world_size,
            local_rank=local_rank,
            device_name=device_name,
        )
        
        for op in operations:
            if op == 'all_reduce':
                results.all_reduce = self._test_all_reduce()
                if results.all_reduce:
                    results.peak_all_reduce_gbps = max(r.bandwidth_gbps for r in results.all_reduce)
            
            elif op == 'all_gather':
                results.all_gather = self._test_all_gather()
                if results.all_gather:
                    results.peak_all_gather_gbps = max(r.bandwidth_gbps for r in results.all_gather)
            
            elif op == 'broadcast':
                results.broadcast = self._test_broadcast()
                if results.broadcast:
                    results.peak_broadcast_gbps = max(r.bandwidth_gbps for r in results.broadcast)
            
            elif op == 'reduce_scatter':
                results.reduce_scatter = self._test_reduce_scatter()
                if results.reduce_scatter:
                    results.peak_reduce_scatter_gbps = max(r.bandwidth_gbps for r in results.reduce_scatter)
        
        return results
    
    def _test_all_reduce(self) -> List[BandwidthResult]:
        """Test AllReduce bandwidth."""
        results = []
        world_size = dist.get_world_size()
        
        for size_mb in self.sizes_mb:
            # Create tensor
            num_elements = int(size_mb * 1024 * 1024 / 4)  # float32
            tensor = torch.randn(num_elements, dtype=torch.float32).cuda()
            
            # Warmup
            for _ in range(self.warmup):
                t = tensor.clone()
                dist.all_reduce(t)
                torch.cuda.synchronize()
            
            # Measure
            times = []
            for _ in range(self.iterations):
                t = tensor.clone()
                
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record()
                dist.all_reduce(t)
                end.record()
                
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
            
            # Calculate statistics
            avg_time = sum(times) / len(times)
            std_time = (sum((t - avg_time)**2 for t in times) / len(times)) ** 0.5
            
            # AllReduce moves 2*(n-1)/n * data_size for ring algorithm
            effective_bytes = 2 * (world_size - 1) / world_size * size_mb * 1024 * 1024
            bandwidth_gbps = effective_bytes * 8 / avg_time / 1e6
            
            results.append(BandwidthResult(
                operation='all_reduce',
                message_size_mb=size_mb,
                time_ms=avg_time,
                bandwidth_gbps=bandwidth_gbps,
                iterations=self.iterations,
                std_ms=std_time,
            ))
            
            del tensor
            torch.cuda.empty_cache()
        
        return results
    
    def _test_all_gather(self) -> List[BandwidthResult]:
        """Test AllGather bandwidth."""
        results = []
        world_size = dist.get_world_size()
        
        for size_mb in self.sizes_mb:
            # Each rank has size_mb, gather to world_size * size_mb total
            num_elements = int(size_mb * 1024 * 1024 / 4)
            tensor = torch.randn(num_elements, dtype=torch.float32).cuda()
            output = [torch.empty(num_elements, dtype=torch.float32).cuda() 
                      for _ in range(world_size)]
            
            # Warmup
            for _ in range(self.warmup):
                dist.all_gather(output, tensor.clone())
                torch.cuda.synchronize()
            
            # Measure
            times = []
            for _ in range(self.iterations):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record()
                dist.all_gather(output, tensor.clone())
                end.record()
                
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
            
            avg_time = sum(times) / len(times)
            std_time = (sum((t - avg_time)**2 for t in times) / len(times)) ** 0.5
            
            # AllGather moves (n-1)/n * data_size per GPU
            effective_bytes = (world_size - 1) / world_size * size_mb * 1024 * 1024
            bandwidth_gbps = effective_bytes * 8 / avg_time / 1e6
            
            results.append(BandwidthResult(
                operation='all_gather',
                message_size_mb=size_mb,
                time_ms=avg_time,
                bandwidth_gbps=bandwidth_gbps,
                iterations=self.iterations,
                std_ms=std_time,
            ))
            
            del tensor, output
            torch.cuda.empty_cache()
        
        return results
    
    def _test_broadcast(self) -> List[BandwidthResult]:
        """Test Broadcast bandwidth."""
        results = []
        
        for size_mb in self.sizes_mb:
            num_elements = int(size_mb * 1024 * 1024 / 4)
            tensor = torch.randn(num_elements, dtype=torch.float32).cuda()
            
            # Warmup
            for _ in range(self.warmup):
                t = tensor.clone()
                dist.broadcast(t, src=0)
                torch.cuda.synchronize()
            
            # Measure
            times = []
            for _ in range(self.iterations):
                t = tensor.clone()
                
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record()
                dist.broadcast(t, src=0)
                end.record()
                
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
            
            avg_time = sum(times) / len(times)
            std_time = (sum((t - avg_time)**2 for t in times) / len(times)) ** 0.5
            
            # Broadcast moves message_size bytes
            bandwidth_gbps = size_mb * 1024 * 1024 * 8 / avg_time / 1e6
            
            results.append(BandwidthResult(
                operation='broadcast',
                message_size_mb=size_mb,
                time_ms=avg_time,
                bandwidth_gbps=bandwidth_gbps,
                iterations=self.iterations,
                std_ms=std_time,
            ))
            
            del tensor
            torch.cuda.empty_cache()
        
        return results
    
    def _test_reduce_scatter(self) -> List[BandwidthResult]:
        """Test ReduceScatter bandwidth."""
        results = []
        world_size = dist.get_world_size()
        
        for size_mb in self.sizes_mb:
            # Total size is world_size * size_mb, each rank gets size_mb
            total_elements = int(size_mb * 1024 * 1024 / 4) * world_size
            per_rank_elements = total_elements // world_size
            
            input_tensor = torch.randn(total_elements, dtype=torch.float32).cuda()
            output_tensor = torch.empty(per_rank_elements, dtype=torch.float32).cuda()
            
            # Warmup
            for _ in range(self.warmup):
                inp = input_tensor.clone()
                out = output_tensor.clone()
                dist.reduce_scatter(out, list(inp.split(per_rank_elements)))
                torch.cuda.synchronize()
            
            # Measure
            times = []
            for _ in range(self.iterations):
                inp = input_tensor.clone()
                out = output_tensor.clone()
                
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record()
                dist.reduce_scatter(out, list(inp.split(per_rank_elements)))
                end.record()
                
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
            
            avg_time = sum(times) / len(times)
            std_time = (sum((t - avg_time)**2 for t in times) / len(times)) ** 0.5
            
            effective_bytes = (world_size - 1) / world_size * size_mb * world_size * 1024 * 1024
            bandwidth_gbps = effective_bytes * 8 / avg_time / 1e6
            
            results.append(BandwidthResult(
                operation='reduce_scatter',
                message_size_mb=size_mb * world_size,
                time_ms=avg_time,
                bandwidth_gbps=bandwidth_gbps,
                iterations=self.iterations,
                std_ms=std_time,
            ))
            
            del input_tensor, output_tensor
            torch.cuda.empty_cache()
        
        return results


def run_bandwidth_test(
    operation: str = 'all_reduce',
    size_mb: float = 64.0,
    iterations: int = 50,
) -> BandwidthResult:
    """
    Quick bandwidth test for a single operation and size.
    
    Args:
        operation: One of 'all_reduce', 'all_gather', 'broadcast', 'reduce_scatter'
        size_mb: Message size in MB
        iterations: Number of iterations
        
    Returns:
        BandwidthResult
    """
    test = BandwidthTest(
        min_size_mb=size_mb,
        max_size_mb=size_mb,
        size_steps=1,
        iterations=iterations,
    )
    
    results = test.run_all(operations=[operation])
    
    if operation == 'all_reduce' and results.all_reduce:
        return results.all_reduce[0]
    elif operation == 'all_gather' and results.all_gather:
        return results.all_gather[0]
    elif operation == 'broadcast' and results.broadcast:
        return results.broadcast[0]
    elif operation == 'reduce_scatter' and results.reduce_scatter:
        return results.reduce_scatter[0]
    
    raise ValueError(f"No results for operation: {operation}")
