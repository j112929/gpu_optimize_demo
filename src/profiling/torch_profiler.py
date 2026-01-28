"""
PyTorch Profiler Wrapper - Comprehensive GPU profiling with automatic trace export.

This module provides a high-level interface to PyTorch's built-in profiler with:
- Automatic trace export to Chrome/TensorBoard format
- Memory profiling
- Kernel-level operation analysis
- Easy-to-use context manager API
"""

import os
import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler


@dataclass
class ProfileConfig:
    """Configuration for PyTorch Profiler."""
    
    # Output settings
    output_dir: str = "./profiler_traces"
    trace_name: Optional[str] = None
    
    # Profiling activities
    profile_cuda: bool = True
    profile_cpu: bool = True
    profile_memory: bool = True
    
    # Schedule settings (for iterative profiling)
    wait_steps: int = 1
    warmup_steps: int = 1
    active_steps: int = 3
    repeat: int = 1
    
    # Additional options
    record_shapes: bool = True
    with_stack: bool = False
    with_flops: bool = True
    with_modules: bool = True
    
    # Export options
    export_chrome_trace: bool = True
    export_tensorboard: bool = True
    export_stacks: bool = False


@dataclass
class ProfileResult:
    """Container for profiling results."""
    
    # Basic info
    trace_path: str
    duration_ms: float
    
    # CUDA metrics
    cuda_time_total_ms: float = 0.0
    cuda_memory_peak_mb: float = 0.0
    
    # CPU metrics
    cpu_time_total_ms: float = 0.0
    cpu_memory_peak_mb: float = 0.0
    
    # Operation breakdown
    top_cuda_ops: List[Dict[str, Any]] = field(default_factory=list)
    top_cpu_ops: List[Dict[str, Any]] = field(default_factory=list)
    
    # Memory events
    memory_events: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "trace_path": self.trace_path,
            "duration_ms": self.duration_ms,
            "cuda_time_total_ms": self.cuda_time_total_ms,
            "cuda_memory_peak_mb": self.cuda_memory_peak_mb,
            "cpu_time_total_ms": self.cpu_time_total_ms,
            "cpu_memory_peak_mb": self.cpu_memory_peak_mb,
            "top_cuda_ops": self.top_cuda_ops,
            "top_cpu_ops": self.top_cpu_ops,
        }
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "GPU PROFILING SUMMARY",
            "=" * 60,
            f"Trace saved to: {self.trace_path}",
            f"Total duration: {self.duration_ms:.2f} ms",
            "",
            "CUDA Metrics:",
            f"  Total CUDA time: {self.cuda_time_total_ms:.2f} ms",
            f"  Peak CUDA memory: {self.cuda_memory_peak_mb:.2f} MB",
            "",
            "CPU Metrics:",
            f"  Total CPU time: {self.cpu_time_total_ms:.2f} ms",
            f"  Peak CPU memory: {self.cpu_memory_peak_mb:.2f} MB",
            "",
            "Top CUDA Operations:",
        ]
        
        for i, op in enumerate(self.top_cuda_ops[:5], 1):
            lines.append(f"  {i}. {op['name']}: {op['cuda_time_ms']:.3f} ms ({op['percentage']:.1f}%)")
        
        lines.append("")
        lines.append("Top CPU Operations:")
        for i, op in enumerate(self.top_cpu_ops[:5], 1):
            lines.append(f"  {i}. {op['name']}: {op['cpu_time_ms']:.3f} ms ({op['percentage']:.1f}%)")
        
        lines.append("=" * 60)
        return "\n".join(lines)


class TorchProfiler:
    """
    Enhanced PyTorch Profiler with automatic analysis and export.
    
    Example:
        >>> with TorchProfiler(output_dir="./traces") as profiler:
        ...     model(input)
        ...     profiler.step()
        >>> print(profiler.result.summary())
    """
    
    def __init__(
        self,
        output_dir: str = "./profiler_traces",
        trace_name: Optional[str] = None,
        config: Optional[ProfileConfig] = None,
        **kwargs
    ):
        """
        Initialize the profiler.
        
        Args:
            output_dir: Directory to save trace files
            trace_name: Optional name for the trace file
            config: Optional ProfileConfig object
            **kwargs: Override config parameters
        """
        if config is None:
            config = ProfileConfig(output_dir=output_dir, trace_name=trace_name, **kwargs)
        self.config = config
        
        # Ensure output directory exists
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate trace name if not provided
        if self.config.trace_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.config.trace_name = f"trace_{timestamp}"
        
        self._profiler = None
        self._result: Optional[ProfileResult] = None
        self._step_count = 0
    
    @property
    def result(self) -> Optional[ProfileResult]:
        """Get profiling result after context exit."""
        return self._result
    
    def _build_activities(self) -> List[ProfilerActivity]:
        """Build list of profiler activities."""
        activities = []
        if self.config.profile_cuda and torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        if self.config.profile_cpu:
            activities.append(ProfilerActivity.CPU)
        return activities
    
    def _build_schedule(self) -> Callable:
        """Build profiler schedule."""
        return torch.profiler.schedule(
            wait=self.config.wait_steps,
            warmup=self.config.warmup_steps,
            active=self.config.active_steps,
            repeat=self.config.repeat,
        )
    
    def _build_trace_handler(self) -> Callable:
        """Build trace export handler."""
        def handler(prof):
            # Export Chrome trace
            if self.config.export_chrome_trace:
                trace_path = os.path.join(
                    self.config.output_dir,
                    f"{self.config.trace_name}.json"
                )
                prof.export_chrome_trace(trace_path)
            
            # Export TensorBoard trace
            if self.config.export_tensorboard:
                tb_path = os.path.join(self.config.output_dir, "tensorboard")
                Path(tb_path).mkdir(parents=True, exist_ok=True)
                prof.export_chrome_trace(
                    os.path.join(tb_path, f"{self.config.trace_name}.json")
                )
            
            # Export stack traces
            if self.config.export_stacks:
                stack_path = os.path.join(
                    self.config.output_dir,
                    f"{self.config.trace_name}_stacks.txt"
                )
                prof.export_stacks(stack_path, "self_cuda_time_total")
        
        return handler
    
    def __enter__(self) -> "TorchProfiler":
        """Enter profiling context."""
        activities = self._build_activities()
        
        self._profiler = profile(
            activities=activities,
            schedule=self._build_schedule(),
            on_trace_ready=self._build_trace_handler(),
            record_shapes=self.config.record_shapes,
            profile_memory=self.config.profile_memory,
            with_stack=self.config.with_stack,
            with_flops=self.config.with_flops,
            with_modules=self.config.with_modules,
        )
        
        self._profiler.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit profiling context and analyze results."""
        self._profiler.__exit__(exc_type, exc_val, exc_tb)
        self._analyze_results()
        return False
    
    def step(self):
        """Signal that a profiling step is complete."""
        if self._profiler:
            self._profiler.step()
            self._step_count += 1
    
    def _analyze_results(self):
        """Analyze profiling results and create ProfileResult."""
        trace_path = os.path.join(
            self.config.output_dir,
            f"{self.config.trace_name}.json"
        )
        
        # Get key averages
        key_averages = self._profiler.key_averages()
        
        # Calculate totals
        cuda_time_total = sum(e.cuda_time_total for e in key_averages) / 1000  # μs to ms
        cpu_time_total = sum(e.cpu_time_total for e in key_averages) / 1000
        
        # Get peak memory
        cuda_memory_peak = 0.0
        if torch.cuda.is_available():
            cuda_memory_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        
        # Top CUDA operations
        cuda_ops = sorted(
            key_averages,
            key=lambda x: x.cuda_time_total,
            reverse=True
        )[:10]
        
        top_cuda_ops = []
        for op in cuda_ops:
            if op.cuda_time_total > 0:
                top_cuda_ops.append({
                    "name": op.key,
                    "cuda_time_ms": op.cuda_time_total / 1000,
                    "percentage": (op.cuda_time_total / 1000 / cuda_time_total * 100) if cuda_time_total > 0 else 0,
                    "count": op.count,
                    "input_shapes": str(op.input_shapes) if hasattr(op, 'input_shapes') else "",
                })
        
        # Top CPU operations
        cpu_ops = sorted(
            key_averages,
            key=lambda x: x.cpu_time_total,
            reverse=True
        )[:10]
        
        top_cpu_ops = []
        for op in cpu_ops:
            if op.cpu_time_total > 0:
                top_cpu_ops.append({
                    "name": op.key,
                    "cpu_time_ms": op.cpu_time_total / 1000,
                    "percentage": (op.cpu_time_total / 1000 / cpu_time_total * 100) if cpu_time_total > 0 else 0,
                    "count": op.count,
                })
        
        self._result = ProfileResult(
            trace_path=trace_path,
            duration_ms=cuda_time_total + cpu_time_total,
            cuda_time_total_ms=cuda_time_total,
            cuda_memory_peak_mb=cuda_memory_peak,
            cpu_time_total_ms=cpu_time_total,
            top_cuda_ops=top_cuda_ops,
            top_cpu_ops=top_cpu_ops,
        )
    
    def print_summary(self):
        """Print profiling summary."""
        if self._result:
            print(self._result.summary())
        else:
            print("No profiling results available. Make sure to exit the context first.")


@contextmanager
def profile_region(name: str, output_dir: str = "./traces"):
    """
    Context manager for profiling a specific code region.
    
    Example:
        >>> with profile_region("forward_pass"):
        ...     output = model(input)
    """
    profiler = TorchProfiler(
        output_dir=output_dir,
        trace_name=name,
        wait_steps=0,
        warmup_steps=0,
        active_steps=1,
    )
    
    with profiler:
        yield profiler
        profiler.step()
    
    profiler.print_summary()


def profile_function(func: Callable, *args, output_dir: str = "./traces", **kwargs) -> ProfileResult:
    """
    Profile a single function call.
    
    Example:
        >>> result = profile_function(model.forward, input_tensor)
        >>> print(result.summary())
    """
    func_name = getattr(func, '__name__', 'unknown_function')
    
    with TorchProfiler(output_dir=output_dir, trace_name=func_name) as profiler:
        result = func(*args, **kwargs)
        profiler.step()
    
    return profiler.result
