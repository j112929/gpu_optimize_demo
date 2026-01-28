"""
torch.compile - PyTorch 2.0 Compilation Utilities.

Provides wrappers for torch.compile with optimal configurations
for different use cases (training, inference, memory-efficient).
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union
from enum import Enum
import time
import functools


class CompileMode(Enum):
    """Compilation modes for different optimization goals."""
    DEFAULT = "default"           # Balanced
    REDUCE_OVERHEAD = "reduce-overhead"  # Fastest for small batches
    MAX_AUTOTUNE = "max-autotune"        # Best performance, slower compile
    MAX_AUTOTUNE_NO_CUDAGRAPHS = "max-autotune-no-cudagraphs"


@dataclass
class CompileConfig:
    """Configuration for torch.compile."""
    # Mode
    mode: CompileMode = CompileMode.DEFAULT
    
    # Backend
    backend: str = "inductor"  # inductor, cudagraphs, onnxrt, tensorrt
    
    # Options
    fullgraph: bool = False  # Require full graph capture
    dynamic: bool = False    # Enable dynamic shapes
    disable: bool = False    # Disable compilation
    
    # Advanced
    options: Optional[Dict[str, Any]] = None


def compile_model(
    model: nn.Module,
    config: Optional[CompileConfig] = None,
    mode: Optional[str] = None,
    backend: str = "inductor",
    **kwargs,
) -> nn.Module:
    """
    Compile a PyTorch model for optimized execution.
    
    Args:
        model: PyTorch model to compile
        config: CompileConfig object
        mode: Shortcut for config.mode
        backend: Compilation backend
        **kwargs: Additional options passed to torch.compile
        
    Returns:
        Compiled model
        
    Example:
        >>> model = compile_model(model, mode="reduce-overhead")
        >>> # 30-200% faster inference
        >>> output = model(input)
    """
    if config is None:
        config = CompileConfig()
    
    if mode is not None:
        config.mode = CompileMode(mode) if isinstance(mode, str) else mode
    
    if config.disable:
        return model
    
    compile_options = config.options or {}
    compile_options.update(kwargs)
    
    compiled = torch.compile(
        model,
        mode=config.mode.value if isinstance(config.mode, CompileMode) else config.mode,
        backend=backend,
        fullgraph=config.fullgraph,
        dynamic=config.dynamic,
        options=compile_options if compile_options else None,
    )
    
    return compiled


def compile_function(
    fn: Optional[Callable] = None,
    *,
    mode: str = "default",
    backend: str = "inductor",
    **kwargs,
) -> Callable:
    """
    Decorator to compile a function with torch.compile.
    
    Example:
        >>> @compile_function(mode="reduce-overhead")
        ... def my_kernel(x, y):
        ...     return x @ y + torch.relu(x)
    """
    def decorator(func: Callable) -> Callable:
        return torch.compile(func, mode=mode, backend=backend, **kwargs)
    
    if fn is not None:
        return decorator(fn)
    return decorator


# =============================================================================
# Preset Configurations
# =============================================================================

def compile_for_training(model: nn.Module) -> nn.Module:
    """
    Compile model optimized for training.
    
    Uses default mode which balances compile time and performance,
    with dynamic shapes enabled for variable batch sizes.
    """
    return compile_model(
        model,
        mode="default",
        dynamic=True,
    )


def compile_for_inference(model: nn.Module) -> nn.Module:
    """
    Compile model optimized for inference.
    
    Uses reduce-overhead mode for minimal latency,
    best for small batch sizes and real-time applications.
    """
    model.eval()
    return compile_model(
        model,
        mode="reduce-overhead",
        fullgraph=True,
    )


def compile_for_throughput(model: nn.Module) -> nn.Module:
    """
    Compile model for maximum throughput.
    
    Uses max-autotune for best performance at cost of
    longer compilation time. Best for batch processing.
    """
    return compile_model(
        model,
        mode="max-autotune",
    )


# =============================================================================
# Benchmarking
# =============================================================================

def benchmark_compile(
    model: nn.Module,
    example_input: torch.Tensor,
    modes: Optional[List[str]] = None,
    warmup: int = 10,
    iterations: int = 100,
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark different compilation modes.
    
    Args:
        model: Model to benchmark
        example_input: Example input tensor
        modes: List of modes to test
        warmup: Warmup iterations
        iterations: Benchmark iterations
        
    Returns:
        Dict with timing results per mode
        
    Example:
        >>> results = benchmark_compile(model, torch.randn(1, 3, 224, 224))
        >>> for mode, times in results.items():
        ...     print(f"{mode}: {times['mean_ms']:.2f}ms")
    """
    if modes is None:
        modes = ["eager", "default", "reduce-overhead", "max-autotune"]
    
    results = {}
    
    for mode in modes:
        print(f"Benchmarking {mode}...")
        
        if mode == "eager":
            test_model = model
        else:
            test_model = compile_model(model, mode=mode)
        
        # Move to GPU
        device = example_input.device
        test_model = test_model.to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = test_model(example_input)
        
        torch.cuda.synchronize()
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(iterations):
                start = time.perf_counter()
                _ = test_model(example_input)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)
        
        results[mode] = {
            "mean_ms": sum(times) / len(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "std_ms": (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
        }
    
    # Print comparison
    print("\n" + "=" * 50)
    print("COMPILATION BENCHMARK RESULTS")
    print("=" * 50)
    print(f"{'Mode':<20} {'Mean (ms)':<12} {'Speedup':<10}")
    print("-" * 50)
    
    eager_time = results.get("eager", {}).get("mean_ms", 1)
    for mode, data in results.items():
        speedup = eager_time / data["mean_ms"] if data["mean_ms"] > 0 else 1
        print(f"{mode:<20} {data['mean_ms']:<12.2f} {speedup:<10.2f}x")
    
    return results


# =============================================================================
# Dynamic Shape Handling
# =============================================================================

class DynamicShapeCompiler:
    """
    Handles compilation with dynamic input shapes.
    
    Caches compiled versions for different input shapes
    to avoid recompilation overhead.
    """
    
    def __init__(self, model: nn.Module, mode: str = "default"):
        self.model = model
        self.mode = mode
        self._compiled_cache: Dict[tuple, nn.Module] = {}
    
    def __call__(self, *args, **kwargs) -> Any:
        # Get input shapes
        shapes = tuple(
            tuple(arg.shape) if isinstance(arg, torch.Tensor) else None
            for arg in args
        )
        
        # Get or compile
        if shapes not in self._compiled_cache:
            self._compiled_cache[shapes] = compile_model(
                self.model,
                mode=self.mode,
                fullgraph=True,
            )
        
        return self._compiled_cache[shapes](*args, **kwargs)
    
    def clear_cache(self):
        """Clear compiled model cache."""
        self._compiled_cache.clear()


# =============================================================================
# Selective Compilation
# =============================================================================

def compile_selective(
    model: nn.Module,
    compile_modules: Optional[List[str]] = None,
    skip_modules: Optional[List[str]] = None,
    mode: str = "default",
) -> nn.Module:
    """
    Selectively compile parts of a model.
    
    Args:
        model: Model to compile
        compile_modules: List of module names to compile (if None, compile all)
        skip_modules: List of module names to skip
        mode: Compilation mode
        
    Returns:
        Partially compiled model
    """
    skip_modules = skip_modules or []
    
    for name, module in model.named_children():
        should_compile = True
        
        if compile_modules is not None and name not in compile_modules:
            should_compile = False
        
        if name in skip_modules:
            should_compile = False
        
        if should_compile and not isinstance(module, (nn.ModuleList, nn.ModuleDict)):
            compiled = torch.compile(module, mode=mode)
            setattr(model, name, compiled)
        else:
            # Recursively handle nested modules
            compile_selective(module, compile_modules, skip_modules, mode)
    
    return model
