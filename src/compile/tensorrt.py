"""
TensorRT Integration - Production-grade inference optimization.

Provides TensorRT compilation for 2-5x inference speedup
with FP16/INT8 precision support.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import warnings


@dataclass
class TensorRTConfig:
    """Configuration for TensorRT conversion."""
    # Precision
    precision: str = "fp16"  # fp32, fp16, int8
    
    # Workspace
    workspace_size: int = 1 << 30  # 1GB
    
    # Optimization
    min_block_size: int = 5
    pass_through_build_failures: bool = False
    
    # Dynamic shapes
    dynamic_batch: bool = False
    min_batch_size: int = 1
    opt_batch_size: int = 8
    max_batch_size: int = 32
    
    # Calibration (for INT8)
    calibration_data: Optional[torch.Tensor] = None
    calibration_batches: int = 100
    
    # Caching
    cache_path: Optional[str] = None


class TensorRTConverter:
    """
    Convert PyTorch models to TensorRT for optimized inference.
    
    Supports:
    - FP32, FP16, INT8 precision
    - Dynamic batch sizes
    - Engine caching
    
    Example:
        >>> converter = TensorRTConverter(config)
        >>> trt_model = converter.convert(model, example_input)
        >>> output = trt_model(input)  # 2-5x faster
    """
    
    def __init__(self, config: Optional[TensorRTConfig] = None):
        self.config = config or TensorRTConfig()
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if TensorRT is available."""
        try:
            import torch_tensorrt
            self.torch_tensorrt = torch_tensorrt
        except ImportError:
            warnings.warn(
                "torch_tensorrt not installed. Install with: "
                "pip install torch-tensorrt"
            )
            self.torch_tensorrt = None
    
    def convert(
        self,
        model: nn.Module,
        example_inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    ) -> nn.Module:
        """
        Convert model to TensorRT.
        
        Args:
            model: PyTorch model
            example_inputs: Example input(s) for tracing
            
        Returns:
            TensorRT-optimized model
        """
        if self.torch_tensorrt is None:
            raise RuntimeError("torch_tensorrt not available")
        
        model = model.cuda().eval()
        
        if isinstance(example_inputs, torch.Tensor):
            example_inputs = (example_inputs.cuda(),)
        else:
            example_inputs = tuple(x.cuda() for x in example_inputs)
        
        # Build input specs
        if self.config.dynamic_batch:
            inputs = [
                self.torch_tensorrt.Input(
                    min_shape=(self.config.min_batch_size,) + tuple(x.shape[1:]),
                    opt_shape=(self.config.opt_batch_size,) + tuple(x.shape[1:]),
                    max_shape=(self.config.max_batch_size,) + tuple(x.shape[1:]),
                    dtype=x.dtype,
                )
                for x in example_inputs
            ]
        else:
            inputs = list(example_inputs)
        
        # Enabled precisions
        enabled_precisions = {torch.float32}
        if self.config.precision in ("fp16", "int8"):
            enabled_precisions.add(torch.float16)
        if self.config.precision == "int8":
            enabled_precisions.add(torch.int8)
        
        # Compile
        trt_model = self.torch_tensorrt.compile(
            model,
            inputs=inputs,
            enabled_precisions=enabled_precisions,
            workspace_size=self.config.workspace_size,
            min_block_size=self.config.min_block_size,
            pass_through_build_failures=self.config.pass_through_build_failures,
        )
        
        # Cache engine if path provided
        if self.config.cache_path:
            self.save_engine(trt_model, self.config.cache_path)
        
        return trt_model
    
    def save_engine(self, model: nn.Module, path: str):
        """Save TensorRT engine to disk."""
        if self.torch_tensorrt is None:
            return
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.jit.save(
            torch.jit.trace(model, torch.randn(1, 3, 224, 224).cuda()),
            str(path)
        )
    
    def load_engine(self, path: str) -> nn.Module:
        """Load TensorRT engine from disk."""
        return torch.jit.load(str(path))


def convert_to_tensorrt(
    model: nn.Module,
    example_input: torch.Tensor,
    precision: str = "fp16",
    dynamic_batch: bool = False,
) -> nn.Module:
    """
    Quick function to convert a model to TensorRT.
    
    Args:
        model: PyTorch model
        example_input: Example input tensor
        precision: "fp32", "fp16", or "int8"
        dynamic_batch: Enable dynamic batch sizes
        
    Returns:
        TensorRT-optimized model
        
    Example:
        >>> trt_model = convert_to_tensorrt(model, torch.randn(1, 3, 224, 224))
        >>> output = trt_model(input)  # Much faster!
    """
    config = TensorRTConfig(
        precision=precision,
        dynamic_batch=dynamic_batch,
    )
    
    converter = TensorRTConverter(config)
    return converter.convert(model, example_input)


# =============================================================================
# ONNX Export + TensorRT
# =============================================================================

class ONNXTensorRTConverter:
    """
    Convert models via ONNX for maximum compatibility.
    
    Path: PyTorch -> ONNX -> TensorRT
    """
    
    def __init__(self, config: Optional[TensorRTConfig] = None):
        self.config = config or TensorRTConfig()
    
    def export_onnx(
        self,
        model: nn.Module,
        example_input: torch.Tensor,
        output_path: str,
        opset_version: int = 17,
    ) -> str:
        """Export model to ONNX format."""
        model = model.cpu().eval()
        example_input = example_input.cpu()
        
        torch.onnx.export(
            model,
            example_input,
            output_path,
            opset_version=opset_version,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            } if self.config.dynamic_batch else None,
        )
        
        return output_path
    
    def onnx_to_trt(
        self,
        onnx_path: str,
        output_path: str,
    ) -> str:
        """
        Convert ONNX model to TensorRT engine.
        
        Requires tensorrt package installed.
        """
        try:
            import tensorrt as trt
        except ImportError:
            raise ImportError("tensorrt package required. Install with pip install tensorrt")
        
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, logger)
        
        # Parse ONNX
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                raise RuntimeError("ONNX parsing failed")
        
        # Build config
        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            self.config.workspace_size
        )
        
        if self.config.precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
        elif self.config.precision == "int8":
            config.set_flag(trt.BuilderFlag.INT8)
        
        # Build engine
        engine = builder.build_serialized_network(network, config)
        
        # Save
        with open(output_path, "wb") as f:
            f.write(engine)
        
        return output_path


# =============================================================================
# Benchmark
# =============================================================================

def benchmark_tensorrt(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    precisions: List[str] = ["fp32", "fp16"],
    iterations: int = 100,
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark TensorRT at different precisions.
    
    Returns:
        Dict with timing results per precision
    """
    import time
    
    model = model.cuda().eval()
    x = torch.randn(*input_shape, device="cuda")
    
    results = {}
    
    # Eager baseline
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(iterations):
            _ = model(x)
        torch.cuda.synchronize()
        eager_time = (time.perf_counter() - start) / iterations * 1000
    
    results["eager"] = {"mean_ms": eager_time, "speedup": 1.0}
    
    # TensorRT at each precision
    for precision in precisions:
        try:
            trt_model = convert_to_tensorrt(model, x, precision=precision)
            
            with torch.no_grad():
                for _ in range(10):
                    _ = trt_model(x)
                torch.cuda.synchronize()
                
                start = time.perf_counter()
                for _ in range(iterations):
                    _ = trt_model(x)
                torch.cuda.synchronize()
                trt_time = (time.perf_counter() - start) / iterations * 1000
            
            results[f"trt_{precision}"] = {
                "mean_ms": trt_time,
                "speedup": eager_time / trt_time,
            }
        except Exception as e:
            results[f"trt_{precision}"] = {"error": str(e)}
    
    # Print results
    print("\n" + "=" * 50)
    print("TENSORRT BENCHMARK RESULTS")
    print("=" * 50)
    for mode, data in results.items():
        if "error" in data:
            print(f"{mode:<15} ERROR: {data['error']}")
        else:
            print(f"{mode:<15} {data['mean_ms']:.2f}ms ({data['speedup']:.2f}x)")
    
    return results
