"""
Compile Optimization Module - PyTorch 2.0 compilation and CUDA optimizations.

Provides:
- torch.compile wrappers with optimal configurations
- CUDA Graphs for low-latency inference
- TensorRT integration for production deployment
"""

from src.compile.torch_compile import (
    compile_model,
    CompileConfig,
    CompileMode,
    benchmark_compile,
)
from src.compile.cuda_graphs import (
    CUDAGraphWrapper,
    cuda_graph_inference,
    warmup_cuda_graph,
)
from src.compile.tensorrt import (
    TensorRTConverter,
    convert_to_tensorrt,
    TensorRTConfig,
)

__all__ = [
    # torch.compile
    "compile_model",
    "CompileConfig",
    "CompileMode",
    "benchmark_compile",
    # CUDA Graphs
    "CUDAGraphWrapper",
    "cuda_graph_inference",
    "warmup_cuda_graph",
    # TensorRT
    "TensorRTConverter",
    "convert_to_tensorrt",
    "TensorRTConfig",
]
