"""
Triton Kernels Module - High-performance GPU kernels using Triton.

This module provides optimized GPU kernels written in Triton, including:
- Fused operations (LayerNorm, GELU, etc.)
- Custom attention implementations
- Memory-efficient operations
- Quantization (INT8/INT4)
- LoRA fused kernels
- KV-Cache optimization
- Kernel benchmarking utilities
"""

from src.triton_kernels.fused_ops import (
    fused_add_layernorm,
    fused_gelu,
    fused_dropout_add,
    fused_softmax,
    fused_silu,
    fused_rmsnorm,
)
from src.triton_kernels.attention import (
    flash_attention_v2,
    multi_head_attention,
    grouped_query_attention,
)
from src.triton_kernels.fp8_attention import (
    flash_decode_fp8_triton,
    FlashInferWrapper,
)
from src.triton_kernels.matmul import (
    triton_matmul,
    triton_matmul_splitk,
)
from src.triton_kernels.quantization import (
    quantize_int8,
    dequantize_int8,
    quantize_int4,
    dequantize_int4,
    int8_matmul,
    QuantizedLinear,
)
from src.triton_kernels.lora import (
    lora_fused_linear,
    lora_qkv_projection,
    LoRALinear,
    QLoRALinear,
)
from src.triton_kernels.kv_cache import (
    KVCache,
    KVCacheConfig,
    PagedKVCache,
    SlidingWindowKVCache,
    kv_cache_attention,
)
from src.triton_kernels.benchmark import (
    TritonBenchmark,
    compare_with_pytorch,
    kernel_autotuner,
)

__all__ = [
    # Fused operations
    "fused_add_layernorm",
    "fused_gelu",
    "fused_dropout_add",
    "fused_softmax",
    "fused_silu",
    "fused_rmsnorm",
    # Attention
    "flash_attention_v2", # Keeping existing for now, as the instruction was ambiguous about removal
    "multi_head_attention",
    "grouped_query_attention",
    "flash_decode_fp8_triton",
    "FlashInferWrapper",
    # MatMul
    "triton_matmul",
    "triton_matmul_splitk",
    # Quantization
    "quantize_int8",
    "dequantize_int8",
    "quantize_int4",
    "dequantize_int4",
    "int8_matmul",
    "QuantizedLinear",
    # LoRA
    "lora_fused_linear",
    "lora_qkv_projection",
    "LoRALinear",
    "QLoRALinear",
    # KV-Cache
    "KVCache",
    "KVCacheConfig",
    "PagedKVCache",
    "SlidingWindowKVCache",
    "kv_cache_attention",
    # Benchmarking
    "TritonBenchmark",
    "compare_with_pytorch",
    "kernel_autotuner",
]
