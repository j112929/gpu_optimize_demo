"""
Inference Optimization Module - High-performance LLM inference.

Provides:
- Speculative Decoding for 2-3x faster generation
- Continuous Batching for optimal throughput
- KV-Cache optimization
- Quantization for memory efficiency
"""

from src.inference.speculative import (
    SpeculativeDecoder,
    SpeculativeConfig,
    speculative_generate,
)
from src.inference.batching import (
    ContinuousBatcher,
    RequestQueue,
    BatchConfig,
)
from src.inference.kv_cache import (
    KVCache,
    PagedKVCache,
    KVCacheConfig,
)
from src.inference.quantize import (
    quantize_model,
    QuantizationConfig,
    AWQQuantizer,
    GPTQQuantizer,
)

__all__ = [
    # Speculative
    "SpeculativeDecoder",
    "SpeculativeConfig",
    "speculative_generate",
    # Batching
    "ContinuousBatcher",
    "RequestQueue",
    "BatchConfig",
    # KV Cache
    "KVCache",
    "PagedKVCache",
    "KVCacheConfig",
    # Quantization
    "quantize_model",
    "QuantizationConfig",
    "AWQQuantizer",
    "GPTQQuantizer",
]
