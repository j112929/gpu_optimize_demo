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
    ChunkedPrefillScheduler,
)
from src.inference.kv_cache import (
    PagedKVCache,
    PrefixCacheManager,
    FP8KVCache,
)
from src.inference.quantize import (
    quantize_model,
    QuantizationConfig,
    AWQQuantizer,
    GPTQQuantizer,
)
from src.inference.tp_worker import (
    TPInferenceWorker,
    ParallelInferenceEngine,
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
    "ChunkedPrefillScheduler",
    # KV Cache
    "PagedKVCache",
    "PrefixCacheManager",
    "FP8KVCache",
    # Quantization
    "quantize_model",
    "QuantizationConfig",
    "AWQQuantizer",
    "GPTQQuantizer",
    # Distributed
    "TPInferenceWorker",
    "ParallelInferenceEngine",
]
