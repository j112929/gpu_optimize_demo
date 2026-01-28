"""
SGLang Integration Module - Fast LLM serving and structured generation.

This module provides SGLang-based LLM inference capabilities:
- High-performance model serving with RadixAttention
- Structured output generation
- Efficient batching and caching
- Multi-modal support
"""

from src.sglang_inference.server import (
    SGLangServer,
    ServerConfig,
    launch_server,
    stop_server,
)
from src.sglang_inference.client import (
    SGLangClient,
    generate,
    generate_batch,
    chat,
)
from src.sglang_inference.structured import (
    StructuredGenerator,
    JsonGenerator,
    ChoiceGenerator,
    RegexGenerator,
)
from src.sglang_inference.programs import (
    SGLProgram,
    multi_turn_chat,
    chain_of_thought,
    few_shot_learning,
)
from src.sglang_inference.benchmark import (
    SGLangBenchmark,
    benchmark_throughput,
    benchmark_latency,
)

__all__ = [
    # Server
    "SGLangServer",
    "ServerConfig",
    "launch_server",
    "stop_server",
    # Client
    "SGLangClient", 
    "generate",
    "generate_batch",
    "chat",
    # Structured Generation
    "StructuredGenerator",
    "JsonGenerator",
    "ChoiceGenerator",
    "RegexGenerator",
    # Programs
    "SGLProgram",
    "multi_turn_chat",
    "chain_of_thought",
    "few_shot_learning",
    # Benchmark
    "SGLangBenchmark",
    "benchmark_throughput",
    "benchmark_latency",
]
