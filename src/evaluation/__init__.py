"""
Evaluation Module - Benchmarking and safety evaluation.

Provides:
- Standard LLM benchmarks (MMLU, HellaSwag, etc.)
- Safety evaluation
- Performance metrics
"""

from src.evaluation.benchmarks import (
    Benchmark,
    MMLUBenchmark,
    HellaSwagBenchmark,
    run_benchmark,
)
from src.evaluation.safety import (
    SafetyEvaluator,
    ToxicityDetector,
    BiasAnalyzer,
    SafetyConfig,
)
from src.evaluation.metrics import (
    compute_perplexity,
    compute_accuracy,
    compute_f1,
    LatencyMetrics,
    ThroughputMetrics,
)

__all__ = [
    # Benchmarks
    "Benchmark",
    "MMLUBenchmark",
    "HellaSwagBenchmark",
    "run_benchmark",
    # Safety
    "SafetyEvaluator",
    "ToxicityDetector",
    "BiasAnalyzer",
    "SafetyConfig",
    # Metrics
    "compute_perplexity",
    "compute_accuracy",
    "compute_f1",
    "LatencyMetrics",
    "ThroughputMetrics",
]
