"""
Evaluation Metrics - Performance and quality metrics.

Provides:
- Language model metrics (perplexity, accuracy)
- Latency and throughput measurement
- Generation quality metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time
import numpy as np
from collections import defaultdict


# =============================================================================
# Language Model Metrics
# =============================================================================

def compute_perplexity(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> float:
    """
    Compute perplexity of a language model on input.
    
    Lower perplexity = better model fit.
    
    Example:
        >>> ppl = compute_perplexity(model, tokenized_text)
        >>> print(f"Perplexity: {ppl:.2f}")
    """
    model.eval()
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
        loss = outputs.loss
    
    return torch.exp(loss).item()


def compute_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> float:
    """
    Compute token-level accuracy.
    
    Args:
        predictions: Predicted token IDs
        labels: Ground truth token IDs
        ignore_index: Index to ignore (e.g., padding)
    """
    mask = labels != ignore_index
    
    if mask.sum() == 0:
        return 0.0
    
    correct = (predictions == labels) & mask
    
    return correct.sum().item() / mask.sum().item()


def compute_f1(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    Compute F1 score for text generation.
    
    Uses token overlap between prediction and reference.
    """
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    
    for pred, ref in zip(predictions, references):
        pred_tokens = set(pred.lower().split())
        ref_tokens = set(ref.lower().split())
        
        if not pred_tokens or not ref_tokens:
            continue
        
        common = pred_tokens & ref_tokens
        
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        total_precision += precision
        total_recall += recall
        total_f1 += f1
    
    n = len(predictions)
    
    return {
        "precision": total_precision / n if n > 0 else 0.0,
        "recall": total_recall / n if n > 0 else 0.0,
        "f1": total_f1 / n if n > 0 else 0.0,
    }


def compute_bleu(
    predictions: List[str],
    references: List[str],
    max_n: int = 4,
) -> float:
    """
    Compute BLEU score for text generation.
    
    Simplified implementation without smoothing.
    """
    def get_ngrams(text: str, n: int) -> Dict[Tuple[str, ...], int]:
        tokens = text.lower().split()
        ngrams = defaultdict(int)
        for i in range(len(tokens) - n + 1):
            ngrams[tuple(tokens[i:i+n])] += 1
        return ngrams
    
    total_matches = [0] * max_n
    total_counts = [0] * max_n
    
    for pred, ref in zip(predictions, references):
        for n in range(1, max_n + 1):
            pred_ngrams = get_ngrams(pred, n)
            ref_ngrams = get_ngrams(ref, n)
            
            for ngram, count in pred_ngrams.items():
                total_matches[n-1] += min(count, ref_ngrams.get(ngram, 0))
                total_counts[n-1] += count
    
    # Compute geometric mean of precisions
    precisions = []
    for i in range(max_n):
        if total_counts[i] > 0:
            precisions.append(total_matches[i] / total_counts[i])
        else:
            precisions.append(0.0)
    
    if all(p > 0 for p in precisions):
        bleu = np.exp(np.mean(np.log(precisions)))
    else:
        bleu = 0.0
    
    return float(bleu)


# =============================================================================
# Latency Metrics
# =============================================================================

@dataclass
class LatencyMetrics:
    """
    Latency measurement for model inference.
    
    Example:
        >>> metrics = LatencyMetrics()
        >>> 
        >>> for input in inputs:
        ...     metrics.start()
        ...     output = model(input)
        ...     metrics.end()
        >>> 
        >>> print(metrics.summary())
    """
    
    # Collected times
    times_ms: List[float] = field(default_factory=list)
    
    # State
    _start_time: float = 0.0
    
    def start(self):
        """Start timing."""
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self._start_time = time.perf_counter()
    
    def end(self):
        """End timing and record."""
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = (time.perf_counter() - self._start_time) * 1000
        self.times_ms.append(elapsed)
    
    def reset(self):
        """Reset collected times."""
        self.times_ms.clear()
    
    @property
    def mean_ms(self) -> float:
        """Mean latency in ms."""
        return np.mean(self.times_ms) if self.times_ms else 0.0
    
    @property
    def std_ms(self) -> float:
        """Standard deviation in ms."""
        return np.std(self.times_ms) if len(self.times_ms) > 1 else 0.0
    
    @property
    def p50_ms(self) -> float:
        """50th percentile (median) latency."""
        return np.percentile(self.times_ms, 50) if self.times_ms else 0.0
    
    @property
    def p95_ms(self) -> float:
        """95th percentile latency."""
        return np.percentile(self.times_ms, 95) if self.times_ms else 0.0
    
    @property
    def p99_ms(self) -> float:
        """99th percentile latency."""
        return np.percentile(self.times_ms, 99) if self.times_ms else 0.0
    
    def summary(self) -> Dict[str, float]:
        """Get summary statistics."""
        return {
            "n_samples": len(self.times_ms),
            "mean_ms": self.mean_ms,
            "std_ms": self.std_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "min_ms": min(self.times_ms) if self.times_ms else 0.0,
            "max_ms": max(self.times_ms) if self.times_ms else 0.0,
        }
    
    def print_summary(self):
        """Print formatted summary."""
        summary = self.summary()
        print("\n" + "=" * 40)
        print("LATENCY METRICS")
        print("=" * 40)
        print(f"Samples:  {summary['n_samples']}")
        print(f"Mean:     {summary['mean_ms']:.2f} ms")
        print(f"Std:      {summary['std_ms']:.2f} ms")
        print(f"P50:      {summary['p50_ms']:.2f} ms")
        print(f"P95:      {summary['p95_ms']:.2f} ms")
        print(f"P99:      {summary['p99_ms']:.2f} ms")
        print("=" * 40)


@dataclass
class ThroughputMetrics:
    """
    Throughput measurement for model inference.
    
    Measures tokens/second, requests/second, etc.
    
    Example:
        >>> metrics = ThroughputMetrics()
        >>> 
        >>> metrics.start()
        >>> for batch in batches:
        ...     output = model(batch)
        ...     metrics.add_tokens(output.shape[1] * batch_size)
        >>> metrics.end()
        >>> 
        >>> print(metrics.summary())
    """
    
    # Counts
    total_tokens: int = 0
    total_requests: int = 0
    
    # Timing
    _start_time: float = 0.0
    _end_time: float = 0.0
    
    def start(self):
        """Start measurement."""
        self._start_time = time.perf_counter()
    
    def end(self):
        """End measurement."""
        self._end_time = time.perf_counter()
    
    def add_tokens(self, n: int):
        """Add token count."""
        self.total_tokens += n
    
    def add_request(self):
        """Add request count."""
        self.total_requests += 1
    
    def reset(self):
        """Reset counts."""
        self.total_tokens = 0
        self.total_requests = 0
        self._start_time = 0.0
        self._end_time = 0.0
    
    @property
    def elapsed_seconds(self) -> float:
        """Total elapsed time in seconds."""
        return self._end_time - self._start_time
    
    @property
    def tokens_per_second(self) -> float:
        """Tokens generated per second."""
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.total_tokens / self.elapsed_seconds
    
    @property
    def requests_per_second(self) -> float:
        """Requests processed per second."""
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.total_requests / self.elapsed_seconds
    
    def summary(self) -> Dict[str, float]:
        """Get summary statistics."""
        return {
            "total_tokens": self.total_tokens,
            "total_requests": self.total_requests,
            "elapsed_seconds": self.elapsed_seconds,
            "tokens_per_second": self.tokens_per_second,
            "requests_per_second": self.requests_per_second,
        }
    
    def print_summary(self):
        """Print formatted summary."""
        summary = self.summary()
        print("\n" + "=" * 40)
        print("THROUGHPUT METRICS")
        print("=" * 40)
        print(f"Total tokens:    {summary['total_tokens']:,}")
        print(f"Total requests:  {summary['total_requests']:,}")
        print(f"Elapsed:         {summary['elapsed_seconds']:.2f} s")
        print(f"Tokens/sec:      {summary['tokens_per_second']:.1f}")
        print(f"Requests/sec:    {summary['requests_per_second']:.1f}")
        print("=" * 40)


# =============================================================================
# Benchmark Utilities
# =============================================================================

def benchmark_model(
    model: nn.Module,
    input_ids: torch.Tensor,
    num_warmup: int = 10,
    num_iterations: int = 100,
) -> Dict[str, float]:
    """
    Benchmark model latency.
    
    Args:
        model: Model to benchmark
        input_ids: Input tensor
        num_warmup: Warmup iterations
        num_iterations: Benchmark iterations
        
    Returns:
        Dict with latency statistics
    """
    model.eval()
    
    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(input_ids)
    
    torch.cuda.synchronize()
    
    # Benchmark
    metrics = LatencyMetrics()
    
    for _ in range(num_iterations):
        metrics.start()
        with torch.no_grad():
            _ = model(input_ids)
        metrics.end()
    
    return metrics.summary()


def compare_models(
    models: Dict[str, nn.Module],
    input_ids: torch.Tensor,
    num_iterations: int = 50,
) -> Dict[str, Dict[str, float]]:
    """
    Compare latency of multiple models.
    
    Example:
        >>> results = compare_models({
        ...     "base": base_model,
        ...     "compiled": compiled_model,
        ...     "quantized": quantized_model,
        ... }, input_ids)
    """
    results = {}
    
    for name, model in models.items():
        results[name] = benchmark_model(model, input_ids, num_iterations=num_iterations)
    
    # Print comparison
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(f"{'Model':<20} {'Mean (ms)':<12} {'P95 (ms)':<12} {'Speedup':<10}")
    print("-" * 60)
    
    baseline_mean = results[list(results.keys())[0]]["mean_ms"]
    
    for name, stats in results.items():
        speedup = baseline_mean / stats["mean_ms"]
        print(f"{name:<20} {stats['mean_ms']:<12.2f} {stats['p95_ms']:<12.2f} {speedup:<10.2f}x")
    
    print("=" * 60)
    
    return results
