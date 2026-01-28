"""
SGLang Benchmark - Performance testing for LLM inference.

Provides tools for measuring throughput, latency, and memory usage.
"""

import time
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    # Throughput
    total_requests: int
    total_tokens_generated: int
    total_time_seconds: float
    requests_per_second: float
    tokens_per_second: float
    
    # Latency
    mean_latency_ms: float
    p50_latency_ms: float
    p90_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    
    # Time to first token
    mean_ttft_ms: float = 0.0
    p50_ttft_ms: float = 0.0
    p99_ttft_ms: float = 0.0
    
    def __repr__(self) -> str:
        return f"""Benchmark Results:
  Throughput:
    Requests/sec: {self.requests_per_second:.2f}
    Tokens/sec: {self.tokens_per_second:.2f}
  Latency:
    Mean: {self.mean_latency_ms:.2f}ms
    P50: {self.p50_latency_ms:.2f}ms
    P90: {self.p90_latency_ms:.2f}ms
    P99: {self.p99_latency_ms:.2f}ms
  TTFT:
    Mean: {self.mean_ttft_ms:.2f}ms
    P99: {self.p99_ttft_ms:.2f}ms"""


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking."""
    num_requests: int = 100
    max_concurrent: int = 16
    prompt_length: int = 128
    output_length: int = 128
    warmup_requests: int = 10
    
    # Sample prompts
    prompts: Optional[List[str]] = None


class SGLangBenchmark:
    """
    Benchmark SGLang server performance.
    
    Features:
    - Throughput measurement
    - Latency percentiles
    - Time to first token (TTFT)
    - Concurrent request handling
    
    Example:
        >>> bench = SGLangBenchmark("http://localhost:30000")
        >>> result = bench.run(num_requests=1000, max_concurrent=32)
        >>> print(result)
    """
    
    def __init__(self, url: str = "http://localhost:30000"):
        from src.sglang_inference.client import SGLangClient
        self.client = SGLangClient(url)
        self.url = url
    
    def _generate_prompts(self, num: int, length: int) -> List[str]:
        """Generate sample prompts."""
        base_prompts = [
            "Explain the concept of",
            "Write a detailed description of",
            "What are the key features of",
            "Describe the process of",
            "List the main benefits of",
        ]
        
        prompts = []
        for i in range(num):
            base = base_prompts[i % len(base_prompts)]
            # Pad to approximate length
            topic = f"topic {i} " * (length // 10)
            prompts.append(f"{base} {topic}")
        
        return prompts
    
    def _run_single_request(
        self,
        prompt: str,
        output_length: int,
    ) -> Dict[str, Any]:
        """Run a single request and measure timing."""
        from src.sglang_inference.client import GenerationConfig
        
        config = GenerationConfig(
            max_new_tokens=output_length,
            temperature=0.7,
        )
        
        start_time = time.perf_counter()
        result = self.client.generate(prompt, config)
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000
        
        return {
            "latency_ms": latency_ms,
            "tokens_generated": result.completion_tokens,
            "prompt_tokens": result.prompt_tokens,
        }
    
    def run(
        self,
        config: Optional[BenchmarkConfig] = None,
        num_requests: int = 100,
        max_concurrent: int = 16,
        prompt_length: int = 128,
        output_length: int = 128,
    ) -> BenchmarkResult:
        """
        Run benchmark.
        
        Args:
            config: Benchmark configuration
            num_requests: Number of requests
            max_concurrent: Maximum concurrent requests
            prompt_length: Approximate prompt length
            output_length: Maximum output length
            
        Returns:
            BenchmarkResult with metrics
        """
        if config:
            num_requests = config.num_requests
            max_concurrent = config.max_concurrent
            prompt_length = config.prompt_length
            output_length = config.output_length
            prompts = config.prompts
        else:
            prompts = None
        
        # Generate or use provided prompts
        if prompts is None:
            prompts = self._generate_prompts(num_requests, prompt_length)
        
        # Warmup
        print("Warming up...")
        for prompt in prompts[:min(10, len(prompts))]:
            self._run_single_request(prompt, output_length)
        
        # Run benchmark
        print(f"Running {num_requests} requests with {max_concurrent} concurrent...")
        
        latencies = []
        tokens_generated = []
        
        start_time = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = []
            for prompt in prompts[:num_requests]:
                future = executor.submit(
                    self._run_single_request,
                    prompt,
                    output_length,
                )
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    latencies.append(result["latency_ms"])
                    tokens_generated.append(result["tokens_generated"])
                except Exception as e:
                    print(f"Request failed: {e}")
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Calculate metrics
        latencies.sort()
        total_tokens = sum(tokens_generated)
        
        def percentile(data: List[float], p: float) -> float:
            idx = int(len(data) * p)
            return data[min(idx, len(data) - 1)]
        
        return BenchmarkResult(
            total_requests=len(latencies),
            total_tokens_generated=total_tokens,
            total_time_seconds=total_time,
            requests_per_second=len(latencies) / total_time,
            tokens_per_second=total_tokens / total_time,
            mean_latency_ms=statistics.mean(latencies),
            p50_latency_ms=percentile(latencies, 0.5),
            p90_latency_ms=percentile(latencies, 0.9),
            p99_latency_ms=percentile(latencies, 0.99),
            min_latency_ms=min(latencies),
            max_latency_ms=max(latencies),
        )


def benchmark_throughput(
    url: str = "http://localhost:30000",
    num_requests: int = 100,
    max_concurrent: int = 16,
) -> float:
    """
    Quick throughput benchmark.
    
    Returns:
        Tokens per second
    """
    bench = SGLangBenchmark(url)
    result = bench.run(num_requests=num_requests, max_concurrent=max_concurrent)
    return result.tokens_per_second


def benchmark_latency(
    url: str = "http://localhost:30000",
    num_requests: int = 50,
) -> Dict[str, float]:
    """
    Quick latency benchmark.
    
    Returns:
        Dict with latency percentiles
    """
    bench = SGLangBenchmark(url)
    result = bench.run(num_requests=num_requests, max_concurrent=1)
    
    return {
        "mean_ms": result.mean_latency_ms,
        "p50_ms": result.p50_latency_ms,
        "p90_ms": result.p90_latency_ms,
        "p99_ms": result.p99_latency_ms,
    }


# =============================================================================
# Comparative Benchmarking
# =============================================================================

def compare_backends(
    endpoints: Dict[str, str],
    num_requests: int = 50,
    max_concurrent: int = 8,
) -> Dict[str, BenchmarkResult]:
    """
    Compare multiple backends.
    
    Args:
        endpoints: Dict of {name: url}
        num_requests: Requests per backend
        max_concurrent: Concurrent requests
        
    Returns:
        Dict of {name: BenchmarkResult}
    """
    results = {}
    
    for name, url in endpoints.items():
        print(f"\nBenchmarking {name}...")
        bench = SGLangBenchmark(url)
        try:
            results[name] = bench.run(
                num_requests=num_requests,
                max_concurrent=max_concurrent,
            )
        except Exception as e:
            print(f"Failed to benchmark {name}: {e}")
    
    # Print comparison
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"{'Backend':<15} {'Tok/s':<10} {'P50 (ms)':<10} {'P99 (ms)':<10}")
    print("-" * 60)
    
    for name, result in results.items():
        print(f"{name:<15} {result.tokens_per_second:<10.2f} {result.p50_latency_ms:<10.2f} {result.p99_latency_ms:<10.2f}")
    
    return results


# =============================================================================
# Load Testing
# =============================================================================

class LoadTester:
    """
    Load tester with ramping and sustained load.
    
    Example:
        >>> tester = LoadTester("http://localhost:30000")
        >>> tester.ramp_up(start_qps=1, end_qps=100, duration=60)
    """
    
    def __init__(self, url: str):
        self.url = url
        self.results: List[Dict] = []
    
    def ramp_up(
        self,
        start_qps: float,
        end_qps: float,
        duration_seconds: float,
        step_duration: float = 5.0,
    ) -> List[Dict]:
        """
        Ramp up load from start to end QPS.
        
        Args:
            start_qps: Starting queries per second
            end_qps: Ending queries per second
            duration_seconds: Total duration
            step_duration: Duration per step
        """
        from src.sglang_inference.client import SGLangClient
        
        client = SGLangClient(self.url)
        num_steps = int(duration_seconds / step_duration)
        qps_increment = (end_qps - start_qps) / num_steps
        
        results = []
        
        for step in range(num_steps):
            current_qps = start_qps + step * qps_increment
            interval = 1.0 / current_qps if current_qps > 0 else 1.0
            
            print(f"Step {step+1}/{num_steps}: {current_qps:.1f} QPS")
            
            step_start = time.time()
            latencies = []
            
            while time.time() - step_start < step_duration:
                req_start = time.time()
                try:
                    client.generate("Test prompt", None)
                    latencies.append((time.time() - req_start) * 1000)
                except:
                    pass
                
                elapsed = time.time() - req_start
                if elapsed < interval:
                    time.sleep(interval - elapsed)
            
            results.append({
                "qps": current_qps,
                "actual_qps": len(latencies) / step_duration,
                "mean_latency": statistics.mean(latencies) if latencies else 0,
                "p99_latency": sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0,
            })
        
        self.results = results
        return results
    
    def find_max_qps(self, target_latency_ms: float = 500) -> float:
        """Find maximum QPS under latency target."""
        for result in reversed(self.results):
            if result["p99_latency"] <= target_latency_ms:
                return result["qps"]
        return 0
