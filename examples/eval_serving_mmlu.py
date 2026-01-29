"""
Evaluation on Serving Endpoint Demo.

Demonstrates:
1. Starting an OpenAI-compatible LLM Server (Continuous Batching)
2. Running MMLU Benchmark against the remote API
3. High-throughput async evaluation

Usage:
    python examples/eval_serving_mmlu.py
"""

import multiprocessing
import time
import sys
import os
import requests
import argparse

# Add src path
sys.path.append(os.getcwd())

from src.serving.server import create_server
from src.evaluation.benchmarks import MMLUBenchmark, BenchmarkConfig

def start_server_process(port=8001):
    """Run server in separate process."""
    # We use a different port to avoid conflict
    create_server("mock-model", port=port)

class RemoteModelClient:
    """Wraps OpenAI API call as a callable function."""
    
    def __init__(self, port=8001):
        self.url = f"http://localhost:{port}/v1/chat/completions"
        
    def __call__(self, prompt: str) -> str:
        """Synchronous call (for benchmark compatibility)."""
        payload = {
            "model": "mock-model",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 5, # We only need A/B/C/D
            "temperature": 0.0,
            "stream": False
        }
        
        try:
            resp = requests.post(self.url, json=payload, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            else:
                return f"Error: {resp.status_code}"
        except Exception as e:
            return f"Error: {e}"

def main():
    print("="*60)
    print("🚀 Serving & Evaluation Integration Demo")
    print("="*60)
    
    port = 8004
    
    # 1. Start Server
    print(f"Starting Local LLM Server on port {port}...")
    server_process = multiprocessing.Process(
        target=start_server_process, 
        args=(port,)
    )
    server_process.start()
    
    # Wait for server to come up
    print("Waiting for server up...")
    time.sleep(5) 
    
    try:
        # 2. Setup Client
        client = RemoteModelClient(port=port)
        
        # Test connection
        print("Testing connection...")
        try:
            res = client("Hello")
            print(f"Server response: {res}")
            print("✅ Server is ready.")
        except:
            print("❌ Server failed to respond.")
            return

        # 3. Run MMLU
        print("\nrunning MMLU Benchmark via API...")
        benchmark = MMLUBenchmark(
            config=BenchmarkConfig(
                num_samples=10, # Keep it small for demo
                few_shot=0
            )
        )
        
        # Pass client as "model"
        # Tokenizer is ignored/mocked in evaluate_sample wrapper
        result = benchmark.run(client, tokenizer=None)
        
        # 4. Report
        print("\n" + "="*40)
        print("EVALUATION RESULTS")
        print("="*40)
        print(f"Benchmark:   {result.benchmark_name}")
        print(f"Accuracy:    {result.accuracy:.2%}")
        print(f"Samples:     {result.num_samples}")
        print(f"Time:        {result.total_time_s:.2f}s")
        print(f"Throughput:  {result.num_samples / result.total_time_s:.1f} req/s")
        print("-" * 40)
        
    finally:
        # Cleanup
        print("\nShutting down server...")
        server_process.terminate()
        server_process.join()

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
