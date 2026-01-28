#!/usr/bin/env python3
"""
Example: SGLang LLM Inference

Demonstrates SGLang for high-performance LLM serving and structured generation.

Usage:
    # Start server first:
    python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf
    
    # Then run examples:
    python examples/sglang_inference.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def demo_server_config():
    """Demonstrate server configuration."""
    from src.sglang_inference.server import ServerConfig
    
    print("\n" + "=" * 60)
    print("SGLANG SERVER CONFIGURATION")
    print("=" * 60)
    
    # Basic config
    config = ServerConfig(
        model_path="meta-llama/Llama-2-7b-chat-hf",
        port=30000,
        tp_size=1,
    )
    
    print("\n📊 Basic Configuration:")
    print(f"   Model: {config.model_path}")
    print(f"   Port: {config.port}")
    print(f"   Tensor Parallelism: {config.tp_size}")
    
    # High-performance config
    hp_config = ServerConfig(
        model_path="meta-llama/Llama-2-70b-chat-hf",
        tp_size=4,  # 4x GPU tensor parallelism
        dp_size=2,  # 2x data parallelism
        max_total_tokens=65536,
        max_running_requests=512,
        enable_flashinfer=True,
        chunked_prefill_size=16384,
    )
    
    print("\n📊 High-Performance Configuration:")
    print(f"   Model: {hp_config.model_path}")
    print(f"   TP + DP: {hp_config.tp_size} x {hp_config.dp_size} = {hp_config.tp_size * hp_config.dp_size} GPUs")
    print(f"   Max tokens: {hp_config.max_total_tokens}")
    print(f"   Max concurrent: {hp_config.max_running_requests}")
    
    # Quantized config
    quant_config = ServerConfig(
        model_path="TheBloke/Llama-2-7B-Chat-AWQ",
        quantization="awq",
        tp_size=1,
    )
    
    print("\n📊 Quantized Configuration (AWQ):")
    print(f"   Model: {quant_config.model_path}")
    print(f"   Quantization: {quant_config.quantization}")
    print("   Memory reduction: ~4x")


def demo_client():
    """Demonstrate client usage."""
    from src.sglang_inference.client import SGLangClient, GenerationConfig
    
    print("\n" + "=" * 60)
    print("SGLANG CLIENT DEMO")
    print("=" * 60)
    
    print("\n📊 Client API:")
    print("""
    # Initialize client
    client = SGLangClient("http://localhost:30000")
    
    # Simple generation
    result = client.generate("What is AI?")
    print(result.text)
    
    # With config
    config = GenerationConfig(
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
    )
    result = client.generate("Explain quantum computing", config)
    
    # Chat completion
    messages = [
        {"role": "user", "content": "Hello!"},
    ]
    result = client.chat(messages)
    print(result.text)
    
    # Batch generation
    prompts = ["Q1?", "Q2?", "Q3?"]
    results = client.generate_batch(prompts)
    
    # Streaming
    for chunk in client.generate("Tell a story", stream=True):
        print(chunk, end="", flush=True)
    """)


def demo_structured():
    """Demonstrate structured generation."""
    from src.sglang_inference.structured import (
        JsonGenerator, ChoiceGenerator, RegexGenerator,
        FunctionCallGenerator, FunctionDef,
    )
    
    print("\n" + "=" * 60)
    print("STRUCTURED GENERATION DEMO")
    print("=" * 60)
    
    print("\n📊 JSON Generation:")
    print("""
    # Define schema
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "skills": {"type": "array", "items": {"type": "string"}}
        }
    }
    
    generator = JsonGenerator(schema)
    result = generator.generate(client, "Generate a developer profile")
    # {"name": "Alice", "age": 28, "skills": ["Python", "ML"]}
    """)
    
    print("\n📊 Choice Generation:")
    print("""
    generator = ChoiceGenerator(["positive", "negative", "neutral"])
    sentiment = generator.generate(client, "Classify: I love this product!")
    # "positive"
    """)
    
    print("\n📊 Regex Generation:")
    print("""
    generator = RegexGenerator(r"\\d{3}-\\d{3}-\\d{4}")
    phone = generator.generate(client, "Generate a US phone number")
    # "555-123-4567"
    """)
    
    print("\n📊 Function Calling:")
    print("""
    functions = [
        FunctionDef(
            name="get_weather",
            description="Get weather for a location",
            parameters={"location": {"type": "string"}}
        ),
        FunctionDef(
            name="search_web",
            description="Search the web",
            parameters={"query": {"type": "string"}}
        ),
    ]
    
    generator = FunctionCallGenerator(functions)
    call = generator.generate(client, "What's the weather in Tokyo?")
    # {"name": "get_weather", "arguments": {"location": "Tokyo"}}
    """)


def demo_programs():
    """Demonstrate composable programs."""
    from src.sglang_inference.programs import (
        SGLProgram, chain_of_thought, few_shot_learning,
        self_consistency, tree_of_thought,
    )
    
    print("\n" + "=" * 60)
    print("COMPOSABLE PROGRAMS DEMO")
    print("=" * 60)
    
    print("\n📊 SGLProgram (Fluent API):")
    print("""
    program = SGLProgram(client)
    result = (program
        .system("You are a helpful coding assistant")
        .user("What is a decorator in Python?")
        .generate(max_tokens=256)
        .user("Give me an example")
        .generate(max_tokens=512, store_as="example")
        .get_variable("example"))
    """)
    
    print("\n📊 Chain-of-Thought:")
    print("""
    result = chain_of_thought(
        client,
        "If a train travels 120km in 2 hours, what is its speed?"
    )
    print(result["reasoning"])
    # "The formula for speed is distance/time..."
    print(result["answer"])
    # "60 km/h"
    """)
    
    print("\n📊 Few-Shot Learning:")
    print("""
    examples = [
        {"input": "hello world", "output": "HELLO WORLD"},
        {"input": "python", "output": "PYTHON"},
    ]
    uppercase = few_shot_learning(client, examples)
    result = uppercase("artificial intelligence")
    # "ARTIFICIAL INTELLIGENCE"
    """)
    
    print("\n📊 Self-Consistency (Voting):")
    print("""
    result = self_consistency(
        client,
        "What is 15% of 80?",
        num_samples=5
    )
    print(result["answer"])      # "12"
    print(result["confidence"])  # 0.8 (4/5 agreed)
    """)
    
    print("\n📊 Tree-of-Thought:")
    print("""
    result = tree_of_thought(
        client,
        "Design a microservices architecture for an e-commerce platform",
        num_branches=3,
        depth=2
    )
    print(result["best_path"])  # List of reasoning steps
    """)


def demo_benchmark():
    """Demonstrate benchmarking."""
    from src.sglang_inference.benchmark import (
        SGLangBenchmark, BenchmarkConfig,
        benchmark_throughput, benchmark_latency,
    )
    
    print("\n" + "=" * 60)
    print("BENCHMARKING DEMO")
    print("=" * 60)
    
    print("\n📊 Quick Throughput Benchmark:")
    print("""
    tokens_per_sec = benchmark_throughput(
        url="http://localhost:30000",
        num_requests=100,
        max_concurrent=32
    )
    print(f"Throughput: {tokens_per_sec:.2f} tok/s")
    """)
    
    print("\n📊 Latency Benchmark:")
    print("""
    latency = benchmark_latency(url="http://localhost:30000")
    print(f"P50: {latency['p50_ms']:.2f}ms")
    print(f"P99: {latency['p99_ms']:.2f}ms")
    """)
    
    print("\n📊 Full Benchmark:")
    print("""
    bench = SGLangBenchmark("http://localhost:30000")
    result = bench.run(
        num_requests=1000,
        max_concurrent=64,
        prompt_length=256,
        output_length=128,
    )
    print(result)
    # Benchmark Results:
    #   Throughput:
    #     Requests/sec: 85.23
    #     Tokens/sec: 10,892.45
    #   Latency:
    #     Mean: 145.32ms
    #     P50: 128.45ms
    #     P90: 198.67ms
    #     P99: 312.89ms
    """)
    
    print("\n📊 Compare Backends:")
    print("""
    from src.sglang_inference.benchmark import compare_backends
    
    results = compare_backends({
        "sglang": "http://localhost:30000",
        "vllm": "http://localhost:30001",
        "tgi": "http://localhost:30002",
    })
    """)


def main():
    print("=" * 60)
    print("SGLANG LLM INFERENCE DEMO")
    print("=" * 60)
    
    print("\n🚀 SGLang Features:")
    print("   - RadixAttention for prefix caching")
    print("   - Continuous batching")
    print("   - FlashInfer backend")
    print("   - Tensor parallelism")
    print("   - Structured generation")
    print("   - Composable programs")
    
    demo_server_config()
    demo_client()
    demo_structured()
    demo_programs()
    demo_benchmark()
    
    print("\n" + "=" * 60)
    print("QUICK START")
    print("=" * 60)
    print("""
# 1. Install SGLang
pip install sglang[all]

# 2. Start server
python -m sglang.launch_server \\
    --model-path meta-llama/Llama-2-7b-chat-hf \\
    --port 30000

# 3. Use client
from src.sglang_inference import generate, chat

response = generate("What is AI?")
print(response)

result = chat([{"role": "user", "content": "Hello!"}])
print(result)
    """)


if __name__ == "__main__":
    main()
