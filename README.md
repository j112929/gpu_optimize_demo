# 🚀 GPU Optimization Toolkit

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA 12.0+](https://img.shields.io/badge/cuda-12.0+-76b900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Production-ready toolkit covering the complete ML lifecycle: Pre-Training → Post-Training → Inference → Evaluation**

---

## 🎯 Why This Toolkit?

Modern ML engineering requires optimization at every stage. This toolkit provides:

| Stage | Challenge | Our Solution |
|-------|-----------|--------------|
| **Pre-Training** | GPU underutilization, slow I/O, OOM | torch.compile, AMP, FSDP, Memory Pool |
| **Post-Training** | Full fine-tuning too expensive | LoRA/QLoRA, PEFT, DPO/RLHF |
| **Inference** | High latency, low throughput | Speculative Decoding, KV-Cache, Quantization |
| **Evaluation** | No standardized benchmarks | MMLU, Safety checks, Latency metrics |

---

## 📊 Performance Gains

| Optimization | Improvement | Use Case |
|:------------|:-----------:|:---------|
| `torch.compile` | **1.5-2×** faster | Any PyTorch model |
| Mixed Precision (AMP) | **2×** speed, **50%** memory | Training |
| LoRA/QLoRA | **99.9%** fewer params | Fine-tuning 70B models on 24GB GPU |
| Speculative Decoding | **2-3×** faster generation | LLM inference |
| GPTQ/AWQ (4-bit) | **4×** memory reduction | Production serving |
| Flash Attention | **5×** faster, **O(N)** memory | Long sequences |
| Continuous Batching | **3×** throughput | High-traffic serving |

---

## 🏗️ Architecture — Complete ML Lifecycle

```
╔════════════════════════════════════════════════════════════════════════════════════╗
║                              GPU OPTIMIZATION TOOLKIT                               ║
╠════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                     ║
║  ┌───────────────────────────────────────────────────────────────────────────────┐ ║
║  │  📚 PRE-TRAINING                                                              │ ║
║  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐          │ ║
║  │  │   Compile    │ │   Training   │ │    Memory    │ │ Distributed  │          │ ║
║  │  │ torch.compile│ │ AMP (FP16/   │ │ Pool, Offload│ │ FSDP,        │          │ ║
║  │  │ CUDA Graphs  │ │ BF16), Grad  │ │ Checkpoint,  │ │ DeepSpeed,   │          │ ║
║  │  │ TensorRT     │ │ Accum/Clip   │ │ Profiling    │ │ Ray Train    │          │ ║
║  │  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘          │ ║
║  └───────────────────────────────────────────────────────────────────────────────┘ ║
║                                         ↓                                           ║
║  ┌───────────────────────────────────────────────────────────────────────────────┐ ║
║  │  🎯 POST-TRAINING                                                             │ ║
║  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                           │ ║
║  │  │    LoRA      │ │     PEFT     │ │  Alignment   │  Fine-tune 70B on 24GB!  │ ║
║  │  │ LoRA, QLoRA, │ │ Adapter,     │ │ RLHF (PPO),  │  Only 0.1% params        │ ║
║  │  │ DoRA, RS-LoRA│ │ Prefix/Prompt│ │ DPO, KTO     │  trainable              │ ║
║  │  └──────────────┘ └──────────────┘ └──────────────┘                           │ ║
║  └───────────────────────────────────────────────────────────────────────────────┘ ║
║                                         ↓                                           ║
║  ┌───────────────────────────────────────────────────────────────────────────────┐ ║
║  │  ⚡ INFERENCE                                                                  │ ║
║  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐          │ ║
║  │  │ Speculative  │ │   Batching   │ │   KV-Cache   │ │ Quantization │          │ ║
║  │  │ Draft model, │ │ Continuous,  │ │ Paged (vLLM),│ │ GPTQ, AWQ,   │          │ ║
║  │  │ Medusa heads │ │ Dynamic, FIFO│ │ Sliding, Pfx │ │ INT4/INT8    │          │ ║
║  │  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘          │ ║
║  └───────────────────────────────────────────────────────────────────────────────┘ ║
║                                         ↓                                           ║
║  ┌───────────────────────────────────────────────────────────────────────────────┐ ║
║  │  📏 SERVING & EVALUATION                                                      │ ║
║  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐          │ ║
║  │  │   SGLang     │ │  Benchmarks  │ │    Safety    │ │   Metrics    │          │ ║
║  │  │ Server, CoT, │ │ MMLU, Hella- │ │ Toxicity,    │ │ Latency, P99,│          │ ║
║  │  │ JSON, Stream │ │ Swag, Eval   │ │ Bias, Filter │ │ Throughput   │          │ ║
║  │  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘          │ ║
║  └───────────────────────────────────────────────────────────────────────────────┘ ║
║                                                                                     ║
╚════════════════════════════════════════════════════════════════════════════════════╝
```

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/your-org/gpu_optimize_demo.git
cd gpu_optimize_demo
pip install -e ".[all]"
```

### 1-Minute Demo

```python
# ═══════════════════════════════════════════════════════════
# PRE-TRAINING: Make training 2x faster
# ═══════════════════════════════════════════════════════════

# 🔥 torch.compile - One line, 30-200% speedup
from src.compile import compile_model
model = compile_model(model, mode="reduce-overhead")

# 📉 Mixed Precision - 2x speed, 50% memory
from src.training import AMPTrainer
trainer = AMPTrainer()
with trainer.autocast():
    loss = model(x)
trainer.backward(loss)

# ═══════════════════════════════════════════════════════════
# POST-TRAINING: Fine-tune 70B on single GPU
# ═══════════════════════════════════════════════════════════

# 🎯 LoRA - Only 0.1% parameters trainable
from src.post_training import apply_lora, LoRAConfig
config = LoRAConfig(r=16, target_modules=["q_proj", "v_proj"])
model = apply_lora(model, config)
print(f"Trainable: {model.get_trainable_params():,}")  # Only 4M of 70B!

# 🤝 DPO Alignment - Simpler than RLHF
from src.post_training import DPOTrainer
trainer = DPOTrainer(model, ref_model)
trainer.step(prompt, chosen_response, rejected_response)

# ═══════════════════════════════════════════════════════════
# INFERENCE: 3x faster generation
# ═══════════════════════════════════════════════════════════

# ⚡ Speculative Decoding - 2-3x speedup
from src.inference import SpeculativeDecoder
decoder = SpeculativeDecoder(llama_70b, llama_7b)  # Draft with 7B
output = decoder.generate(input_ids)

# 📦 4-bit Quantization - 4x memory reduction
from src.inference import quantize_model, QuantizationConfig
model = quantize_model(model, QuantizationConfig(bits=4, method="gptq"))

# ═══════════════════════════════════════════════════════════
# EVALUATION: Benchmark and safety
# ═══════════════════════════════════════════════════════════

# 📊 Run MMLU benchmark
from src.evaluation import run_benchmark
result = run_benchmark(model, tokenizer, "mmlu")
print(f"MMLU: {result.accuracy:.2%}")

# 🛡️ Safety evaluation
from src.evaluation import SafetyEvaluator
evaluator = SafetyEvaluator()
report = evaluator.evaluate_model(model, tokenizer, test_prompts)
```

---

## 📦 Module Reference

### 1️⃣ Pre-Training Optimization

<details>
<summary><b>🔧 Compile Optimization</b> — torch.compile, CUDA Graphs, TensorRT</summary>

```python
from src.compile import (
    compile_model,           # torch.compile wrapper
    CUDAGraphWrapper,        # CUDA Graphs for low latency
    TensorRTConverter,       # TensorRT for production
    benchmark_compile,       # Compare modes
)

# torch.compile with best mode
model = compile_model(model, mode="max-autotune")  # Best throughput

# CUDA Graphs - 10-30% latency reduction
wrapper = CUDAGraphWrapper(model, example_input)
output = wrapper(input)  # Graph replay

# TensorRT - Production deployment
converter = TensorRTConverter(model)
trt_model = converter.convert(precision="fp16")
```
</details>

<details>
<summary><b>📉 Training Optimization</b> — AMP, Gradient Utilities, Optimizers</summary>

```python
from src.training import (
    AMPTrainer, AMPConfig,           # Mixed precision
    GradientAccumulator,             # Gradient accumulation
    gradient_checkpoint_model,       # Memory savings
    create_optimizer,                # Fused optimizers
)

# Mixed Precision Training
config = AMPConfig(dtype="bfloat16", enabled=True)
trainer = AMPTrainer(config)

for batch in dataloader:
    with trainer.autocast():
        loss = model(batch)
    trainer.backward(loss)
    trainer.step(optimizer)

# Gradient Accumulation (simulate larger batches)
accumulator = GradientAccumulator(accumulation_steps=4)
for batch in dataloader:
    loss = model(batch)
    if accumulator.step(loss, optimizer):
        optimizer.step()

# Gradient Checkpointing (50-70% memory savings)
model = gradient_checkpoint_model(model, checkpoint_ratio=0.5)
```
</details>

<details>
<summary><b>💾 Memory Optimization</b> — Pool, Offloading, Profiling</summary>

```python
from src.memory import (
    MemoryPool,              # Tensor reuse
    CPUOffloader,            # Optimizer state offload
    MemoryProfiler,          # Track usage
    print_memory_summary,    # Quick stats
)

# Memory Pool for inference
pool = MemoryPool()
tensor = pool.allocate((1024, 768), dtype=torch.float16)
# ... use tensor ...
pool.release(tensor)  # Reuse later

# CPU Offload (2x model size)
offloader = CPUOffloader(model, optimizer)
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    offloader.step()  # Moves states CPU↔GPU

# Memory Profiling
profiler = MemoryProfiler()
profiler.snapshot("before_forward")
output = model(x)
profiler.snapshot("after_forward")
print(profiler.delta("before_forward", "after_forward"))
```
</details>

<details>
<summary><b>🌐 Distributed Training</b> — FSDP, DeepSpeed, Ray</summary>

```python
from src.training import FSDPWrapper, FSDPConfig, DeepSpeedWrapper
from src.ray_distributed import RayTrainer, RayTrainerConfig

# FSDP - Shard across GPUs
config = FSDPConfig(
    sharding_strategy="full_shard",
    mixed_precision=True,
    activation_checkpointing=True,
)
model = FSDPWrapper(config).wrap(model)

# DeepSpeed ZeRO-3
ds_model = DeepSpeedWrapper(model, stage=3)

# Ray Train - Multi-node, fault-tolerant
trainer = RayTrainer(RayTrainerConfig(
    num_workers=8,
    use_gpu=True,
    checkpoint_interval=100,
))
trainer.fit(training_function, dataset)
```
</details>

---

### 2️⃣ Post-Training Optimization

<details>
<summary><b>🎯 LoRA / QLoRA</b> — 10,000x fewer trainable parameters</summary>

```python
from src.post_training import (
    LoRAModel, LoRAConfig,
    QuantizedLoRA,                   # 4-bit QLoRA
    apply_lora, merge_lora,         # Convenience functions
)

# Standard LoRA
config = LoRAConfig(
    r=16,                            # Rank
    alpha=32,                        # Scaling
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    dropout=0.05,
)
model = LoRAModel(base_model, config)
model.print_trainable_params()  # "Trainable: 4,194,304 (0.01%)"

# Train normally - only LoRA params updated
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
for batch in dataloader:
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()

# Merge for inference (no overhead)
merged_model = merge_lora(model)
```
</details>

<details>
<summary><b>🧩 PEFT Methods</b> — Adapter, Prefix, Prompt Tuning</summary>

```python
from src.post_training import (
    AdapterModel, AdapterConfig,     # Adapter layers
    PrefixTuning, PrefixConfig,      # Prefix tuning
    PromptTuning, PromptConfig,      # Soft prompts
)

# Adapter Layers (~1-5% extra params)
config = AdapterConfig(bottleneck_dim=64)
model = AdapterModel(base_model, config)

# Prefix Tuning (learnable KV prefixes)
prefix = PrefixTuning(PrefixConfig(
    prefix_length=20,
    num_layers=32,
))

# Prompt Tuning (prepend learnable tokens)
prompt = PromptTuning(PromptConfig(num_virtual_tokens=20))
input_embeds = prompt(model.embed(input_ids))
```
</details>

<details>
<summary><b>🤝 Alignment</b> — RLHF, DPO, KTO</summary>

```python
from src.post_training import (
    RewardModel,              # Train reward model
    RLHFTrainer, PPOConfig,   # PPO-based RLHF
    DPOTrainer, DPOConfig,    # Direct Preference Optimization
    KTOTrainer,               # Non-paired preferences
)

# DPO - Simpler than RLHF, no reward model needed
trainer = DPOTrainer(
    model=policy_model,
    ref_model=reference_model,
    config=DPOConfig(beta=0.1),
)

for batch in preference_data:
    stats = trainer.step(
        prompt_ids=batch["prompt"],
        chosen_ids=batch["chosen"],
        rejected_ids=batch["rejected"],
    )
    print(f"Loss: {stats['loss']:.4f}, Acc: {stats['accuracy']:.2%}")

# RLHF with PPO
reward_model = RewardModel(base_model)
rlhf = RLHFTrainer(policy_model, ref_model, reward_model)
```
</details>

---

### 3️⃣ Inference Optimization

<details>
<summary><b>⚡ Speculative Decoding</b> — 2-3x faster generation</summary>

```python
from src.inference import (
    SpeculativeDecoder, SpeculativeConfig,
    speculative_generate,
    benchmark_speculative,
)

# Use small model to draft, large model to verify
decoder = SpeculativeDecoder(
    target_model=llama_70b,
    draft_model=llama_7b,
    config=SpeculativeConfig(num_speculative_tokens=5),
)

output = decoder.generate(input_ids, max_new_tokens=256)
decoder.print_stats()
# Accept rate: 78%
# Speedup: 2.3x

# Quick benchmark
results = benchmark_speculative(llama_70b, llama_7b, input_ids)
print(f"Speedup: {results['speedup']:.2f}x")
```
</details>

<details>
<summary><b>📦 Continuous Batching</b> — Dynamic request handling</summary>

```python
from src.inference import (
    ContinuousBatcher, BatchConfig,
    Request, RequestQueue,
)

# Setup batcher
batcher = ContinuousBatcher(
    model,
    config=BatchConfig(max_batch_size=32),
    tokenizer=tokenizer,
)
batcher.start()

# Submit requests (non-blocking)
request = Request(prompt="What is AI?", max_new_tokens=100)
batcher.submit(request)

# Get result
result = batcher.wait(request.id)

# Stats
print(batcher.get_stats())
# {'tokens_per_second': 1250, 'requests_per_second': 15}
```
</details>

<details>
<summary><b>🗄️ KV-Cache</b> — Paged, Sliding Window, Prefix</summary>

```python
from src.inference import (
    KVCache, KVCacheConfig,
    PagedKVCache,            # vLLM-style paging
    SlidingWindowCache,      # For long sequences
    PrefixCache,             # Share common prefixes
)

# Standard KV-Cache
cache = KVCache(KVCacheConfig(
    num_layers=32,
    num_heads=32,
    max_sequence_length=4096,
))

# Paged KV-Cache (reduce fragmentation)
paged_cache = PagedKVCache(config)
block_ids = paged_cache.allocate(request_id, num_tokens=100)

# Sliding Window (constant memory for any length)
sliding = SlidingWindowCache(window_size=4096)
```
</details>

<details>
<summary><b>📉 Quantization</b> — GPTQ, AWQ, INT4/INT8</summary>

```python
from src.inference import (
    quantize_model, QuantizationConfig,
    GPTQQuantizer, AWQQuantizer,
    estimate_quantization_savings,
)

# Quick quantization
config = QuantizationConfig(
    bits=4,                  # 4-bit
    method="gptq",           # or "awq"
    group_size=128,
)
quantized = quantize_model(model, config, calibration_data)

# Estimate savings
savings = estimate_quantization_savings(model, bits=4)
print(f"Current: {savings['current_mb']:.0f} MB")
print(f"After:   {savings['quantized_mb']:.0f} MB")
print(f"Savings: {savings['compression_ratio']}x")
```
</details>

---

### 4️⃣ Serving & Evaluation

<details>
<summary><b>🤖 SGLang Inference</b> — High-performance LLM serving</summary>

```python
from src.sglang_inference import (
    SGLangServer, ServerConfig,
    generate, chat, generate_batch,
    JsonGenerator, chain_of_thought,
)

# Start server
server = SGLangServer(ServerConfig(
    model_path="meta-llama/Llama-2-70b-chat-hf",
    tp_size=4,               # Tensor parallel
    port=30000,
)).start()

# Generate
response = generate("Explain quantum computing", max_tokens=512)

# Structured JSON output
json_gen = JsonGenerator(schema={"type": "object", ...})
data = json_gen.generate("Create a user profile")

# Chain-of-thought
result = chain_of_thought(client, "What is 15% of 80?")
print(result["reasoning"])  # Step-by-step
print(result["answer"])     # 12
```
</details>

<details>
<summary><b>📊 Benchmarks</b> — MMLU, HellaSwag, HumanEval</summary>

```python
from src.evaluation import (
    run_benchmark, run_all_benchmarks,
    MMLUBenchmark, HellaSwagBenchmark,
)

# Run single benchmark
result = run_benchmark(model, tokenizer, "mmlu")
print(f"MMLU Accuracy: {result.accuracy:.2%}")

# Run all benchmarks
results = run_all_benchmarks(model, tokenizer, 
    benchmarks=["mmlu", "hellaswag", "truthfulqa"]
)

for name, result in results.items():
    print(f"{name}: {result.accuracy:.2%}")
```
</details>

<details>
<summary><b>🛡️ Safety Evaluation</b> — Toxicity, Bias, Red-teaming</summary>

```python
from src.evaluation import (
    SafetyEvaluator, SafetyConfig,
    ToxicityDetector, BiasAnalyzer,
    ContentFilter,
)

# Full safety evaluation
evaluator = SafetyEvaluator()
results = evaluator.evaluate_model(model, tokenizer, test_prompts)
evaluator.print_report(results)

# Toxicity check
detector = ToxicityDetector()
result = detector.check("some text")
if not result.is_safe:
    print(f"Flags: {result.flags}")

# Content filter (for production)
filter = ContentFilter()
is_safe, text = filter.filter_output(response)
```
</details>

<details>
<summary><b>📈 Metrics</b> — Latency, Throughput, Perplexity</summary>

```python
from src.evaluation import (
    LatencyMetrics, ThroughputMetrics,
    compute_perplexity, benchmark_model,
    compare_models,
)

# Latency tracking
latency = LatencyMetrics()
for batch in dataloader:
    latency.start()
    output = model(batch)
    latency.end()

latency.print_summary()
# Mean: 12.5ms, P95: 15.2ms, P99: 18.1ms

# Compare models
compare_models({
    "baseline": base_model,
    "compiled": compiled_model,
    "quantized": quantized_model,
}, input_ids)
```
</details>

---

## 📁 Project Structure

```
gpu_optimize_demo/
├── src/
│   ├── compile/             # ⚙️ Compilation (torch.compile, CUDA Graphs, TensorRT)
│   ├── training/            # 📉 Training (AMP, Gradients, Distributed, Optimizers)
│   ├── memory/              # � Memory (Pool, Offload, Profiling)
│   ├── post_training/       # 🎯 Post-Training (LoRA, PEFT, RLHF, DPO)
│   ├── inference/           # ⚡ Inference (Speculative, Batching, KV-Cache, Quant)
│   ├── evaluation/          # 📏 Evaluation (Benchmarks, Safety, Metrics)
│   ├── sglang_inference/    # 🤖 SGLang (Server, Client, Structured Gen)
│   ├── ray_distributed/     # 🚀 Ray (Train, Tune, Serve)
│   ├── profiling/           # 🔬 Profiling (Torch Profiler, CUDA Timer)
│   ├── io_optimize/         # ⚡ I/O (DataLoader, Prefetcher, MMap)
│   ├── nccl/                # 🌐 NCCL (Comm Profiler, Overlap)
│   └── triton_kernels/      # 🔥 Triton (Flash Attention, Fused Ops)
│
├── examples/                # 📚 Ready-to-run examples
│   ├── training_optimization.py
│   ├── lora_finetuning.py
│   ├── inference_optimization.py
│   └── ...
│
└── benchmarks/              # 📊 Benchmark suites
```

---

## 🧪 Examples

```bash
# Pre-Training
python examples/training_optimization.py    # AMP, compile, checkpointing

# Post-Training  
python examples/lora_finetuning.py          # LoRA/QLoRA fine-tuning
python examples/dpo_alignment.py            # DPO preference learning

# Inference
python examples/speculative_decoding.py     # 2-3x faster generation
python examples/quantization.py             # GPTQ/AWQ 4-bit

# Evaluation
python examples/run_benchmarks.py           # MMLU, HellaSwag
python examples/safety_evaluation.py        # Toxicity, bias checks
```

---

## 📋 Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | ≥2.0.0 | Core framework |
| Triton | ≥2.1.0 | Custom GPU kernels |
| CUDA | ≥12.0 | GPU acceleration |
| Ray | ≥2.9.0 | Distributed computing |
| SGLang | ≥0.2.0 | LLM inference |

<details>
<summary>📦 Full requirements.txt</summary>

```
torch>=2.0.0
triton>=2.1.0
ray[all]>=2.9.0
sglang[all]>=0.2.0
numpy>=1.24.0
transformers>=4.35.0
accelerate>=0.25.0
bitsandbytes>=0.41.0
tensorboard>=2.14.0
```
</details>

---

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md).

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">

**[📖 Docs](docs/)** · **[💻 Examples](examples/)** · **[📊 Benchmarks](benchmarks/)** · **[🐛 Issues](https://github.com/your-org/gpu_optimize_demo/issues)**

---

_Built for ML engineers who want production-ready optimizations_ 🚀

</div>
