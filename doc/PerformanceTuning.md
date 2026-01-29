# 性能调优指南

> 如何榨干 GPU 的每一滴性能

---

## 📊 性能优化清单

### ✅ 必做优化 (低成本高收益)

| 优化 | 收益 | 成本 | 优先级 |
|:-----|:----:|:----:|:------:|
| 启用 BF16/FP16 混合精度 | 2× 速度 | 1 行代码 | ⭐⭐⭐⭐⭐ |
| torch.compile | 1.5-2× 速度 | 1 行代码 | ⭐⭐⭐⭐⭐ |
| 使用 DistributedSampler | 正确的多卡训练 | 5 行代码 | ⭐⭐⭐⭐⭐ |
| 设置 num_workers > 0 | I/O 不阻塞 | 1 行代码 | ⭐⭐⭐⭐ |
| pin_memory=True | 更快的数据传输 | 1 行代码 | ⭐⭐⭐⭐ |

### ⚡ 进阶优化

| 优化 | 收益 | 成本 | 何时用 |
|:-----|:----:|:----:|:-------|
| 梯度累积 | 更大有效 batch | 中等 | GPU 显存不够 |
| 梯度检查点 | 50% 内存 | 25% 速度 | OOM |
| FSDP | 3× 内存 | 10-20% 速度 | 大模型 |
| CPU Offload | 更大模型 | 很慢 | 极限情况 |

---

## 🔥 混合精度 (AMP)

### BF16 vs FP16

| 特性 | BF16 | FP16 |
|:-----|:-----|:-----|
| 动态范围 | 大 (同 FP32) | 小 |
| 精度 | 低 | 高 |
| 稳定性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| 需要 Loss Scaling | ❌ | ✅ |
| 硬件要求 | A100, H100, 4090 | 所有 |

**推荐**: 优先使用 **BF16**

```python
from src.training import AMPTrainer, AMPConfig

# BF16 (推荐)
trainer = AMPTrainer(AMPConfig(dtype="bfloat16"))

# FP16 (老显卡)
trainer = AMPTrainer(AMPConfig(dtype="float16"))
```

---

## ⚙️ torch.compile 调优

### 模式选择

```python
from src.compile import compile_model

# 默认模式 - 平衡
model = compile_model(model, mode="default")

# 低开销模式 - 快速编译
model = compile_model(model, mode="reduce-overhead")

# 最大自动调优 - 最快运行 (编译慢)
model = compile_model(model, mode="max-autotune")
```

### 常见问题

| 问题 | 解决方案 |
|:-----|:---------|
| 编译失败 | 检查动态 shape，设置 `dynamic=True` |
| 重新编译 | 固定 batch size，避免动态 input |
| 与 DDP 不兼容 | 先 compile 再 DDP |
| 与 FSDP 不兼容 | 设置 `use_orig_params=True` |

---

## 📦 数据加载优化

### 最佳实践

```python
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# ✅ 优化配置
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,                    # CPU 核数 / GPU 数
    pin_memory=True,                  # 固定内存，加速传输
    prefetch_factor=2,                # 预取 batch 数
    persistent_workers=True,          # 保持 worker 存活
    sampler=DistributedSampler(dataset),  # 多卡必需
)
```

### num_workers 选择

| 场景 | 推荐值 |
|:-----|:-------|
| 简单数据 | 2-4 |
| 复杂预处理 | 8-16 |
| 大图像 | 4-8 |
| NVMe SSD | 4-8 |
| HDD | 2-4 |

---

## 💾 内存优化

### 1. 梯度累积

```python
from src.training import DDPConfig, DDPWrapper

config = DDPConfig(
    gradient_accumulation_steps=4,  # 有效 batch = 4 × batch_size
)
```

**效果**: 减少显存占用，模拟更大 batch

### 2. 梯度检查点

```python
from src.training import gradient_checkpoint_model

model = gradient_checkpoint_model(model, checkpoint_ratio=0.5)
```

**效果**: 省 50-70% 内存，牺牲 20-30% 速度

### 3. FSDP 分片

```python
from src.training import FSDPConfig

config = FSDPConfig(
    sharding_strategy="full_shard",     # 完全分片
    activation_checkpointing=True,      # 配合使用
)
```

**效果**: 内存减少 N 倍 (N = GPU 数)

---

## 🌐 分布式训练调优

### DDP 优化

```python
DDPConfig(
    static_graph=True,              # 固定图优化
    gradient_as_bucket_view=True,   # 减少内存复制
    bucket_cap_mb=25.0,             # 调整 bucket 大小
)
```

### FSDP 优化

```python
FSDPConfig(
    backward_prefetch="backward_pre",   # 预取下一层参数
    forward_prefetch=True,              # 前向预取
    limit_all_gathers=True,             # 限制并发 AllGather
    use_orig_params=True,               # 启用更多优化
)
```

### 通信优化

```python
# 环境变量
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0              # 启用 InfiniBand
export NCCL_SOCKET_IFNAME=eth0        # 指定网卡
```

---

## 📈 Benchmark 对比

### 测试环境

- 模型: LLaMA-7B
- 硬件: 8 × A100 80GB
- 数据: 1M samples

### 结果

| 配置 | 吞吐量 | 内存/卡 | 备注 |
|:-----|:------:|:-------:|:-----|
| Baseline | 3,000 tok/s | 72 GB | FP32, 单卡 |
| + AMP BF16 | 6,000 tok/s | 38 GB | 2× 加速 |
| + torch.compile | 9,000 tok/s | 38 GB | 3× 加速 |
| + DDP (8卡) | 68,000 tok/s | 38 GB | 线性扩展 |
| + FSDP | 58,000 tok/s | 12 GB | 内存友好 |

---

## 🔍 性能分析工具

### PyTorch Profiler

```python
from src.profiling import profile_model

with profile_model(model, "my_profile") as profiler:
    for batch in dataloader:
        loss = model(batch)
        loss.backward()

profiler.print_summary()
# 输出: CUDA 时间、内存峰值、热点函数
```

### 内存分析

```python
from src.memory import MemoryProfiler

profiler = MemoryProfiler()
profiler.snapshot("start")

output = model(x)

profiler.snapshot("after_forward")
loss.backward()

profiler.snapshot("after_backward")
print(profiler.summary())
```

---

## 🎯 场景优化建议

### 预训练 (大规模)

```python
# 推荐配置
FSDPConfig(
    sharding_strategy="full_shard",
    mixed_precision=True,
    precision="bf16",
    activation_checkpointing=True,
)
+ torch.compile(mode="max-autotune")
```

### 微调 (单卡/少量卡)

```python
# 推荐配置
DDPConfig(
    mixed_precision=True,
    precision="bf16",
    gradient_accumulation_steps=4,
    static_graph=True,
)
+ LoRA(r=16)
```

### 推理 (低延迟)

```python
# 推荐配置
model = torch.compile(model, mode="reduce-overhead")
model = quantize_model(model, bits=4, method="gptq")
+ CUDAGraphWrapper
+ SpeculativeDecoder
```

---

## ⚠️ 常见性能陷阱

| 陷阱 | 症状 | 解决方案 |
|:-----|:-----|:---------|
| 数据加载慢 | GPU 利用率低 | 增加 num_workers |
| 频繁 GC | 训练卡顿 | 减少临时对象 |
| 通信瓶颈 | 多卡扩展差 | 检查 NCCL 配置 |
| 动态 shape | 频繁重编译 | 固定 batch size |
| 未对齐 tensor | 效率低 | 使用 2 的幂次大小 |

---

## 📚 更多资源

- [DDP vs FSDP 对比](./DDPvsFSDP.md)
- [PyTorch 性能指南](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [NVIDIA 深度学习性能](https://developer.nvidia.com/deep-learning-performance-training-inference)

---

<div align="center">

**核心原则**: 先测量，再优化 📊

</div>
