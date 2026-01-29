# 快速入门指南

> 5 分钟掌握 GPU Optimization Toolkit 核心功能

---

## 🎯 你要解决什么问题？

| 问题 | 解决方案 | 跳转 |
|:-----|:---------|:-----|
| 训练太慢 | torch.compile, AMP | [训练加速](#训练加速) |
| 单卡放不下模型 | FSDP, DeepSpeed | [分布式训练](#分布式训练) |
| 微调成本太高 | LoRA, QLoRA | [高效微调](#高效微调) |
| 推理延迟高 | Speculative Decoding | [推理优化](#推理优化) |
| 显存不够 | Quantization, Offload | [内存优化](#内存优化) |

---

## ⚡ 训练加速

### 1. torch.compile (一行代码，30-200% 加速)

```python
from src.compile import compile_model

# 就这一行！
model = compile_model(model, mode="max-autotune")

# 然后正常训练
for batch in dataloader:
    loss = model(batch)
    loss.backward()
```

### 2. 混合精度 (2× 速度, 50% 内存)

```python
from src.training import AMPTrainer

trainer = AMPTrainer()

for batch in dataloader:
    with trainer.autocast():  # 自动混合精度
        loss = model(batch)
    trainer.backward(loss)
    trainer.step(optimizer)
```

---

## 🌐 分布式训练

### 模型放得下单卡 → DDP

```python
from src.training import DDPWrapper, DDPConfig, setup_distributed

setup_distributed()

wrapper = DDPWrapper(DDPConfig(mixed_precision=True))
model = wrapper.wrap(model)

for batch in dataloader:
    with wrapper.autocast():
        loss = model(batch)
    wrapper.backward(loss, optimizer, model)
```

**运行**:
```bash
torchrun --nproc_per_node=4 train.py
```

### 模型放不下单卡 → FSDP

```python
from src.training import FSDPWrapper, FSDPConfig

config = FSDPConfig(
    sharding_strategy="full_shard",  # 内存减少 3×
    activation_checkpointing=True,   # 再减少 40%
)
model = FSDPWrapper(config).wrap(model)
```

📖 详细对比: [DDP vs FSDP](./DDPvsFSDP.md)

---

## 🎯 高效微调

### LoRA (0.1% 参数可训练)

```python
from src.post_training import apply_lora, LoRAConfig

config = LoRAConfig(
    r=16,
    target_modules=["q_proj", "v_proj"],
)
model = apply_lora(model, config)

# 正常训练，只更新 LoRA 参数
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
```

### QLoRA (4-bit 量化 + LoRA)

```python
from src.post_training import QuantizedLoRA

# 70B 模型在 24GB 单卡上微调！
model = QuantizedLoRA(base_model, bits=4, lora_config=config)
```

---

## ⚡ 推理优化

### Speculative Decoding (2-3× 加速)

```python
from src.inference import SpeculativeDecoder

# 小模型草拟，大模型验证
decoder = SpeculativeDecoder(llama_70b, llama_7b)
output = decoder.generate(input_ids)
```

### 4-bit 量化 (4× 内存减少)

```python
from src.inference import quantize_model, QuantizationConfig

config = QuantizationConfig(bits=4, method="gptq")
model = quantize_model(model, config, calibration_data)
```

---

## 💾 内存优化

### 梯度检查点 (50% 内存节省)

```python
from src.training import gradient_checkpoint_model

model = gradient_checkpoint_model(model, checkpoint_ratio=0.5)
```

### CPU Offload

```python
from src.memory import CPUOffloader

offloader = CPUOffloader(model, optimizer)
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    offloader.step()  # 自动 CPU ↔ GPU
```

---

## 🚀 一键运行示例

```bash
# 分布式训练
torchrun --nproc_per_node=4 examples/ddp_fsdp_training.py --strategy fsdp

# 训练优化
python examples/training_optimization.py

# LoRA 微调
python examples/lora_finetuning.py

# 推理优化
python examples/speculative_decoding.py
```

---

## 📊 性能速查

| 优化 | 速度提升 | 内存节省 | 复杂度 |
|:-----|:--------:|:--------:|:------:|
| torch.compile | 1.5-2× | - | ⭐ |
| AMP (BF16) | 2× | 50% | ⭐ |
| DDP | N× (N卡) | - | ⭐⭐ |
| FSDP | N× | 3× | ⭐⭐⭐ |
| LoRA | - | 99% params | ⭐⭐ |
| Speculative | 2-3× | - | ⭐⭐ |
| GPTQ 4-bit | - | 4× | ⭐⭐ |

---

<div align="center">

**下一步**: 查看 [API 文档](./API.md) | [DDP vs FSDP 详解](./DDPvsFSDP.md)

</div>
