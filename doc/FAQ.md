# 常见问题 (FAQ)

> GPU Optimization Toolkit 使用过程中的常见问题和解答

---

## 🌐 分布式训练

### Q: DDP 和 FSDP 有什么区别？

**简短回答**: DDP 复制完整模型到每张卡，FSDP 将模型分片到多卡。

| 特性 | DDP | FSDP |
|:-----|:----|:-----|
| 模型大小限制 | 必须放入单卡 | 可超过单卡显存 |
| 内存效率 | 低 | 高 (3× 节省) |
| 通信开销 | 低 | 较高 |
| 配置复杂度 | 简单 | 中等 |

📖 详细对比: [DDP vs FSDP](./DDPvsFSDP.md)

---

### Q: 何时使用 DDP？何时使用 FSDP？

```
模型能放进单张 GPU 显存？
├── ✅ 能 → 使用 DDP (更快更简单)
└── ❌ 不能 → 使用 FSDP (分片存储)

具体判断:
- 7B 模型 + 80GB A100 → DDP
- 7B 模型 + 24GB 4090 → FSDP
- 70B 模型 → 必须 FSDP
```

---

### Q: torchrun 和 python -m torch.distributed.launch 有什么区别？

`torchrun` 是新版推荐的启动方式：

```bash
# ✅ 推荐 (PyTorch 1.10+)
torchrun --nproc_per_node=4 train.py

# ❌ 已废弃
python -m torch.distributed.launch --nproc_per_node=4 train.py
```

---

### Q: 多机多卡怎么运行？

```bash
# 机器 1 (主节点)
torchrun --nnodes=2 --node_rank=0 --master_addr=192.168.1.1 --master_port=29500 \
    --nproc_per_node=8 train.py

# 机器 2
torchrun --nnodes=2 --node_rank=1 --master_addr=192.168.1.1 --master_port=29500 \
    --nproc_per_node=8 train.py
```

---

### Q: 为什么多卡训练后每个卡的输出都一样？

**原因**: 忘记使用 `DistributedSampler`

**解决**:
```python
from torch.utils.data.distributed import DistributedSampler

sampler = DistributedSampler(dataset)
dataloader = DataLoader(dataset, sampler=sampler)

# 每个 epoch 开始时
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)  # 重要！
    for batch in dataloader:
        ...
```

---

### Q: FSDP 保存的 checkpoint 为什么加载失败？

**原因**: FSDP 的 state_dict 格式特殊

**解决**: 使用专用函数
```python
from src.training import save_fsdp_checkpoint, load_fsdp_checkpoint

# 保存
save_fsdp_checkpoint(model, optimizer, epoch, "checkpoint.pt")

# 加载
load_fsdp_checkpoint(model, optimizer, "checkpoint.pt")
```

---

## 💾 内存问题

### Q: 训练时 OOM 怎么办？

按优先级尝试：

1. **减小 batch size**
2. **启用梯度累积**
   ```python
   DDPConfig(gradient_accumulation_steps=4)
   ```
3. **启用梯度检查点**
   ```python
   model = gradient_checkpoint_model(model)
   ```
4. **使用 FSDP**
   ```python
   FSDPConfig(sharding_strategy="full_shard")
   ```
5. **启用 CPU offload**
   ```python
   FSDPConfig(cpu_offload=True)
   ```

---

### Q: 如何估算模型训练需要多少显存？

```python
from src.training import estimate_memory_usage

estimates = estimate_memory_usage(
    model,
    batch_size=8,
    seq_length=512,
    precision="bf16",
)
print(f"预计需要: {estimates['total_gb']:.1f} GB")
```

**经验公式**:
```
训练内存 ≈ 参数量 × 16-20 bytes (BF16 + Adam)
```

---

### Q: 为什么显存不断增长？

**常见原因**:
1. 损失历史累积
   ```python
   # ❌ 错误
   losses.append(loss)
   
   # ✅ 正确
   losses.append(loss.item())  # 用 .item() 取出数值
   ```

2. 未释放中间变量
   ```python
   # 使用 del 和 torch.cuda.empty_cache()
   del intermediate_tensor
   torch.cuda.empty_cache()
   ```

---

## ⚡ 性能问题

### Q: torch.compile 后为什么反而更慢？

**原因**: 编译开销、动态 shape、不支持的操作

**解决**:
```python
# 1. 使用更快的编译模式
model = compile_model(model, mode="reduce-overhead")

# 2. 固定 batch size，避免动态 shape
# 3. 预热：前几个 batch 会慢，之后会快

# 4. 检查警告
import torch._dynamo
torch._dynamo.config.verbose = True
```

---

### Q: 数据加载是瓶颈，GPU 利用率低怎么办？

**症状**: `nvidia-smi` 显示 GPU 利用率忽高忽低

**解决**:
```python
DataLoader(
    dataset,
    num_workers=8,           # 增加 worker 数
    pin_memory=True,         # 固定内存
    prefetch_factor=4,       # 增加预取
    persistent_workers=True, # 保持 worker
)
```

---

### Q: 多卡训练速度不是线性提升？

**可能原因**:
1. 通信瓶颈 → 检查网络
2. 数据加载瓶颈 → 增加 num_workers
3. 负载不均衡 → 检查 batch size

**调试**:
```bash
export NCCL_DEBUG=INFO
```

---

## 🎯 微调相关

### Q: LoRA 应该作用于哪些层？

**推荐目标层**:
```python
# Transformer 注意力层
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

# 更激进 (效果可能更好，显存需求更高)
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                  "gate_proj", "up_proj", "down_proj"]
```

---

### Q: LoRA 的 r 值怎么选？

| r 值 | 参数量 | 效果 | 推荐场景 |
|:----:|:------:|:----:|:---------|
| 4 | 最少 | 一般 | 快速实验 |
| 8 | 少 | 较好 | 通用推荐 |
| 16 | 中等 | 好 | 复杂任务 |
| 64+ | 较多 | 最好 | 追求极限 |

---

## 🔧 环境问题

### Q: CUDA 版本不匹配怎么办？

```bash
# 检查版本
python -c "import torch; print(torch.version.cuda)"
nvidia-smi  # 查看驱动支持的 CUDA

# 解决: 安装匹配的 PyTorch
pip install torch==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

---

### Q: NCCL 错误怎么调试？

```bash
# 启用调试信息
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# 常见问题
# - 防火墙阻止端口 → 开放端口或禁用防火墙
# - 网卡选择错误 → export NCCL_SOCKET_IFNAME=eth0
# - InfiniBand 问题 → export NCCL_IB_DISABLE=1
```

---

### Q: 如何在没有 GPU 的机器上测试代码？

```python
# 设置环境变量
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# 或使用 CPU 后端
setup_distributed(backend="gloo")  # 替代 nccl
```

---

## 📊 评估相关

### Q: 如何保证评估结果可复现？

```python
import torch
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

---

## 🔗 其他资源

- [DDP vs FSDP 详解](./DDPvsFSDP.md)
- [性能调优指南](./PerformanceTuning.md)
- [API 文档](./API.md)
- [PyTorch 官方 FAQ](https://pytorch.org/docs/stable/notes/faq.html)

---

<div align="center">

**没找到答案？** 提交 [Issue](https://github.com/your-org/gpu_optimize_demo/issues) 🐛

</div>
