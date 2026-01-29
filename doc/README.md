# 📚 GPU Optimization Toolkit 文档

> 完整的 GPU 优化工具包文档中心

---

## 🚀 快速导航

| 文档 | 描述 | 适合谁 |
|:-----|:-----|:-------|
| [**快速入门**](./QuickStart.md) | 5 分钟上手核心功能 | 新用户 |
| [**DDP vs FSDP**](./DDPvsFSDP.md) | 分布式训练策略选择 | 需要多卡训练 |
| [**API 参考**](./API.md) | 完整 API 文档 | 开发者 |
| [**性能调优**](./PerformanceTuning.md) | 榨干 GPU 性能 | 追求极致 |
| [**常见问题**](./FAQ.md) | 问题排查 | 遇到问题时 |

---

## 📖 按场景阅读

### 我想要...

#### 🌐 多卡训练

1. **首先阅读**: [DDP vs FSDP](./DDPvsFSDP.md) - 选择正确的策略
2. **然后查看**: [API 参考 - 分布式训练](./API.md#分布式训练)
3. **遇到问题**: [FAQ - 分布式训练](./FAQ.md#分布式训练)

#### ⚡ 加速训练

1. [快速入门 - 训练加速](./QuickStart.md#训练加速)
2. [性能调优指南](./PerformanceTuning.md)

#### 💾 解决 OOM

1. [FAQ - 内存问题](./FAQ.md#内存问题)
2. [性能调优 - 内存优化](./PerformanceTuning.md#内存优化)

#### 🎯 微调大模型

1. [快速入门 - 高效微调](./QuickStart.md#高效微调)
2. [FAQ - 微调相关](./FAQ.md#微调相关)

---

## 📁 文档结构

```
doc/
├── README.md              # 本文档 (索引)
├── QuickStart.md          # 快速入门
├── DDPvsFSDP.md           # DDP vs FSDP 对比
├── API.md                 # API 参考文档
├── PerformanceTuning.md   # 性能调优指南
└── FAQ.md                 # 常见问题
```

---

## 🔗 其他资源

### 项目资源

- [README](../README.md) - 项目概览
- [示例代码](../examples/) - 可运行的示例
- [配置文件](../configs/) - 配置模板

### 外部链接

- [PyTorch 官方文档](https://pytorch.org/docs/stable/)
- [PyTorch 分布式教程](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [NVIDIA 深度学习性能指南](https://developer.nvidia.com/deep-learning-performance-training-inference)

---

## 📝 文档贡献

发现错误或想要改进？欢迎提交 PR！

```bash
# 克隆仓库
git clone https://github.com/your-org/gpu_optimize_demo.git

# 编辑文档
cd gpu_optimize_demo/doc
# 进行修改...

# 提交 PR
git add .
git commit -m "docs: improve documentation"
git push
```

---

<div align="center">

**GPU Optimization Toolkit** — 让 GPU 优化变得简单 🚀

[GitHub](https://github.com/your-org/gpu_optimize_demo) · [Issues](https://github.com/your-org/gpu_optimize_demo/issues)

</div>
