# API 参考文档

> GPU Optimization Toolkit 核心 API 速查

---

## 📦 模块总览

```python
from src.training import (
    # 分布式
    DDPWrapper, DDPConfig,
    FSDPWrapper, FSDPConfig,
    DeepSpeedWrapper, DeepSpeedConfig,
    DistributedTrainer, DistributedTrainerConfig,
    
    # 工具
    setup_distributed, cleanup_distributed,
    get_rank, get_world_size, is_main_process,
    
    # 混合精度
    AMPTrainer, AMPConfig,
    
    # 梯度
    GradientAccumulator, gradient_checkpoint_model,
)

from src.compile import (
    compile_model, CUDAGraphWrapper, TensorRTConverter,
)

from src.post_training import (
    LoRAModel, LoRAConfig, apply_lora, merge_lora,
    DPOTrainer, DPOConfig,
)

from src.inference import (
    SpeculativeDecoder, SpeculativeConfig,
    quantize_model, QuantizationConfig,
    ContinuousBatcher, PagedKVCache,
)

from src.evaluation import (
    run_benchmark, SafetyEvaluator,
)
```

---

## 🌐 分布式训练

### DDPConfig

```python
@dataclass
class DDPConfig:
    # 核心设置
    find_unused_parameters: bool = False  # 有未使用参数时设为 True
    broadcast_buffers: bool = True        # 同步 buffer
    
    # 性能优化
    gradient_as_bucket_view: bool = True  # 减少内存复制
    static_graph: bool = False            # 固定计算图优化
    bucket_cap_mb: float = 25.0           # AllReduce bucket 大小
    
    # 混合精度
    mixed_precision: bool = True
    precision: str = "fp16"               # "fp16" 或 "bf16"
    
    # 梯度
    gradient_clipping: Optional[float] = 1.0
    gradient_accumulation_steps: int = 1
    
    # 激活检查点
    checkpoint_activations: bool = False
```

### DDPWrapper

```python
class DDPWrapper:
    def __init__(self, config: DDPConfig = None): ...
    
    def wrap(self, model: nn.Module) -> DDP:
        """包装模型为 DDP"""
    
    def autocast(self) -> ContextManager:
        """混合精度上下文"""
    
    def backward(self, loss, optimizer, model) -> bool:
        """反向传播 + 梯度处理，返回是否执行了 optimizer.step()"""
    
    def save_checkpoint(self, model, optimizer, epoch, path, extra_state=None):
        """保存检查点 (仅 rank 0)"""
    
    def load_checkpoint(self, model, optimizer, path) -> Dict:
        """加载检查点"""
    
    @property
    def device(self) -> torch.device:
        """当前设备"""
```

### FSDPConfig

```python
@dataclass
class FSDPConfig:
    # 分片策略
    sharding_strategy: str = "full_shard"
    # "full_shard": 参数+梯度+优化器都分片 (~3× 内存节省)
    # "shard_grad_op": 仅梯度+优化器分片 (~2× 内存节省)
    # "no_shard": 不分片，等同于 DDP
    # "hybrid_shard": 节点内分片，节点间复制
    
    # Offload
    cpu_offload: bool = False
    
    # 混合精度
    mixed_precision: bool = True
    precision: str = "bf16"
    
    # 自动包装
    auto_wrap_policy: str = "size"       # "size" 或 "transformer"
    min_num_params: int = 100_000_000    # size 模式的阈值
    transformer_layer_cls: Optional[Set[Type]] = None  # transformer 模式需要
    
    # 性能
    backward_prefetch: str = "backward_pre"
    forward_prefetch: bool = True
    limit_all_gathers: bool = True
    use_orig_params: bool = True         # torch.compile 兼容
    
    # 激活检查点
    activation_checkpointing: bool = False
    
    # 梯度
    gradient_clipping: Optional[float] = 1.0
    gradient_accumulation_steps: int = 1
    
    # 检查点
    state_dict_type: str = "full"        # "full" 或 "sharded"
```

### FSDPWrapper

```python
class FSDPWrapper:
    def __init__(self, config: FSDPConfig = None): ...
    
    def wrap(self, model: nn.Module) -> FSDP:
        """包装模型为 FSDP"""
```

### DistributedTrainer

```python
class DistributedTrainer:
    """统一的分布式训练接口"""
    
    def __init__(self, model: nn.Module, config: DistributedTrainerConfig): ...
    
    def prepare_dataloader(self, dataset, batch_size, shuffle=True, num_workers=4) -> DataLoader:
        """创建分布式 DataLoader"""
    
    def autocast(self) -> ContextManager:
        """混合精度上下文"""
    
    def backward(self, loss, optimizer) -> bool:
        """反向传播"""
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """记录指标 (自动跨进程同步)"""
    
    def save_checkpoint(self, path, optimizer, extra_state=None):
        """保存检查点"""
    
    def load_checkpoint(self, path, optimizer) -> Dict:
        """加载检查点"""
    
    @property
    def model(self) -> nn.Module:
        """包装后的模型"""
    
    @property
    def device(self) -> torch.device:
        """当前设备"""
```

---

## ⚡ 工具函数

### 分布式

```python
def setup_distributed(backend: str = "nccl") -> int:
    """初始化分布式环境，返回 local_rank"""

def cleanup_distributed():
    """清理分布式环境"""

def get_rank() -> int:
    """获取全局 rank"""

def get_local_rank() -> int:
    """获取节点内 rank"""

def get_world_size() -> int:
    """获取总进程数"""

def is_main_process() -> bool:
    """是否为主进程 (rank 0)"""

def barrier():
    """同步屏障"""

def all_reduce(tensor: Tensor, op: str = "sum") -> Tensor:
    """AllReduce 操作，op: 'sum', 'avg', 'max', 'min'"""

def all_gather(tensor: Tensor) -> List[Tensor]:
    """AllGather 操作"""

def broadcast(tensor: Tensor, src: int = 0) -> Tensor:
    """Broadcast 操作"""
```

### 辅助

```python
def print_model_size(model: nn.Module, name: str = "Model"):
    """打印模型大小信息 (仅 rank 0)"""

def estimate_memory_usage(model, batch_size, seq_length=512, precision="fp16") -> Dict:
    """估算训练内存需求
    
    返回:
        {
            "parameters_gb": float,
            "gradients_gb": float,
            "optimizer_states_gb": float,
            "activations_gb": float,
            "total_gb": float,
        }
    """

def auto_select_strategy(model, available_gpus: int, gpu_memory_gb: float) -> str:
    """自动选择分布式策略
    
    返回: "ddp", "fsdp", 或 "deepspeed"
    """
```

---

## 🔧 编译优化

### compile_model

```python
def compile_model(
    model: nn.Module,
    mode: str = "default",       # "default", "reduce-overhead", "max-autotune"
    fullgraph: bool = False,     # 强制完整图编译
    dynamic: bool = False,       # 动态 shape 支持
    backend: str = "inductor",   # 编译后端
) -> nn.Module:
    """torch.compile 包装器"""
```

### CUDAGraphWrapper

```python
class CUDAGraphWrapper:
    def __init__(self, model: nn.Module, example_input: Tensor): ...
    
    def __call__(self, input: Tensor) -> Tensor:
        """执行 CUDA Graph (低延迟推理)"""
```

---

## 🎯 后训练

### LoRAConfig

```python
@dataclass
class LoRAConfig:
    r: int = 8                   # Low-rank 维度
    alpha: int = 16              # 缩放因子
    dropout: float = 0.0
    target_modules: List[str] = None  # 目标层，如 ["q_proj", "v_proj"]
    bias: str = "none"           # "none", "all", "lora_only"
```

### LoRA 函数

```python
def apply_lora(model: nn.Module, config: LoRAConfig) -> nn.Module:
    """应用 LoRA 到模型"""

def merge_lora(model: nn.Module) -> nn.Module:
    """合并 LoRA 权重到基础模型 (推理时无开销)"""
```

### DPOTrainer

```python
class DPOTrainer:
    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        config: DPOConfig,
    ): ...
    
    def step(
        self,
        prompt_ids: Tensor,
        chosen_ids: Tensor,
        rejected_ids: Tensor,
    ) -> Dict[str, float]:
        """执行一步 DPO 训练
        
        返回: {"loss": float, "accuracy": float, ...}
        """
```

---

## ⚡ 推理优化

### SpeculativeDecoder

```python
class SpeculativeDecoder:
    def __init__(
        self,
        target_model: nn.Module,     # 大模型 (验证)
        draft_model: nn.Module,      # 小模型 (草拟)
        config: SpeculativeConfig = None,
    ): ...
    
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> Tensor:
        """投机解码生成"""
    
    def print_stats(self):
        """打印接受率和加速比"""
```

### quantize_model

```python
def quantize_model(
    model: nn.Module,
    config: QuantizationConfig,
    calibration_data: Dataset = None,
) -> nn.Module:
    """模型量化
    
    QuantizationConfig:
        bits: int = 4            # 4 或 8
        method: str = "gptq"     # "gptq" 或 "awq"
        group_size: int = 128
    """
```

---

## 📊 评估

### run_benchmark

```python
def run_benchmark(
    model: nn.Module,
    tokenizer,
    benchmark: str,              # "mmlu", "hellaswag", "truthfulqa"
    num_samples: int = None,
) -> BenchmarkResult:
    """运行基准测试
    
    BenchmarkResult:
        accuracy: float
        num_samples: int
        details: Dict
    """
```

### SafetyEvaluator

```python
class SafetyEvaluator:
    def evaluate_model(
        self,
        model: nn.Module,
        tokenizer,
        test_prompts: List[str],
    ) -> SafetyReport:
        """安全评估
        
        SafetyReport:
            toxicity_score: float
            bias_metrics: Dict
            flagged_outputs: List
        """
```

---

## 📚 更多文档

- [快速入门](./QuickStart.md)
- [DDP vs FSDP 详解](./DDPvsFSDP.md)
- [性能调优指南](./PerformanceTuning.md)
- [常见问题](./FAQ.md)

---

<div align="center">

**需要更多示例？** 查看 [examples/](../examples/) 目录

</div>
