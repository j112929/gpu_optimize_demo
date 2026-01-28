"""
Distributed Training - FSDP and DeepSpeed Integration.

Provides wrappers for:
- Fully Sharded Data Parallel (FSDP)
- DeepSpeed ZeRO optimization
- Multi-node training setup
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    CPUOffload,
    MixedPrecision,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union
import functools


@dataclass
class FSDPConfig:
    """Configuration for FSDP."""
    # Sharding
    sharding_strategy: str = "full_shard"  # full_shard, shard_grad_op, no_shard
    
    # Offloading
    cpu_offload: bool = False
    
    # Mixed Precision
    mixed_precision: bool = True
    precision: str = "fp16"  # fp16, bf16
    
    # Wrapping
    auto_wrap_policy: str = "size"  # size, transformer
    min_num_params: int = 100_000_000  # 100M params
    transformer_layer_cls: Optional[Set[Type[nn.Module]]] = None
    
    # Performance
    backward_prefetch: str = "backward_pre"  # backward_pre, backward_post
    forward_prefetch: bool = True
    limit_all_gathers: bool = True
    
    # Checkpointing
    activation_checkpointing: bool = False


class FSDPWrapper:
    """
    Wrapper for Fully Sharded Data Parallel (FSDP).
    
    Enables training of large models that don't fit in single GPU memory
    by sharding model parameters, gradients, and optimizer states.
    
    Example:
        >>> config = FSDPConfig(sharding_strategy="full_shard")
        >>> wrapper = FSDPWrapper(config)
        >>> model = wrapper.wrap(model)
        >>> 
        >>> # Now model is sharded across GPUs
        >>> output = model(input)
        >>> loss.backward()
    """
    
    def __init__(self, config: Optional[FSDPConfig] = None):
        self.config = config or FSDPConfig()
    
    def _get_sharding_strategy(self) -> ShardingStrategy:
        """Get FSDP sharding strategy."""
        strategies = {
            "full_shard": ShardingStrategy.FULL_SHARD,
            "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
            "no_shard": ShardingStrategy.NO_SHARD,
            "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
        }
        return strategies.get(self.config.sharding_strategy, ShardingStrategy.FULL_SHARD)
    
    def _get_mixed_precision(self) -> Optional[MixedPrecision]:
        """Get mixed precision config."""
        if not self.config.mixed_precision:
            return None
        
        dtype = torch.float16 if self.config.precision == "fp16" else torch.bfloat16
        
        return MixedPrecision(
            param_dtype=dtype,
            reduce_dtype=dtype,
            buffer_dtype=dtype,
        )
    
    def _get_auto_wrap_policy(self) -> Optional[Callable]:
        """Get auto wrap policy."""
        if self.config.auto_wrap_policy == "size":
            return functools.partial(
                size_based_auto_wrap_policy,
                min_num_params=self.config.min_num_params,
            )
        elif self.config.auto_wrap_policy == "transformer":
            if self.config.transformer_layer_cls is None:
                return None
            return functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls=self.config.transformer_layer_cls,
            )
        return None
    
    def _get_backward_prefetch(self) -> Optional[BackwardPrefetch]:
        """Get backward prefetch setting."""
        if self.config.backward_prefetch == "backward_pre":
            return BackwardPrefetch.BACKWARD_PRE
        elif self.config.backward_prefetch == "backward_post":
            return BackwardPrefetch.BACKWARD_POST
        return None
    
    def wrap(self, model: nn.Module) -> FSDP:
        """
        Wrap model with FSDP.
        
        Args:
            model: Model to wrap
            
        Returns:
            FSDP-wrapped model
        """
        fsdp_model = FSDP(
            model,
            sharding_strategy=self._get_sharding_strategy(),
            cpu_offload=CPUOffload(offload_params=True) if self.config.cpu_offload else None,
            mixed_precision=self._get_mixed_precision(),
            auto_wrap_policy=self._get_auto_wrap_policy(),
            backward_prefetch=self._get_backward_prefetch(),
            forward_prefetch=self.config.forward_prefetch,
            limit_all_gathers=self.config.limit_all_gathers,
        )
        
        # Apply activation checkpointing if enabled
        if self.config.activation_checkpointing:
            self._apply_activation_checkpointing(fsdp_model)
        
        return fsdp_model
    
    def _apply_activation_checkpointing(self, model: FSDP):
        """Apply activation checkpointing to FSDP model."""
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            apply_activation_checkpointing,
            checkpoint_wrapper,
            CheckpointImpl,
        )
        
        # Apply to all FSDP units
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=lambda m: checkpoint_wrapper(
                m,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            ),
        )


def setup_distributed(
    backend: str = "nccl",
    init_method: str = "env://",
) -> int:
    """
    Setup distributed training.
    
    Args:
        backend: "nccl" for GPU, "gloo" for CPU
        init_method: How to initialize ("env://" uses environment variables)
        
    Returns:
        Local rank
    """
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method=init_method)
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    
    return local_rank


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


# =============================================================================
# DeepSpeed Integration
# =============================================================================

@dataclass
class DeepSpeedConfig:
    """Configuration for DeepSpeed."""
    # ZeRO Stage
    zero_stage: int = 2  # 0, 1, 2, 3
    
    # Offloading
    offload_optimizer: bool = False
    offload_param: bool = False
    offload_device: str = "cpu"  # cpu, nvme
    
    # Mixed Precision
    fp16: bool = True
    bf16: bool = False
    
    # Optimization
    gradient_accumulation_steps: int = 1
    gradient_clipping: float = 1.0
    
    # Memory
    partition_activations: bool = False
    contiguous_gradients: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to DeepSpeed config dict."""
        config = {
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto",
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "gradient_clipping": self.gradient_clipping,
            "zero_optimization": {
                "stage": self.zero_stage,
                "contiguous_gradients": self.contiguous_gradients,
            },
        }
        
        # ZeRO Stage 3 specific
        if self.zero_stage == 3:
            config["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"] = True
        
        # Offloading
        if self.offload_optimizer:
            config["zero_optimization"]["offload_optimizer"] = {
                "device": self.offload_device,
                "pin_memory": True,
            }
        
        if self.offload_param and self.zero_stage == 3:
            config["zero_optimization"]["offload_param"] = {
                "device": self.offload_device,
                "pin_memory": True,
            }
        
        # Precision
        if self.fp16:
            config["fp16"] = {
                "enabled": True,
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "initial_scale_power": 16,
            }
        elif self.bf16:
            config["bf16"] = {"enabled": True}
        
        # Activation checkpointing
        if self.partition_activations:
            config["activation_checkpointing"] = {
                "partition_activations": True,
                "contiguous_memory_optimization": True,
            }
        
        return config


class DeepSpeedWrapper:
    """
    Wrapper for DeepSpeed initialization.
    
    Provides:
    - ZeRO optimization (Stage 1/2/3)
    - CPU/NVMe offloading
    - Activation checkpointing
    
    Example:
        >>> config = DeepSpeedConfig(zero_stage=2)
        >>> wrapper = DeepSpeedWrapper(config)
        >>> model, optimizer, _, scheduler = wrapper.initialize(
        ...     model=model,
        ...     optimizer=optimizer,
        ... )
    """
    
    def __init__(self, config: Optional[DeepSpeedConfig] = None):
        self.config = config or DeepSpeedConfig()
        self._check_deepspeed()
    
    def _check_deepspeed(self):
        """Check if DeepSpeed is available."""
        try:
            import deepspeed
            self.deepspeed = deepspeed
        except ImportError:
            raise ImportError(
                "DeepSpeed not installed. Install with: pip install deepspeed"
            )
    
    def initialize(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        model_parameters: Optional[Any] = None,
        lr_scheduler: Optional[Any] = None,
        config: Optional[Dict] = None,
    ):
        """
        Initialize model with DeepSpeed.
        
        Returns:
            Tuple of (model_engine, optimizer, dataloader, lr_scheduler)
        """
        ds_config = config or self.config.to_dict()
        
        model_engine, optimizer, _, lr_scheduler = self.deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            model_parameters=model_parameters or model.parameters(),
            config=ds_config,
            lr_scheduler=lr_scheduler,
        )
        
        return model_engine, optimizer, None, lr_scheduler
    
    def save_checkpoint(
        self,
        model_engine,
        save_dir: str,
        tag: str = "latest",
    ):
        """Save DeepSpeed checkpoint."""
        model_engine.save_checkpoint(save_dir, tag=tag)
    
    def load_checkpoint(
        self,
        model_engine,
        load_dir: str,
        tag: Optional[str] = None,
    ):
        """Load DeepSpeed checkpoint."""
        model_engine.load_checkpoint(load_dir, tag=tag)


# =============================================================================
# Utilities
# =============================================================================

def get_world_size() -> int:
    """Get world size (number of processes)."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """Get global rank."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_local_rank() -> int:
    """Get local rank (within node)."""
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process() -> bool:
    """Check if this is the main process."""
    return get_rank() == 0


def barrier():
    """Synchronization barrier."""
    if dist.is_initialized():
        dist.barrier()


def all_reduce(tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
    """All-reduce tensor across processes."""
    if not dist.is_initialized():
        return tensor
    
    ops = {
        "sum": dist.ReduceOp.SUM,
        "avg": dist.ReduceOp.SUM,  # Divide after
        "max": dist.ReduceOp.MAX,
        "min": dist.ReduceOp.MIN,
    }
    
    dist.all_reduce(tensor, op=ops.get(op, dist.ReduceOp.SUM))
    
    if op == "avg":
        tensor /= get_world_size()
    
    return tensor
