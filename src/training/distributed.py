"""
Distributed Training - DDP, FSDP, and DeepSpeed Integration.

Provides comprehensive distributed training support:
- Distributed Data Parallel (DDP) - Simple multi-GPU training
- Fully Sharded Data Parallel (FSDP) - Memory-efficient large model training
- DeepSpeed ZeRO optimization
- Multi-node training setup
- Automatic checkpointing and resumption
- Distributed metrics and logging

Usage:
    # DDP (simple, recommended for models < GPU memory)
    trainer = DistributedTrainer(model, strategy="ddp")
    trainer.train(dataloader)
    
    # FSDP (for large models that don't fit in single GPU)
    trainer = DistributedTrainer(model, strategy="fsdp")
    trainer.train(dataloader)
"""

import os
import time
import json
import logging
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    CPUOffload,
    MixedPrecision,
    BackwardPrefetch,
    StateDictType,
    FullStateDictConfig,
    ShardedStateDictConfig,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
from torch.utils.data import DataLoader, DistributedSampler
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union, Tuple
import functools


# =============================================================================
# Distributed Data Parallel (DDP)
# =============================================================================

@dataclass
class DDPConfig:
    """
    Configuration for Distributed Data Parallel (DDP).
    
    DDP is recommended when:
    - Model fits in single GPU memory
    - You want simplest distributed training setup
    - You don't need gradient sharding
    
    Attributes:
        find_unused_parameters: Set True if model has unused parameters in forward
        broadcast_buffers: Whether to sync module buffers every forward
        gradient_as_bucket_view: Memory optimization for gradients
        static_graph: Enable optimizations for static computational graphs
        bucket_cap_mb: Bucket size for gradient AllReduce (default 25MB)
    """
    # Core settings
    find_unused_parameters: bool = False
    broadcast_buffers: bool = True
    
    # Performance optimizations
    gradient_as_bucket_view: bool = True
    static_graph: bool = False
    bucket_cap_mb: float = 25.0
    
    # Mixed precision
    mixed_precision: bool = True
    precision: str = "fp16"  # fp16, bf16
    
    # Gradient settings
    gradient_clipping: Optional[float] = 1.0
    gradient_accumulation_steps: int = 1
    
    # Checkpointing
    checkpoint_activations: bool = False


class DDPWrapper:
    """
    Wrapper for PyTorch's DistributedDataParallel.
    
    Provides a clean interface for DDP training with:
    - Automatic process group management
    - Mixed precision training support
    - Gradient clipping and accumulation
    - Checkpoint saving/loading
    
    Example:
        >>> config = DDPConfig(mixed_precision=True)
        >>> wrapper = DDPWrapper(config)
        >>> model = wrapper.wrap(model)
        >>> 
        >>> for batch in dataloader:
        ...     with wrapper.autocast():
        ...         loss = model(batch)
        ...     wrapper.backward(loss, optimizer)
    """
    
    def __init__(self, config: Optional[DDPConfig] = None):
        self.config = config or DDPConfig()
        self._scaler = None
        self._step_count = 0
        self._setup_amp()
    
    def _setup_amp(self):
        """Setup automatic mixed precision."""
        if self.config.mixed_precision and torch.cuda.is_available():
            self._scaler = torch.cuda.amp.GradScaler()
    
    @property
    def local_rank(self) -> int:
        """Get local rank."""
        return int(os.environ.get("LOCAL_RANK", 0))
    
    @property
    def device(self) -> torch.device:
        """Get device for this rank."""
        return torch.device(f"cuda:{self.local_rank}")
    
    def wrap(self, model: nn.Module, device_ids: Optional[List[int]] = None) -> DDP:
        """
        Wrap model with DDP.
        
        Args:
            model: Model to wrap
            device_ids: GPU device IDs (defaults to local rank)
            
        Returns:
            DDP-wrapped model
        """
        if device_ids is None:
            device_ids = [self.local_rank]
        
        # Move model to device
        model = model.to(self.device)
        
        # Apply activation checkpointing if enabled
        if self.config.checkpoint_activations:
            self._apply_activation_checkpointing(model)
        
        # Wrap with DDP
        ddp_model = DDP(
            model,
            device_ids=device_ids,
            find_unused_parameters=self.config.find_unused_parameters,
            broadcast_buffers=self.config.broadcast_buffers,
            gradient_as_bucket_view=self.config.gradient_as_bucket_view,
            static_graph=self.config.static_graph,
            bucket_cap_mb=self.config.bucket_cap_mb,
        )
        
        return ddp_model
    
    def _apply_activation_checkpointing(self, model: nn.Module):
        """Apply activation checkpointing to model."""
        from torch.utils.checkpoint import checkpoint_sequential
        # Apply to transformer blocks if available
        for name, module in model.named_children():
            if hasattr(module, 'gradient_checkpointing_enable'):
                module.gradient_checkpointing_enable()
    
    @contextmanager
    def autocast(self):
        """Context manager for mixed precision forward pass."""
        if self.config.mixed_precision and torch.cuda.is_available():
            dtype = torch.float16 if self.config.precision == "fp16" else torch.bfloat16
            with torch.cuda.amp.autocast(dtype=dtype):
                yield
        else:
            yield
    
    def backward(
        self, 
        loss: torch.Tensor, 
        optimizer: torch.optim.Optimizer,
        model: nn.Module,
    ) -> bool:
        """
        Backward pass with gradient accumulation and clipping.
        
        Args:
            loss: Loss tensor
            optimizer: Optimizer
            model: Model (for gradient clipping)
            
        Returns:
            True if optimizer step was taken
        """
        self._step_count += 1
        
        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps
        
        if self.config.mixed_precision and self._scaler is not None:
            self._scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Only step optimizer at accumulation boundary
        if self._step_count % self.config.gradient_accumulation_steps == 0:
            if self.config.mixed_precision and self._scaler is not None:
                # Unscale gradients for clipping
                self._scaler.unscale_(optimizer)
                
                # Gradient clipping
                if self.config.gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        self.config.gradient_clipping
                    )
                
                self._scaler.step(optimizer)
                self._scaler.update()
            else:
                if self.config.gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        self.config.gradient_clipping
                    )
                optimizer.step()
            
            optimizer.zero_grad()
            return True
        
        return False
    
    def save_checkpoint(
        self,
        model: DDP,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        path: str,
        extra_state: Optional[Dict] = None,
    ):
        """
        Save DDP checkpoint (only on rank 0).
        
        Args:
            model: DDP model
            optimizer: Optimizer
            epoch: Current epoch
            path: Path to save checkpoint
            extra_state: Additional state to save
        """
        if get_rank() != 0:
            return
        
        checkpoint = {
            "model": model.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "scaler": self._scaler.state_dict() if self._scaler else None,
            "config": self.config.__dict__,
        }
        if extra_state:
            checkpoint.update(extra_state)
        
        torch.save(checkpoint, path)
        logging.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(
        self,
        model: DDP,
        optimizer: torch.optim.Optimizer,
        path: str,
    ) -> Dict:
        """
        Load DDP checkpoint.
        
        Args:
            model: DDP model
            optimizer: Optimizer
            path: Path to checkpoint
            
        Returns:
            Extra state from checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        model.module.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        
        if self._scaler and checkpoint.get("scaler"):
            self._scaler.load_state_dict(checkpoint["scaler"])
        
        logging.info(f"Checkpoint loaded from {path}")
        return checkpoint


# =============================================================================
# Fully Sharded Data Parallel (FSDP)
# =============================================================================

@dataclass
class FSDPConfig:
    """
    Configuration for Fully Sharded Data Parallel (FSDP).
    
    FSDP is recommended when:
    - Model doesn't fit in single GPU memory
    - You need to train 7B+ parameter models
    - You want maximum memory efficiency
    
    Memory savings by sharding strategy:
    - FULL_SHARD: ~3x memory reduction (params + grads + optimizer states)
    - SHARD_GRAD_OP: ~2x reduction (grads + optimizer states only)
    - HYBRID_SHARD: Balance between memory and communication
    """
    # Sharding strategy
    sharding_strategy: str = "full_shard"  # full_shard, shard_grad_op, no_shard, hybrid_shard
    
    # Offloading
    cpu_offload: bool = False
    
    # Mixed Precision
    mixed_precision: bool = True
    precision: str = "bf16"  # fp16, bf16 (bf16 recommended for stability)
    
    # Auto wrapping
    auto_wrap_policy: str = "size"  # size, transformer
    min_num_params: int = 100_000_000  # 100M params for size-based
    transformer_layer_cls: Optional[Set[Type[nn.Module]]] = None
    
    # Performance tuning
    backward_prefetch: str = "backward_pre"  # backward_pre, backward_post
    forward_prefetch: bool = True
    limit_all_gathers: bool = True
    use_orig_params: bool = True  # Enable for torch.compile compatibility
    
    # Activation checkpointing
    activation_checkpointing: bool = False
    
    # Gradient settings
    gradient_clipping: Optional[float] = 1.0
    gradient_accumulation_steps: int = 1
    
    # State dict settings
    state_dict_type: str = "full"  # full, sharded


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


def all_gather(tensor: torch.Tensor) -> List[torch.Tensor]:
    """Gather tensors from all processes."""
    if not dist.is_initialized():
        return [tensor]
    
    world_size = get_world_size()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return gathered


def broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcast tensor from source rank."""
    if dist.is_initialized():
        dist.broadcast(tensor, src=src)
    return tensor


# =============================================================================
# FSDP Checkpoint Utilities
# =============================================================================

def save_fsdp_checkpoint(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: str,
    state_dict_type: str = "full",
    extra_state: Optional[Dict] = None,
):
    """
    Save FSDP checkpoint.
    
    Args:
        model: FSDP model
        optimizer: Optimizer
        epoch: Current epoch
        path: Path to save checkpoint
        state_dict_type: "full" or "sharded"
        extra_state: Additional state to save
    """
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if state_dict_type == "full":
        # Full state dict (consolidate to rank 0)
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            model_state = model.state_dict()
            optim_state = FSDP.optim_state_dict(model, optimizer)
            
            if get_rank() == 0:
                checkpoint = {
                    "model": model_state,
                    "optimizer": optim_state,
                    "epoch": epoch,
                }
                if extra_state:
                    checkpoint.update(extra_state)
                torch.save(checkpoint, path)
                logging.info(f"Full FSDP checkpoint saved to {path}")
    else:
        # Sharded state dict (each rank saves its shard)
        with FSDP.state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
            ShardedStateDictConfig(offload_to_cpu=True),
        ):
            model_state = model.state_dict()
            optim_state = FSDP.optim_state_dict(model, optimizer)
            
            rank = get_rank()
            shard_path = save_path.parent / f"{save_path.stem}_rank{rank}{save_path.suffix}"
            
            checkpoint = {
                "model": model_state,
                "optimizer": optim_state,
                "epoch": epoch,
                "rank": rank,
            }
            if extra_state:
                checkpoint.update(extra_state)
            torch.save(checkpoint, shard_path)
            logging.info(f"Sharded FSDP checkpoint saved to {shard_path}")
    
    barrier()


def load_fsdp_checkpoint(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    path: str,
    state_dict_type: str = "full",
) -> Dict:
    """
    Load FSDP checkpoint.
    
    Args:
        model: FSDP model
        optimizer: Optimizer
        path: Path to checkpoint
        state_dict_type: "full" or "sharded"
        
    Returns:
        Extra state from checkpoint
    """
    if state_dict_type == "full":
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            # Load on rank 0 and broadcast
            if get_rank() == 0:
                checkpoint = torch.load(path, map_location="cpu")
            else:
                checkpoint = {}
            
            barrier()
            
            # Load model state
            model.load_state_dict(checkpoint.get("model", {}))
            
            # Load optimizer state
            if "optimizer" in checkpoint:
                optim_state = FSDP.optim_state_dict_to_load(
                    model, optimizer, checkpoint["optimizer"]
                )
                optimizer.load_state_dict(optim_state)
    else:
        # Load sharded checkpoint
        save_path = Path(path)
        rank = get_rank()
        shard_path = save_path.parent / f"{save_path.stem}_rank{rank}{save_path.suffix}"
        
        with FSDP.state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
            ShardedStateDictConfig(offload_to_cpu=True),
        ):
            checkpoint = torch.load(shard_path, map_location="cpu")
            model.load_state_dict(checkpoint["model"])
            
            if "optimizer" in checkpoint:
                optim_state = FSDP.optim_state_dict_to_load(
                    model, optimizer, checkpoint["optimizer"]
                )
                optimizer.load_state_dict(optim_state)
    
    barrier()
    logging.info(f"FSDP checkpoint loaded from {path}")
    return checkpoint


# =============================================================================
# Distributed Metrics
# =============================================================================

class DistributedMetrics:
    """
    Distributed metrics tracker for training.
    
    Automatically aggregates metrics across all processes.
    
    Example:
        >>> metrics = DistributedMetrics()
        >>> metrics.update({"loss": 0.5, "accuracy": 0.9})
        >>> metrics.update({"loss": 0.4, "accuracy": 0.92})
        >>> print(metrics.get_average())
    """
    
    def __init__(self):
        self._metrics: Dict[str, List[float]] = defaultdict(list)
        self._device = torch.device(f"cuda:{get_local_rank()}")
    
    def update(self, metrics: Dict[str, float]):
        """Add metric values."""
        for key, value in metrics.items():
            self._metrics[key].append(value)
    
    def get_average(self, sync: bool = True) -> Dict[str, float]:
        """
        Get averaged metrics.
        
        Args:
            sync: If True, synchronize across processes
            
        Returns:
            Averaged metrics
        """
        result = {}
        for key, values in self._metrics.items():
            avg = sum(values) / len(values) if values else 0.0
            
            if sync and dist.is_initialized():
                tensor = torch.tensor(avg, device=self._device)
                all_reduce(tensor, op="avg")
                avg = tensor.item()
            
            result[key] = avg
        
        return result
    
    def reset(self):
        """Clear all metrics."""
        self._metrics.clear()
    
    def log(self, step: int, prefix: str = "train/"):
        """Log metrics (rank 0 only)."""
        if get_rank() != 0:
            return
        
        avg_metrics = self.get_average()
        msg = f"Step {step}: " + ", ".join(
            f"{prefix}{k}={v:.4f}" for k, v in avg_metrics.items()
        )
        logging.info(msg)
        return avg_metrics


# =============================================================================
# Unified Distributed Trainer
# =============================================================================

@dataclass
class DistributedTrainerConfig:
    """
    Configuration for DistributedTrainer.
    
    Attributes:
        strategy: "ddp", "fsdp", or "deepspeed"
        ddp_config: DDP configuration (if strategy="ddp")
        fsdp_config: FSDP configuration (if strategy="fsdp")
        deepspeed_config: DeepSpeed configuration (if strategy="deepspeed")
        logging_dir: Directory for logs and checkpoints
        log_interval: Steps between logging
        checkpoint_interval: Steps between checkpoints
        max_grad_norm: Max gradient norm for clipping
    """
    strategy: str = "ddp"  # ddp, fsdp, deepspeed
    
    # Strategy-specific configs
    ddp_config: Optional[DDPConfig] = None
    fsdp_config: Optional[FSDPConfig] = None
    deepspeed_config: Optional[DeepSpeedConfig] = None
    
    # Training
    logging_dir: str = "./logs"
    log_interval: int = 10
    checkpoint_interval: int = 1000
    max_grad_norm: float = 1.0
    
    # Resume
    resume_from_checkpoint: Optional[str] = None


class DistributedTrainer:
    """
    Unified interface for distributed training with DDP, FSDP, or DeepSpeed.
    
    Provides:
    - Automatic strategy selection and model wrapping
    - Distributed data loading with DistributedSampler
    - Checkpoint saving and resumption
    - Distributed metrics logging
    
    Example:
        >>> # Setup distributed
        >>> setup_distributed()
        >>> 
        >>> # Create trainer
        >>> config = DistributedTrainerConfig(strategy="fsdp")
        >>> trainer = DistributedTrainer(model, config)
        >>> 
        >>> # Train
        >>> for epoch in range(num_epochs):
        ...     trainer.train_epoch(dataloader, optimizer)
        ...     trainer.save_checkpoint(f"checkpoint_epoch{epoch}.pt")
        >>> 
        >>> # Cleanup
        >>> cleanup_distributed()
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[DistributedTrainerConfig] = None,
    ):
        self.config = config or DistributedTrainerConfig()
        self.original_model = model
        self.metrics = DistributedMetrics()
        self._step = 0
        self._scaler = None
        
        # Initialize strategy
        self._init_strategy()
        
        # Wrap model
        self.model = self._wrap_model(model)
        
        # Setup logging
        self._setup_logging()
        
        logging.info(
            f"DistributedTrainer initialized with strategy={self.config.strategy}, "
            f"world_size={get_world_size()}, rank={get_rank()}"
        )
    
    def _init_strategy(self):
        """Initialize the chosen distributed strategy."""
        if self.config.strategy == "ddp":
            self.wrapper = DDPWrapper(self.config.ddp_config)
            self._scaler = self.wrapper._scaler
        elif self.config.strategy == "fsdp":
            self.wrapper = FSDPWrapper(self.config.fsdp_config)
            # FSDP handles mixed precision internally
        elif self.config.strategy == "deepspeed":
            self.wrapper = DeepSpeedWrapper(self.config.deepspeed_config)
        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")
    
    def _wrap_model(self, model: nn.Module) -> nn.Module:
        """Wrap model with the chosen strategy."""
        if self.config.strategy in ["ddp", "fsdp"]:
            return self.wrapper.wrap(model)
        else:
            # DeepSpeed wrapping happens during optimizer initialization
            return model
    
    def _setup_logging(self):
        """Setup logging directory."""
        if get_rank() == 0:
            log_dir = Path(self.config.logging_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s",
                handlers=[
                    logging.FileHandler(log_dir / "training.log"),
                    logging.StreamHandler(),
                ],
            )
    
    def prepare_dataloader(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 4,
        **kwargs,
    ) -> DataLoader:
        """
        Create a distributed DataLoader.
        
        Args:
            dataset: Dataset to load
            batch_size: Per-GPU batch size
            shuffle: Whether to shuffle (handled by sampler)
            num_workers: Number of data loading workers
            
        Returns:
            DataLoader with DistributedSampler
        """
        sampler = DistributedSampler(
            dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=shuffle,
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            **kwargs,
        )
    
    @contextmanager
    def autocast(self):
        """Context manager for automatic mixed precision."""
        if self.config.strategy == "ddp":
            with self.wrapper.autocast():
                yield
        elif self.config.strategy == "fsdp":
            # FSDP handles AMP via MixedPrecision config
            yield
        else:
            yield
    
    def backward(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> bool:
        """
        Backward pass with gradient handling.
        
        Args:
            loss: Loss tensor
            optimizer: Optimizer
            
        Returns:
            True if optimizer step was taken
        """
        self._step += 1
        
        if self.config.strategy == "ddp":
            return self.wrapper.backward(loss, optimizer, self.model)
        else:
            loss.backward()
            
            # Gradient clipping
            if self.config.max_grad_norm:
                if self.config.strategy == "fsdp":
                    self.model.clip_grad_norm_(self.config.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )
            
            optimizer.step()
            optimizer.zero_grad()
            return True
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log training metrics."""
        self.metrics.update(metrics)
        
        step = step or self._step
        if step % self.config.log_interval == 0:
            self.metrics.log(step)
            self.metrics.reset()
    
    def save_checkpoint(
        self,
        path: str,
        optimizer: torch.optim.Optimizer,
        extra_state: Optional[Dict] = None,
    ):
        """Save training checkpoint."""
        if self.config.strategy == "ddp":
            self.wrapper.save_checkpoint(
                self.model, optimizer, self._step // 1000, path, extra_state
            )
        elif self.config.strategy == "fsdp":
            state_dict_type = (
                self.config.fsdp_config.state_dict_type
                if self.config.fsdp_config
                else "full"
            )
            save_fsdp_checkpoint(
                self.model, optimizer, self._step // 1000, path,
                state_dict_type, extra_state
            )
        elif self.config.strategy == "deepspeed":
            self.wrapper.save_checkpoint(self.model, path)
        
        barrier()
    
    def load_checkpoint(
        self,
        path: str,
        optimizer: torch.optim.Optimizer,
    ) -> Dict:
        """Load training checkpoint."""
        if self.config.strategy == "ddp":
            return self.wrapper.load_checkpoint(self.model, optimizer, path)
        elif self.config.strategy == "fsdp":
            state_dict_type = (
                self.config.fsdp_config.state_dict_type
                if self.config.fsdp_config
                else "full"
            )
            return load_fsdp_checkpoint(
                self.model, optimizer, path, state_dict_type
            )
        elif self.config.strategy == "deepspeed":
            self.wrapper.load_checkpoint(self.model, path)
            return {}
    
    def get_model(self) -> nn.Module:
        """Get the underlying model (unwrapped)."""
        if self.config.strategy == "ddp":
            return self.model.module
        elif self.config.strategy == "fsdp":
            return self.model
        return self.model
    
    @property
    def device(self) -> torch.device:
        """Get the device for this rank."""
        return torch.device(f"cuda:{get_local_rank()}")


# =============================================================================
# Convenience Functions
# =============================================================================

def print_model_size(model: nn.Module, name: str = "Model"):
    """Print model size information (rank 0 only)."""
    if get_rank() != 0:
        return
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate memory
    param_size_mb = total_params * 4 / 1024 / 1024  # Assuming float32
    
    print(f"\n{'='*50}")
    print(f"{name} Size Information")
    print(f"{'='*50}")
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable:        {total_params - trainable_params:,}")
    print(f"Model Size (FP32):    {param_size_mb:.2f} MB")
    print(f"Model Size (FP16):    {param_size_mb / 2:.2f} MB")
    print(f"{'='*50}\n")


def estimate_memory_usage(
    model: nn.Module,
    batch_size: int,
    seq_length: int = 512,
    precision: str = "fp16",
) -> Dict[str, float]:
    """
    Estimate memory usage for training.
    
    Returns:
        Dict with memory estimates in GB
    """
    total_params = sum(p.numel() for p in model.parameters())
    
    # Bytes per parameter
    bytes_per_param = 2 if precision in ["fp16", "bf16"] else 4
    
    # Model parameters
    param_memory = total_params * bytes_per_param / 1024**3
    
    # Gradients (same size as parameters)
    grad_memory = param_memory
    
    # Optimizer states (Adam has 2 states per parameter)
    # Optimizer states are typically in fp32
    optim_memory = total_params * 4 * 2 / 1024**3
    
    # Activations (rough estimate)
    activation_memory = batch_size * seq_length * 4096 * bytes_per_param / 1024**3
    
    total_memory = param_memory + grad_memory + optim_memory + activation_memory
    
    return {
        "parameters_gb": param_memory,
        "gradients_gb": grad_memory,
        "optimizer_states_gb": optim_memory,
        "activations_gb": activation_memory,
        "total_gb": total_memory,
    }


def auto_select_strategy(
    model: nn.Module,
    available_gpus: int = 1,
    gpu_memory_gb: float = 24.0,
) -> str:
    """
    Automatically select the best distributed strategy.
    
    Args:
        model: Model to train
        available_gpus: Number of available GPUs
        gpu_memory_gb: Memory per GPU in GB
        
    Returns:
        Recommended strategy: "ddp", "fsdp", or "deepspeed"
    """
    estimates = estimate_memory_usage(model, batch_size=1)
    total_memory = estimates["total_gb"]
    
    if total_memory < gpu_memory_gb * 0.7:
        # Model fits comfortably in single GPU
        return "ddp"
    elif total_memory < gpu_memory_gb * available_gpus * 0.7:
        # Model fits with sharding
        return "fsdp"
    else:
        # Need aggressive memory optimization
        return "deepspeed"
