"""
Training Optimization Module - Mixed precision, gradient accumulation, and more.

Provides:
- Automatic Mixed Precision (AMP)
- Gradient Accumulation & Checkpointing
- DDP / FSDP / DeepSpeed Integration
- Flash Attention (v2/v3/SDPA)
- Gradient Compression
- Optimized DataLoader
- Fused Operations
"""

from src.training.amp import (
    AMPTrainer,
    AMPConfig,
    auto_cast_forward,
    scale_loss,
)
from src.training.gradient import (
    GradientAccumulator,
    gradient_checkpoint_model,
    clip_grad_norm,
)
from src.training.distributed import (
    # DDP
    DDPWrapper,
    DDPConfig,
    # FSDP
    FSDPWrapper,
    FSDPConfig,
    # DeepSpeed
    DeepSpeedWrapper,
    DeepSpeedConfig,
    # Unified Trainer
    DistributedTrainer,
    DistributedTrainerConfig,
    DistributedMetrics,
    # Setup and utilities
    setup_distributed,
    cleanup_distributed,
    get_world_size,
    get_rank,
    get_local_rank,
    is_main_process,
    barrier,
    all_reduce,
    all_gather,
    broadcast,
    # Checkpoint utilities
    save_fsdp_checkpoint,
    load_fsdp_checkpoint,
    # Helper functions
    print_model_size,
    estimate_memory_usage,
    auto_select_strategy,
)
from src.training.optimizer import (
    create_optimizer,
    OptimizerConfig,
    get_cosine_schedule,
    get_linear_schedule,
)

# Flash Attention
from src.training.attention import (
    FlashAttention,
    FlashAttentionConfig,
    SlidingWindowAttention,
    MemoryEfficientAttentionContext,
    detect_flash_attention_version,
    create_attention_layer,
    benchmark_attention_backends,
)

# Gradient Compression
from src.training.compression import (
    GradientCompressionConfig,
    TopKCompressor,
    RandomCompressor,
    QuantizationCompressor,
    PowerSGDCompressor,
    DistributedGradientCompressor,
    GradientCompressionHook,
    create_gradient_compressor,
    estimate_compression_ratio,
)

# Optimized DataLoader
from src.training.dataloader import (
    DataLoaderConfig,
    CUDAPrefetcher,
    BackgroundPrefetcher,
    StreamingDataset,
    MemoryMappedDataset,
    DynamicBatchCollator,
    create_dataloader,
    create_prefetched_loader,
    benchmark_dataloader,
)

# Fused Operations
from src.training.fused_ops import (
    FusedOpsConfig,
    FusedLayerNorm,
    RMSNorm,
    FusedCrossEntropyLoss,
    FusedAdamW,
    FusedRoPE,
    FusedSwiGLU,
    apply_fused_ops,
    benchmark_fused_ops,
)

# ZeRO++
from src.training.zero_plus import (
    ZeroPlusConfig,
    QuantizedCommunicator,
)

# Parallelism
from src.training.parallelism import (
    ParallelismConfig,
    ColumnParallelLinear,
    RowParallelLinear,
    SequenceParallelWrapper,
)

# Pipeline
from src.training.pipeline import (
    PipelineStage,
    PipelineScheduler,
    split_model_into_stages,
)

__all__ = [
    # AMP
    "AMPTrainer",
    "AMPConfig",
    "auto_cast_forward",
    "scale_loss",
    # Gradient
    "GradientAccumulator",
    "gradient_checkpoint_model",
    "clip_grad_norm",
    # DDP
    "DDPWrapper",
    "DDPConfig",
    # FSDP
    "FSDPWrapper",
    "FSDPConfig",
    # DeepSpeed
    "DeepSpeedWrapper",
    "DeepSpeedConfig",
    # Unified Trainer
    "DistributedTrainer",
    "DistributedTrainerConfig",
    "DistributedMetrics",
    # Setup
    "setup_distributed",
    "cleanup_distributed",
    "get_world_size",
    "get_rank",
    "get_local_rank",
    "is_main_process",
    "barrier",
    "all_reduce",
    "all_gather",
    "broadcast",
    # Checkpoints
    "save_fsdp_checkpoint",
    "load_fsdp_checkpoint",
    # Helpers
    "print_model_size",
    "estimate_memory_usage",
    "auto_select_strategy",
    # Optimizer
    "create_optimizer",
    "OptimizerConfig",
    "get_cosine_schedule",
    "get_linear_schedule",
    
    # === NEW: Flash Attention ===
    "FlashAttention",
    "FlashAttentionConfig",
    "SlidingWindowAttention",
    "MemoryEfficientAttentionContext",
    "detect_flash_attention_version",
    "create_attention_layer",
    "benchmark_attention_backends",
    
    # === NEW: Gradient Compression ===
    "GradientCompressionConfig",
    "TopKCompressor",
    "RandomCompressor",
    "QuantizationCompressor",
    "PowerSGDCompressor",
    "DistributedGradientCompressor",
    "GradientCompressionHook",
    "create_gradient_compressor",
    "estimate_compression_ratio",
    
    # === NEW: DataLoader ===
    "DataLoaderConfig",
    "CUDAPrefetcher",
    "BackgroundPrefetcher",
    "StreamingDataset",
    "MemoryMappedDataset",
    "DynamicBatchCollator",
    "create_dataloader",
    "create_prefetched_loader",
    "benchmark_dataloader",
    
    # === NEW: Fused Operations ===
    "FusedOpsConfig",
    "FusedLayerNorm",
    "RMSNorm",
    "FusedCrossEntropyLoss",
    "FusedAdamW",
    "FusedRoPE",
    "FusedSwiGLU",
    "apply_fused_ops",
    "benchmark_fused_ops",
    
    # === NEW: ZeRO++ ===
    "ZeroPlusConfig",
    "QuantizedCommunicator",
    
    # === NEW: Parallelism ===
    "ParallelismConfig",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "SequenceParallelWrapper",
    
    # === NEW: Pipeline ===
    "PipelineStage",
    "PipelineScheduler",
    "split_model_into_stages",
]
