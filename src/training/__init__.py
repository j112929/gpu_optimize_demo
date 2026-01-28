"""
Training Optimization Module - Mixed precision, gradient accumulation, and more.

Provides:
- Automatic Mixed Precision (AMP)
- Gradient Accumulation
- Gradient Checkpointing
- DeepSpeed / FSDP Integration
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
    FSDPWrapper,
    FSDPConfig,
    DeepSpeedWrapper,
    setup_distributed,
)
from src.training.optimizer import (
    create_optimizer,
    OptimizerConfig,
    get_cosine_schedule,
    get_linear_schedule,
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
    # Distributed
    "FSDPWrapper",
    "FSDPConfig",
    "DeepSpeedWrapper",
    "setup_distributed",
    # Optimizer
    "create_optimizer",
    "OptimizerConfig",
    "get_cosine_schedule",
    "get_linear_schedule",
]
