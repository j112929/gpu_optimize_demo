"""
Ray Trainer - Distributed training orchestration.

Provides multi-node, multi-GPU training with automatic fault tolerance.
"""

import ray
from ray import train
from ray.train import ScalingConfig, RunConfig, CheckpointConfig
from ray.train.torch import TorchTrainer
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class RayTrainerConfig:
    """Configuration for Ray distributed training."""
    # Scaling
    num_workers: int = 4
    use_gpu: bool = True
    resources_per_worker: Dict[str, float] = field(default_factory=lambda: {"CPU": 4, "GPU": 1})
    
    # Training
    num_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-3
    
    # Checkpointing
    checkpoint_frequency: int = 1
    keep_checkpoints_num: int = 3
    
    # Fault tolerance
    max_failures: int = 3
    
    # Logging
    log_dir: str = "./ray_results"


class RayTrainer:
    """
    Ray-based distributed trainer for PyTorch models.
    
    Features:
    - Multi-node, multi-GPU training
    - Automatic data sharding
    - Fault tolerance and recovery
    - Integrated checkpointing
    
    Example:
        >>> config = RayTrainerConfig(num_workers=4, use_gpu=True)
        >>> trainer = RayTrainer(config)
        >>> trainer.fit(model, train_dataset, val_dataset)
    """
    
    def __init__(self, config: RayTrainerConfig):
        self.config = config
        self._trainer: Optional[TorchTrainer] = None
        self._result = None
        
        if not ray.is_initialized():
            ray.init()
    
    def fit(
        self,
        model_fn: Callable[[], nn.Module],
        train_dataset,
        val_dataset=None,
        optimizer_fn: Optional[Callable] = None,
        loss_fn: Optional[Callable] = None,
    ):
        """
        Train a model in distributed fashion.
        
        Args:
            model_fn: Function that returns the model
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
            optimizer_fn: Function that creates optimizer
            loss_fn: Loss function
        """
        def train_loop_per_worker(config):
            # Get distributed components
            model = train.torch.prepare_model(model_fn())
            
            # Create DataLoader with automatic sharding
            train_loader = DataLoader(
                train_dataset,
                batch_size=config["batch_size"],
                shuffle=True,
            )
            train_loader = train.torch.prepare_data_loader(train_loader)
            
            # Optimizer
            if optimizer_fn:
                optimizer = optimizer_fn(model.parameters())
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
            
            # Loss
            criterion = loss_fn if loss_fn else nn.CrossEntropyLoss()
            
            # Training loop
            for epoch in range(config["num_epochs"]):
                model.train()
                total_loss = 0
                num_batches = 0
                
                for batch in train_loader:
                    if isinstance(batch, (list, tuple)):
                        inputs, targets = batch[0], batch[1]
                    else:
                        inputs, targets = batch["input"], batch["target"]
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                
                avg_loss = total_loss / num_batches
                
                # Report metrics
                train.report(
                    {"loss": avg_loss, "epoch": epoch},
                    checkpoint=train.Checkpoint.from_dict(
                        {"model_state_dict": model.state_dict()}
                    ) if (epoch + 1) % config["checkpoint_frequency"] == 0 else None,
                )
        
        # Configure scaling
        scaling_config = ScalingConfig(
            num_workers=self.config.num_workers,
            use_gpu=self.config.use_gpu,
            resources_per_worker=self.config.resources_per_worker,
        )
        
        # Configure checkpointing
        checkpoint_config = CheckpointConfig(
            num_to_keep=self.config.keep_checkpoints_num,
        )
        
        # Configure run
        run_config = RunConfig(
            storage_path=self.config.log_dir,
            checkpoint_config=checkpoint_config,
            failure_config=train.FailureConfig(max_failures=self.config.max_failures),
        )
        
        # Create trainer
        self._trainer = TorchTrainer(
            train_loop_per_worker=train_loop_per_worker,
            train_loop_config={
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "num_epochs": self.config.num_epochs,
                "checkpoint_frequency": self.config.checkpoint_frequency,
            },
            scaling_config=scaling_config,
            run_config=run_config,
        )
        
        # Run training
        self._result = self._trainer.fit()
        
        return self._result
    
    def get_best_checkpoint(self):
        """Get the best checkpoint from training."""
        if self._result:
            return self._result.checkpoint
        return None
    
    def load_model(self, model_fn: Callable[[], nn.Module]) -> nn.Module:
        """Load model from best checkpoint."""
        checkpoint = self.get_best_checkpoint()
        if checkpoint:
            model = model_fn()
            state_dict = checkpoint.to_dict()["model_state_dict"]
            model.load_state_dict(state_dict)
            return model
        raise RuntimeError("No checkpoint available")


def distributed_train(
    model_fn: Callable[[], nn.Module],
    train_fn: Callable,
    num_workers: int = 4,
    use_gpu: bool = True,
    config: Optional[Dict] = None,
) -> Any:
    """
    Simple API for distributed training.
    
    Args:
        model_fn: Function returning the model
        train_fn: Training function
        num_workers: Number of workers
        use_gpu: Whether to use GPUs
        config: Additional config
        
    Returns:
        Training result
    """
    if not ray.is_initialized():
        ray.init()
    
    def train_loop(train_config):
        model = train.torch.prepare_model(model_fn())
        train_fn(model, train_config)
    
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop,
        train_loop_config=config or {},
        scaling_config=ScalingConfig(
            num_workers=num_workers,
            use_gpu=use_gpu,
        ),
    )
    
    return trainer.fit()


# =============================================================================
# Training Utilities
# =============================================================================

def get_distributed_context() -> Dict[str, Any]:
    """Get current distributed training context."""
    return {
        "world_size": train.get_context().get_world_size(),
        "world_rank": train.get_context().get_world_rank(),
        "local_rank": train.get_context().get_local_rank(),
        "node_rank": train.get_context().get_node_rank(),
    }


def sync_batch_norm(model: nn.Module) -> nn.Module:
    """Convert BatchNorm to SyncBatchNorm for distributed training."""
    return nn.SyncBatchNorm.convert_sync_batchnorm(model)


def gradient_checkpoint_model(model: nn.Module) -> nn.Module:
    """Enable gradient checkpointing for memory efficiency."""
    from torch.utils.checkpoint import checkpoint_sequential
    
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    return model
