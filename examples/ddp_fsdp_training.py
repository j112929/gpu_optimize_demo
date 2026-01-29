#!/usr/bin/env python3
"""
Example: DDP and FSDP Distributed Training

This example demonstrates how to use both DDP and FSDP for distributed training:
1. DDP (Distributed Data Parallel) - Simple multi-GPU training
2. FSDP (Fully Sharded Data Parallel) - Memory-efficient large model training

Key Features:
- Automatic distributed setup
- Mixed precision training (AMP)
- Gradient accumulation and clipping
- Checkpoint saving and loading
- Distributed metrics logging

Usage:
    # Single GPU (for testing)
    python examples/ddp_fsdp_training.py --strategy ddp
    
    # Multi-GPU with DDP
    torchrun --nproc_per_node=4 examples/ddp_fsdp_training.py --strategy ddp
    
    # Multi-GPU with FSDP (for large models)
    torchrun --nproc_per_node=4 examples/ddp_fsdp_training.py --strategy fsdp
    
    # Multi-node with FSDP
    torchrun --nnodes=2 --node_rank=0 --master_addr=<MASTER_IP> --master_port=29500 \\
        --nproc_per_node=4 examples/ddp_fsdp_training.py --strategy fsdp

When to use DDP vs FSDP:
    - DDP: Model fits in single GPU memory. Simpler, lower overhead.
    - FSDP: Model too large for single GPU. Shards params/grads/optimizer states.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training import (
    # DDP
    DDPWrapper,
    DDPConfig,
    # FSDP
    FSDPWrapper,
    FSDPConfig,
    # Unified interface
    DistributedTrainer,
    DistributedTrainerConfig,
    # Utilities
    setup_distributed,
    cleanup_distributed,
    get_world_size,
    get_rank,
    get_local_rank,
    is_main_process,
    barrier,
    print_model_size,
    estimate_memory_usage,
    auto_select_strategy,
)


# =============================================================================
# Model Definition
# =============================================================================

class TransformerBlock(nn.Module):
    """A simple transformer block for demonstration."""
    
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class SimpleTransformer(nn.Module):
    """A simple transformer model for demonstration."""
    
    def __init__(
        self,
        vocab_size: int = 50000,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        max_seq_length: int = 512,
        num_classes: int = 2,
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(max_seq_length, hidden_size)
        
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length = input_ids.shape
        
        # Get embeddings
        positions = torch.arange(seq_length, device=input_ids.device)
        x = self.embedding(input_ids) + self.pos_embedding(positions)
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Pool and classify
        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling
        logits = self.classifier(x)
        
        return logits


# =============================================================================
# Dataset
# =============================================================================

class DummyDataset(Dataset):
    """Dummy dataset for demonstration."""
    
    def __init__(self, num_samples: int = 10000, seq_length: int = 128, vocab_size: int = 50000):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Random input sequence
        input_ids = torch.randint(0, self.vocab_size, (self.seq_length,))
        # Random binary label
        label = torch.randint(0, 2, (1,)).squeeze()
        return input_ids, label


# =============================================================================
# Training Functions
# =============================================================================

def train_with_ddp(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int = 2,
    lr: float = 1e-4,
    checkpoint_dir: str = "./checkpoints",
):
    """Train with DDP (Distributed Data Parallel)."""
    print_rank0("\n" + "=" * 70)
    print_rank0("Training with DDP (Distributed Data Parallel)")
    print_rank0("=" * 70)
    
    # Create DDP config
    config = DDPConfig(
        mixed_precision=True,
        precision="bf16",  # or "fp16"
        gradient_clipping=1.0,
        gradient_accumulation_steps=1,
        static_graph=False,  # Set True if model structure doesn't change
    )
    
    # Create wrapper and wrap model
    wrapper = DDPWrapper(config)
    ddp_model = wrapper.wrap(model)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=lr)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    device = wrapper.device
    global_step = 0
    
    for epoch in range(num_epochs):
        ddp_model.train()
        epoch_loss = 0.0
        
        for batch_idx, (input_ids, labels) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            # Forward pass with mixed precision
            with wrapper.autocast():
                logits = ddp_model(input_ids)
                loss = criterion(logits, labels)
            
            # Backward pass (handles scaling, accumulation, clipping)
            step_taken = wrapper.backward(loss, optimizer, ddp_model)
            
            epoch_loss += loss.item()
            global_step += 1
            
            if step_taken and global_step % 10 == 0:
                print_rank0(
                    f"Epoch {epoch+1}/{num_epochs}, "
                    f"Step {global_step}, "
                    f"Loss: {loss.item():.4f}"
                )
        
        avg_loss = epoch_loss / len(train_loader)
        print_rank0(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if is_main_process():
            checkpoint_path = f"{checkpoint_dir}/ddp_epoch{epoch+1}.pt"
            wrapper.save_checkpoint(ddp_model, optimizer, epoch + 1, checkpoint_path)
    
    print_rank0("\n✅ DDP training completed!")
    return ddp_model


def train_with_fsdp(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int = 2,
    lr: float = 1e-4,
    checkpoint_dir: str = "./checkpoints",
):
    """Train with FSDP (Fully Sharded Data Parallel)."""
    print_rank0("\n" + "=" * 70)
    print_rank0("Training with FSDP (Fully Sharded Data Parallel)")
    print_rank0("=" * 70)
    
    # Create FSDP config
    config = FSDPConfig(
        sharding_strategy="full_shard",  # Maximum memory savings
        mixed_precision=True,
        precision="bf16",
        auto_wrap_policy="transformer",  # Wrap each transformer block
        transformer_layer_cls={TransformerBlock},  # Specify layer type
        activation_checkpointing=True,  # Save memory
        gradient_clipping=1.0,
        state_dict_type="full",  # or "sharded" for large models
    )
    
    # Create wrapper and wrap model
    wrapper = FSDPWrapper(config)
    fsdp_model = wrapper.wrap(model)
    
    # Create optimizer (must be after FSDP wrapping)
    optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=lr)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    device = torch.device(f"cuda:{get_local_rank()}")
    global_step = 0
    
    for epoch in range(num_epochs):
        fsdp_model.train()
        epoch_loss = 0.0
        
        for batch_idx, (input_ids, labels) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass (FSDP handles mixed precision internally)
            logits = fsdp_model(input_ids)
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (FSDP method)
            fsdp_model.clip_grad_norm_(config.gradient_clipping)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            if global_step % 10 == 0:
                print_rank0(
                    f"Epoch {epoch+1}/{num_epochs}, "
                    f"Step {global_step}, "
                    f"Loss: {loss.item():.4f}"
                )
        
        avg_loss = epoch_loss / len(train_loader)
        print_rank0(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
        
        # Synchronize before checkpoint
        barrier()
    
    print_rank0("\n✅ FSDP training completed!")
    return fsdp_model


def train_with_unified_trainer(
    model: nn.Module,
    train_loader: DataLoader,
    strategy: str = "ddp",
    num_epochs: int = 2,
    lr: float = 1e-4,
):
    """Train using the unified DistributedTrainer interface."""
    print_rank0("\n" + "=" * 70)
    print_rank0(f"Training with DistributedTrainer (strategy={strategy})")
    print_rank0("=" * 70)
    
    # Create trainer config
    if strategy == "ddp":
        ddp_config = DDPConfig(mixed_precision=True, precision="bf16")
        config = DistributedTrainerConfig(
            strategy="ddp",
            ddp_config=ddp_config,
            log_interval=10,
        )
    else:
        fsdp_config = FSDPConfig(
            sharding_strategy="full_shard",
            mixed_precision=True,
            precision="bf16",
            transformer_layer_cls={TransformerBlock},
        )
        config = DistributedTrainerConfig(
            strategy="fsdp",
            fsdp_config=fsdp_config,
            log_interval=10,
        )
    
    # Create trainer
    trainer = DistributedTrainer(model, config)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=lr)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    global_step = 0
    
    for epoch in range(num_epochs):
        trainer.model.train()
        
        for batch_idx, (input_ids, labels) in enumerate(train_loader):
            input_ids = input_ids.to(trainer.device)
            labels = labels.to(trainer.device)
            
            # Forward pass with autocast
            with trainer.autocast():
                logits = trainer.model(input_ids)
                loss = criterion(logits, labels)
            
            # Backward pass (handles gradients and optimizer step)
            trainer.backward(loss, optimizer)
            
            # Log metrics
            trainer.log_metrics({"loss": loss.item()})
            
            global_step += 1
        
        print_rank0(f"Epoch {epoch+1} completed.")
    
    print_rank0("\n✅ Unified trainer training completed!")
    return trainer.model


# =============================================================================
# Helper Functions
# =============================================================================

def print_rank0(msg: str):
    """Print only on rank 0."""
    if is_main_process():
        print(msg)


def create_dataloaders(
    batch_size: int,
    num_workers: int = 4,
) -> DataLoader:
    """Create distributed data loaders."""
    dataset = DummyDataset(num_samples=1000)
    
    # Use DistributedSampler for multi-GPU
    from torch.utils.data.distributed import DistributedSampler
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=True,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return loader


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="DDP/FSDP Training Example")
    parser.add_argument(
        "--strategy",
        type=str,
        default="ddp",
        choices=["ddp", "fsdp", "auto", "unified"],
        help="Distributed training strategy",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--hidden-size", type=int, default=768, help="Model hidden size")
    parser.add_argument("--num-layers", type=int, default=6, help="Number of layers")
    args = parser.parse_args()
    
    # Setup distributed
    setup_distributed()
    
    print_rank0("=" * 70)
    print_rank0("DDP / FSDP TRAINING EXAMPLE")
    print_rank0("=" * 70)
    print_rank0(f"World Size: {get_world_size()}")
    print_rank0(f"Rank: {get_rank()}")
    print_rank0(f"Local Rank: {get_local_rank()}")
    print_rank0(f"Strategy: {args.strategy}")
    
    if torch.cuda.is_available():
        print_rank0(f"GPU: {torch.cuda.get_device_name()}")
        print_rank0(f"CUDA: {torch.version.cuda}")
    
    # Create model
    model = SimpleTransformer(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
    )
    
    # Print model info
    print_model_size(model, "SimpleTransformer")
    
    # Estimate memory usage
    if is_main_process():
        estimates = estimate_memory_usage(model, args.batch_size)
        print(f"Estimated Memory Usage:")
        print(f"  Parameters:       {estimates['parameters_gb']:.2f} GB")
        print(f"  Gradients:        {estimates['gradients_gb']:.2f} GB")
        print(f"  Optimizer states: {estimates['optimizer_states_gb']:.2f} GB")
        print(f"  Activations:      {estimates['activations_gb']:.2f} GB")
        print(f"  Total:            {estimates['total_gb']:.2f} GB")
    
    # Auto-select strategy if requested
    if args.strategy == "auto":
        args.strategy = auto_select_strategy(
            model,
            available_gpus=get_world_size(),
            gpu_memory_gb=torch.cuda.get_device_properties(0).total_memory / 1024**3,
        )
        print_rank0(f"Auto-selected strategy: {args.strategy}")
    
    # Create data loader
    train_loader = create_dataloaders(args.batch_size)
    
    # Create checkpoint directory
    checkpoint_dir = Path("./checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    try:
        if args.strategy == "ddp":
            train_with_ddp(
                model,
                train_loader,
                num_epochs=args.epochs,
                lr=args.lr,
                checkpoint_dir=str(checkpoint_dir),
            )
        elif args.strategy == "fsdp":
            train_with_fsdp(
                model,
                train_loader,
                num_epochs=args.epochs,
                lr=args.lr,
                checkpoint_dir=str(checkpoint_dir),
            )
        elif args.strategy == "unified":
            # Use the unified trainer interface
            train_with_unified_trainer(
                model,
                train_loader,
                strategy="ddp",  # or "fsdp"
                num_epochs=args.epochs,
                lr=args.lr,
            )
    except Exception as e:
        print(f"[Rank {get_rank()}] Error: {e}")
        raise
    finally:
        cleanup_distributed()
    
    print_rank0("\n🎉 All training completed successfully!")


if __name__ == "__main__":
    main()
