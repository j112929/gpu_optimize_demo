"""
Fast DataLoader - Optimized data loading with best practices.

This module provides an enhanced DataLoader with:
- Optimal worker configuration
- Pinned memory for faster GPU transfer
- Persistent workers to reduce startup overhead
- Automatic prefetching
"""

import os
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional, Union

import torch
from torch.utils.data import DataLoader, Dataset, Sampler


@dataclass
class DataLoaderConfig:
    """Configuration for optimized DataLoader."""
    
    # Basic settings
    batch_size: int = 32
    shuffle: bool = True
    
    # Worker settings
    num_workers: int = 4
    prefetch_factor: int = 2
    persistent_workers: bool = True
    
    # Memory settings
    pin_memory: bool = True
    pin_memory_device: str = ""
    
    # Performance settings
    drop_last: bool = False
    timeout: float = 0
    
    # Multiprocessing
    multiprocessing_context: Optional[str] = None  # 'spawn', 'fork', 'forkserver'
    
    @classmethod
    def auto_config(cls, dataset_size: int, gpu_memory_gb: float = 16) -> "DataLoaderConfig":
        """
        Auto-configure based on dataset size and available resources.
        
        Args:
            dataset_size: Number of samples in dataset
            gpu_memory_gb: Available GPU memory in GB
        """
        # Determine optimal batch size based on GPU memory
        # Rough heuristic: larger GPU memory allows larger batches
        if gpu_memory_gb >= 32:
            batch_size = 128
        elif gpu_memory_gb >= 16:
            batch_size = 64
        else:
            batch_size = 32
        
        # Determine optimal workers based on CPU cores
        cpu_count = os.cpu_count() or 4
        num_workers = min(cpu_count - 1, 8)  # Leave one core free
        
        # Adjust prefetch based on dataset size
        prefetch_factor = 2 if dataset_size > 10000 else 1
        
        return cls(
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=num_workers > 0,
        )


class FastDataLoader:
    """
    High-performance DataLoader wrapper with automatic optimization.
    
    Features:
    - Automatic worker configuration
    - Memory pinning for faster GPU transfer
    - Persistent workers to reduce startup overhead
    - Integrated prefetching
    
    Example:
        >>> dataset = MyDataset()
        >>> loader = FastDataLoader(dataset, batch_size=64)
        >>> for batch in loader:
        ...     train_step(batch)
    """
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: Optional[int] = None,
        pin_memory: bool = True,
        drop_last: bool = False,
        sampler: Optional[Sampler] = None,
        collate_fn: Optional[Callable] = None,
        config: Optional[DataLoaderConfig] = None,
        **kwargs
    ):
        """
        Initialize FastDataLoader.
        
        Args:
            dataset: PyTorch dataset
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes (auto if None)
            pin_memory: Pin memory for faster GPU transfer
            drop_last: Drop last incomplete batch
            sampler: Custom sampler
            collate_fn: Custom collate function
            config: Optional DataLoaderConfig
            **kwargs: Additional DataLoader arguments
        """
        self.dataset = dataset
        
        # Use config or create from arguments
        if config is None:
            if num_workers is None:
                num_workers = self._optimal_workers()
            
            config = DataLoaderConfig(
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory and torch.cuda.is_available(),
                drop_last=drop_last,
            )
        
        self.config = config
        
        # Build DataLoader arguments
        loader_kwargs = {
            "dataset": dataset,
            "batch_size": config.batch_size,
            "shuffle": config.shuffle if sampler is None else False,
            "num_workers": config.num_workers,
            "pin_memory": config.pin_memory,
            "drop_last": config.drop_last,
            "timeout": config.timeout,
            "sampler": sampler,
            "collate_fn": collate_fn,
        }
        
        # Add optional arguments
        if config.num_workers > 0:
            loader_kwargs["prefetch_factor"] = config.prefetch_factor
            loader_kwargs["persistent_workers"] = config.persistent_workers
            
            if config.multiprocessing_context:
                loader_kwargs["multiprocessing_context"] = config.multiprocessing_context
        
        if config.pin_memory_device:
            loader_kwargs["pin_memory_device"] = config.pin_memory_device
        
        # Merge additional kwargs
        loader_kwargs.update(kwargs)
        
        self._loader = DataLoader(**loader_kwargs)
        self._iterator: Optional[Iterator] = None
    
    def _optimal_workers(self) -> int:
        """Determine optimal number of workers."""
        cpu_count = os.cpu_count() or 4
        
        # Use half of available CPUs, capped at 8
        optimal = min(cpu_count // 2, 8)
        
        # Ensure at least 1 worker if multiprocessing is desired
        return max(optimal, 0)
    
    def __iter__(self) -> Iterator:
        """Return iterator over batches."""
        self._iterator = iter(self._loader)
        return self
    
    def __next__(self) -> Any:
        """Get next batch."""
        if self._iterator is None:
            raise StopIteration
        return next(self._iterator)
    
    def __len__(self) -> int:
        """Return number of batches."""
        return len(self._loader)
    
    @property
    def batch_size(self) -> int:
        """Get batch size."""
        return self.config.batch_size
    
    @property
    def num_workers(self) -> int:
        """Get number of workers."""
        return self.config.num_workers
    
    def metrics(self) -> dict:
        """Get loader metrics."""
        return {
            "dataset_size": len(self.dataset),
            "batch_size": self.config.batch_size,
            "num_batches": len(self),
            "num_workers": self.config.num_workers,
            "pin_memory": self.config.pin_memory,
            "prefetch_factor": self.config.prefetch_factor,
        }


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: Optional[int] = None,
    pin_memory: bool = True,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    **kwargs
) -> DataLoader:
    """
    Create an optimized DataLoader with best practices.
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        shuffle: Whether to shuffle (ignored if distributed)
        num_workers: Number of workers (auto if None)
        pin_memory: Pin memory for GPU
        distributed: Use distributed sampler
        rank: Process rank for distributed
        world_size: Total processes for distributed
        **kwargs: Additional DataLoader arguments
        
    Returns:
        Optimized DataLoader
    """
    # Auto-configure workers
    if num_workers is None:
        cpu_count = os.cpu_count() or 4
        num_workers = min(cpu_count // 2, 8)
    
    # Handle distributed training
    sampler = None
    if distributed and world_size > 1:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
        )
        shuffle = False  # Sampler handles shuffling
    
    # Build DataLoader
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory and torch.cuda.is_available(),
        "sampler": sampler,
    }
    
    # Add worker-specific settings
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["persistent_workers"] = True
    
    loader_kwargs.update(kwargs)
    
    return DataLoader(**loader_kwargs)


class InfiniteDataLoader:
    """
    DataLoader that cycles indefinitely.
    
    Useful for training loops where you want continuous iteration
    without manually resetting the loader.
    
    Example:
        >>> loader = InfiniteDataLoader(dataset, batch_size=32)
        >>> for i, batch in enumerate(loader):
        ...     if i >= max_steps:
        ...         break
        ...     train_step(batch)
    """
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        **kwargs
    ):
        """
        Initialize infinite loader.
        
        Args:
            dataset: PyTorch dataset
            batch_size: Batch size
            **kwargs: Additional DataLoader arguments
        """
        self._loader = FastDataLoader(dataset, batch_size=batch_size, **kwargs)
        self._iterator: Optional[Iterator] = None
    
    def __iter__(self) -> "InfiniteDataLoader":
        """Return self as iterator."""
        return self
    
    def __next__(self) -> Any:
        """Get next batch, cycling if necessary."""
        if self._iterator is None:
            self._iterator = iter(self._loader)
        
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self._loader)
            return next(self._iterator)
    
    @property
    def batch_size(self) -> int:
        """Get batch size."""
        return self._loader.batch_size


class BalancedDataLoader:
    """
    DataLoader that balances samples across classes.
    
    Useful for imbalanced datasets where you want equal representation
    of each class in every batch.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Initialize balanced loader.
        
        Args:
            dataset: PyTorch dataset
            batch_size: Batch size
            labels: Class labels for each sample
            **kwargs: Additional DataLoader arguments
        """
        if labels is None:
            # Try to extract labels from dataset
            if hasattr(dataset, 'targets'):
                labels = torch.tensor(dataset.targets)
            elif hasattr(dataset, 'labels'):
                labels = torch.tensor(dataset.labels)
            else:
                raise ValueError("Labels must be provided for BalancedDataLoader")
        
        # Calculate weights for balanced sampling
        class_counts = torch.bincount(labels)
        weights = 1.0 / class_counts[labels].float()
        
        sampler = torch.utils.data.WeightedRandomSampler(
            weights,
            num_samples=len(dataset),
            replacement=True,
        )
        
        self._loader = FastDataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            **kwargs
        )
    
    def __iter__(self) -> Iterator:
        """Return iterator."""
        return iter(self._loader)
    
    def __len__(self) -> int:
        """Return number of batches."""
        return len(self._loader)
