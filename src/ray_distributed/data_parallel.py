"""
Ray Data Parallel - Distributed data loading and preprocessing.

Enables scalable data preprocessing across multiple nodes.
"""

import ray
from ray import data as ray_data
from typing import Any, Callable, Dict, Iterator, List, Optional, Union
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, IterableDataset


@dataclass
class RayDataConfig:
    """Configuration for Ray Data loading."""
    num_workers: int = 4
    batch_size: int = 32
    prefetch_batches: int = 2
    local_shuffle_buffer_size: int = 1000
    use_gpu: bool = True


class RayDataset:
    """
    Ray-based distributed dataset for scalable data loading.
    
    Wraps Ray Data for seamless integration with PyTorch training.
    
    Example:
        >>> ds = RayDataset.from_items(data_list)
        >>> ds = ds.map(preprocess_fn)
        >>> for batch in ds.iter_batches(batch_size=32):
        ...     train_step(batch)
    """
    
    def __init__(self, ray_dataset: ray_data.Dataset):
        self._dataset = ray_dataset
    
    @classmethod
    def from_items(cls, items: List[Any]) -> "RayDataset":
        """Create from a list of items."""
        ds = ray_data.from_items(items)
        return cls(ds)
    
    @classmethod
    def from_numpy(cls, arrays: Dict[str, Any]) -> "RayDataset":
        """Create from numpy arrays."""
        ds = ray_data.from_numpy(arrays)
        return cls(ds)
    
    @classmethod
    def from_torch(cls, dataset: Dataset) -> "RayDataset":
        """Create from PyTorch dataset."""
        items = [dataset[i] for i in range(len(dataset))]
        return cls.from_items(items)
    
    @classmethod
    def from_parquet(cls, path: str) -> "RayDataset":
        """Create from Parquet files."""
        ds = ray_data.read_parquet(path)
        return cls(ds)
    
    @classmethod
    def from_images(cls, path: str) -> "RayDataset":
        """Create from image directory."""
        ds = ray_data.read_images(path)
        return cls(ds)
    
    def map(
        self,
        fn: Callable,
        num_cpus: float = 1,
        num_gpus: float = 0,
        batch_size: Optional[int] = None,
    ) -> "RayDataset":
        """
        Apply a function to each item.
        
        Args:
            fn: Function to apply
            num_cpus: CPUs per worker
            num_gpus: GPUs per worker
            batch_size: If set, apply fn to batches
        """
        if batch_size:
            new_ds = self._dataset.map_batches(
                fn,
                batch_size=batch_size,
                num_cpus=num_cpus,
                num_gpus=num_gpus,
            )
        else:
            new_ds = self._dataset.map(
                fn,
                num_cpus=num_cpus,
                num_gpus=num_gpus,
            )
        return RayDataset(new_ds)
    
    def filter(self, fn: Callable[[Any], bool]) -> "RayDataset":
        """Filter items."""
        return RayDataset(self._dataset.filter(fn))
    
    def shuffle(self, seed: Optional[int] = None) -> "RayDataset":
        """Shuffle the dataset."""
        return RayDataset(self._dataset.random_shuffle(seed=seed))
    
    def split(self, n: int) -> List["RayDataset"]:
        """Split into n datasets."""
        splits = self._dataset.split(n)
        return [RayDataset(s) for s in splits]
    
    def iter_batches(
        self,
        batch_size: int = 32,
        prefetch_batches: int = 2,
        local_shuffle_buffer_size: Optional[int] = None,
    ) -> Iterator:
        """Iterate over batches."""
        return self._dataset.iter_batches(
            batch_size=batch_size,
            prefetch_batches=prefetch_batches,
            local_shuffle_buffer_size=local_shuffle_buffer_size,
        )
    
    def iter_torch_batches(
        self,
        batch_size: int = 32,
        device: str = "cuda",
    ) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate as PyTorch tensors."""
        for batch in self._dataset.iter_torch_batches(batch_size=batch_size):
            yield {k: v.to(device) for k, v in batch.items()}
    
    def count(self) -> int:
        """Count items."""
        return self._dataset.count()
    
    def schema(self):
        """Get schema."""
        return self._dataset.schema()
    
    @property
    def raw(self) -> ray_data.Dataset:
        """Get underlying Ray Dataset."""
        return self._dataset


class RayDataLoader:
    """
    Ray-based DataLoader for distributed training.
    
    Provides efficient data loading across multiple workers/nodes
    with automatic sharding.
    
    Example:
        >>> loader = RayDataLoader(dataset, batch_size=64, num_workers=4)
        >>> for batch in loader:
        ...     train_step(batch)
    """
    
    def __init__(
        self,
        dataset: Union[RayDataset, Dataset],
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle: bool = True,
        prefetch_batches: int = 2,
        device: str = "cuda",
    ):
        if isinstance(dataset, Dataset):
            dataset = RayDataset.from_torch(dataset)
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.prefetch_batches = prefetch_batches
        self.device = device
    
    def __iter__(self):
        ds = self.dataset
        if self.shuffle:
            ds = ds.shuffle()
        
        for batch in ds.iter_torch_batches(
            batch_size=self.batch_size,
            device=self.device,
        ):
            yield batch
    
    def __len__(self):
        return (self.dataset.count() + self.batch_size - 1) // self.batch_size


def distributed_map(
    items: List[Any],
    fn: Callable,
    num_cpus: float = 1,
    num_gpus: float = 0,
    batch_size: int = 100,
) -> List[Any]:
    """
    Apply a function to items in parallel using Ray.
    
    Args:
        items: List of items to process
        fn: Function to apply
        num_cpus: CPUs per task
        num_gpus: GPUs per task
        batch_size: Batch size for processing
        
    Returns:
        List of processed items
    """
    if not ray.is_initialized():
        ray.init()
    
    ds = RayDataset.from_items(items)
    ds = ds.map(fn, num_cpus=num_cpus, num_gpus=num_gpus, batch_size=batch_size)
    
    return list(ds.raw.take_all())


def distributed_preprocess(
    dataset: Dataset,
    preprocess_fn: Callable,
    num_workers: int = 4,
    batch_size: int = 100,
) -> RayDataset:
    """
    Preprocess a dataset in parallel.
    
    Args:
        dataset: PyTorch dataset
        preprocess_fn: Preprocessing function
        num_workers: Number of parallel workers
        batch_size: Batch size for processing
        
    Returns:
        Preprocessed RayDataset
    """
    if not ray.is_initialized():
        ray.init()
    
    ds = RayDataset.from_torch(dataset)
    ds = ds.map(preprocess_fn, batch_size=batch_size)
    
    return ds


# =============================================================================
# GPU Preprocessing Actor
# =============================================================================

@ray.remote(num_gpus=1)
class GPUPreprocessActor:
    """Actor for GPU-accelerated preprocessing."""
    
    def __init__(self, preprocess_fn: Callable):
        import torch
        self.device = torch.device("cuda")
        self.preprocess_fn = preprocess_fn
    
    def process_batch(self, batch: List[Any]) -> List[Any]:
        """Process a batch on GPU."""
        return [self.preprocess_fn(item, self.device) for item in batch]


def distributed_gpu_preprocess(
    items: List[Any],
    preprocess_fn: Callable,
    num_gpus: int = 4,
    batch_size: int = 32,
) -> List[Any]:
    """
    Preprocess data using multiple GPUs in parallel.
    
    Args:
        items: Items to preprocess
        preprocess_fn: GPU preprocessing function
        num_gpus: Number of GPUs to use
        batch_size: Batch size per GPU
        
    Returns:
        Preprocessed items
    """
    if not ray.is_initialized():
        ray.init()
    
    # Create GPU actors
    actors = [GPUPreprocessActor.remote(preprocess_fn) for _ in range(num_gpus)]
    
    # Distribute batches across actors
    batches = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]
    
    # Process in parallel
    futures = []
    for i, batch in enumerate(batches):
        actor = actors[i % num_gpus]
        futures.append(actor.process_batch.remote(batch))
    
    # Collect results
    results = []
    for future in futures:
        results.extend(ray.get(future))
    
    return results
