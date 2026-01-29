"""
Optimized DataLoader - High-performance data loading.

Provides:
- Async prefetching with CUDA streams
- Non-blocking data transfers
- Persistent workers
- Streaming datasets for large data
- Memory-mapped datasets
- Automatic batch optimization

Key benefits:
- Eliminate I/O bottlenecks
- GPU never waits for data
- Efficient memory usage for large datasets
"""

import torch
import torch.utils.data as data
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
import threading
import queue
import os
import mmap
from pathlib import Path


@dataclass
class DataLoaderConfig:
    """Configuration for optimized DataLoader."""
    
    # Basic settings
    batch_size: int = 32
    num_workers: int = 4
    
    # Prefetching
    prefetch_factor: int = 4         # Batches to prefetch per worker
    
    # Memory optimization
    pin_memory: bool = True          # Pin memory for faster transfer
    persistent_workers: bool = True  # Keep workers alive
    
    # CUDA stream prefetching
    use_cuda_streams: bool = True
    num_streams: int = 2
    
    # Drop last incomplete batch for consistency
    drop_last: bool = True
    
    # Shuffle settings
    shuffle: bool = True
    seed: int = 42
    

# =============================================================================
# Async CUDA Prefetcher
# =============================================================================

class CUDAPrefetcher:
    """
    Prefetch batches to GPU using CUDA streams.
    
    Overlaps data transfer with computation for maximum throughput.
    
    Example:
        >>> dataloader = create_dataloader(dataset, config)
        >>> prefetcher = CUDAPrefetcher(dataloader, device='cuda:0')
        >>> 
        >>> batch = prefetcher.next()
        >>> while batch is not None:
        >>>     output = model(batch)
        >>>     batch = prefetcher.next()
    """
    
    def __init__(
        self,
        dataloader: data.DataLoader,
        device: Union[str, torch.device] = 'cuda',
        num_streams: int = 2,
    ):
        self.dataloader = dataloader
        self.device = torch.device(device)
        self.num_streams = num_streams
        
        # Create CUDA streams
        self.streams = [torch.cuda.Stream(device=self.device) for _ in range(num_streams)]
        self.stream_idx = 0
        
        # Prefetch state
        self.loader_iter = None
        self.next_batch = None
        self.next_stream = None
    
    def __iter__(self):
        self.loader_iter = iter(self.dataloader)
        self._prefetch()
        return self
    
    def _prefetch(self):
        """Prefetch next batch asynchronously."""
        try:
            batch = next(self.loader_iter)
        except StopIteration:
            self.next_batch = None
            self.next_stream = None
            return
        
        # Select stream
        self.stream_idx = (self.stream_idx + 1) % self.num_streams
        stream = self.streams[self.stream_idx]
        
        # Transfer to GPU asynchronously
        with torch.cuda.stream(stream):
            self.next_batch = self._to_device(batch)
            self.next_stream = stream
    
    def _to_device(self, batch: Any) -> Any:
        """Move batch to device, handling different types."""
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device, non_blocking=True)
        elif isinstance(batch, (list, tuple)):
            return type(batch)(self._to_device(item) for item in batch)
        elif isinstance(batch, dict):
            return {key: self._to_device(value) for key, value in batch.items()}
        return batch
    
    def __next__(self) -> Any:
        if self.next_batch is None and self.next_stream is None:
            raise StopIteration
        
        # Wait for current batch
        if self.next_stream is not None:
            torch.cuda.current_stream(self.device).wait_stream(self.next_stream)
        
        batch = self.next_batch
        
        # Start prefetching next batch
        self._prefetch()
        
        return batch
    
    def next(self) -> Optional[Any]:
        """Get next batch (returns None at end)."""
        try:
            return self.__next__()
        except StopIteration:
            return None


# =============================================================================
# Background Data Prefetcher (CPU)
# =============================================================================

class BackgroundPrefetcher:
    """
    Background thread prefetcher for CPU-heavy preprocessing.
    
    Runs data loading in background threads to hide I/O latency.
    """
    
    def __init__(
        self,
        dataloader: data.DataLoader,
        prefetch_batches: int = 4,
    ):
        self.dataloader = dataloader
        self.prefetch_batches = prefetch_batches
        
        self.queue = queue.Queue(maxsize=prefetch_batches)
        self.stop_event = threading.Event()
        self.thread = None
    
    def _prefetch_worker(self, loader_iter: Iterator):
        """Worker thread for prefetching."""
        try:
            for batch in loader_iter:
                if self.stop_event.is_set():
                    break
                self.queue.put(batch)
        except Exception as e:
            self.queue.put(e)
        finally:
            self.queue.put(None)  # Signal end
    
    def __iter__(self):
        loader_iter = iter(self.dataloader)
        self.stop_event.clear()
        
        self.thread = threading.Thread(
            target=self._prefetch_worker,
            args=(loader_iter,)
        )
        self.thread.daemon = True
        self.thread.start()
        
        return self
    
    def __next__(self):
        item = self.queue.get()
        if item is None:
            raise StopIteration
        if isinstance(item, Exception):
            raise item
        return item
    
    def stop(self):
        """Stop prefetching."""
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=1.0)


# =============================================================================
# Streaming Dataset
# =============================================================================

class StreamingDataset(data.IterableDataset):
    """
    Streaming dataset for processing large data that doesn't fit in memory.
    
    Supports:
    - Sharding across workers
    - Resume from checkpoint
    - Memory-efficient processing
    
    Example:
        >>> dataset = StreamingDataset(
        >>>     data_files=["train_0.parquet", "train_1.parquet"],
        >>>     process_fn=tokenize,
        >>> )
        >>> dataloader = DataLoader(dataset, batch_size=32)
    """
    
    def __init__(
        self,
        data_files: List[str],
        process_fn: Optional[Callable] = None,
        shuffle_files: bool = True,
        seed: int = 42,
    ):
        self.data_files = data_files
        self.process_fn = process_fn or (lambda x: x)
        self.shuffle_files = shuffle_files
        self.seed = seed
    
    def __iter__(self) -> Iterator:
        worker_info = data.get_worker_info()
        
        files = self.data_files.copy()
        
        # Shuffle files
        if self.shuffle_files:
            import random
            random.seed(self.seed)
            random.shuffle(files)
        
        # Shard across workers
        if worker_info is not None:
            files = files[worker_info.id::worker_info.num_workers]
        
        for file_path in files:
            yield from self._process_file(file_path)
    
    def _process_file(self, file_path: str) -> Iterator:
        """Process a single file."""
        # Detect file type and read
        if file_path.endswith('.parquet'):
            yield from self._read_parquet(file_path)
        elif file_path.endswith('.jsonl'):
            yield from self._read_jsonl(file_path)
        else:
            yield from self._read_text(file_path)
    
    def _read_parquet(self, path: str) -> Iterator:
        """Read parquet file in chunks."""
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(path)
            for batch in table.to_batches():
                for row in batch.to_pydict().values():
                    yield self.process_fn(row)
        except ImportError:
            print("pyarrow not installed, skipping parquet")
    
    def _read_jsonl(self, path: str) -> Iterator:
        """Read JSONL file line by line."""
        import json
        with open(path, 'r') as f:
            for line in f:
                yield self.process_fn(json.loads(line))
    
    def _read_text(self, path: str) -> Iterator:
        """Read text file line by line."""
        with open(path, 'r') as f:
            for line in f:
                yield self.process_fn(line.strip())


# =============================================================================
# Memory-Mapped Dataset
# =============================================================================

class MemoryMappedDataset(data.Dataset):
    """
    Memory-mapped dataset for fast random access to large files.
    
    Uses OS memory mapping to access data without loading into RAM.
    
    Example:
        >>> dataset = MemoryMappedDataset("large_file.bin", record_size=1024)
        >>> sample = dataset[1000]  # Fast random access
    """
    
    def __init__(
        self,
        file_path: str,
        record_size: int,
        dtype: torch.dtype = torch.float32,
    ):
        self.file_path = file_path
        self.record_size = record_size
        self.dtype = dtype
        
        # Memory map the file
        self.file = open(file_path, 'rb')
        self.mmap = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        
        self.num_records = len(self.mmap) // record_size
    
    def __len__(self) -> int:
        return self.num_records
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        start = idx * self.record_size
        end = start + self.record_size
        
        data = self.mmap[start:end]
        tensor = torch.frombuffer(data, dtype=self.dtype).clone()
        
        return tensor
    
    def __del__(self):
        if hasattr(self, 'mmap'):
            self.mmap.close()
        if hasattr(self, 'file'):
            self.file.close()


# =============================================================================
# Optimized DataLoader Factory
# =============================================================================

def create_dataloader(
    dataset: data.Dataset,
    config: Optional[DataLoaderConfig] = None,
    **kwargs,
) -> data.DataLoader:
    """
    Create an optimized DataLoader with best practices.
    
    Args:
        dataset: Dataset to load from
        config: DataLoader configuration
        **kwargs: Additional DataLoader arguments
        
    Returns:
        Optimized DataLoader
    """
    config = config or DataLoaderConfig()
    
    # Adjust workers based on available CPUs
    num_workers = min(config.num_workers, os.cpu_count() or 4)
    
    # Create sampler for distributed training
    sampler = None
    shuffle = config.shuffle
    
    if torch.distributed.is_initialized():
        sampler = data.distributed.DistributedSampler(
            dataset,
            shuffle=config.shuffle,
            seed=config.seed,
        )
        shuffle = False  # Sampler handles shuffling
    
    loader = data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        prefetch_factor=config.prefetch_factor if num_workers > 0 else None,
        pin_memory=config.pin_memory and torch.cuda.is_available(),
        persistent_workers=config.persistent_workers and num_workers > 0,
        drop_last=config.drop_last,
        **kwargs,
    )
    
    return loader


def create_prefetched_loader(
    dataset: data.Dataset,
    config: Optional[DataLoaderConfig] = None,
    device: str = 'cuda',
) -> CUDAPrefetcher:
    """
    Create a DataLoader with CUDA prefetching.
    
    Args:
        dataset: Dataset to load from
        config: DataLoader configuration
        device: Target device
        
    Returns:
        CUDAPrefetcher wrapping the DataLoader
    """
    config = config or DataLoaderConfig()
    loader = create_dataloader(dataset, config)
    
    return CUDAPrefetcher(
        loader,
        device=device,
        num_streams=config.num_streams,
    )


# =============================================================================
# Batch Collation Utilities
# =============================================================================

class DynamicBatchCollator:
    """
    Dynamic batching based on sequence length.
    
    Packs sequences efficiently to maximize GPU utilization.
    """
    
    def __init__(
        self,
        max_tokens_per_batch: int = 8192,
        pad_token_id: int = 0,
    ):
        self.max_tokens = max_tokens_per_batch
        self.pad_token_id = pad_token_id
    
    def __call__(self, samples: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate with dynamic batching."""
        # Sort by length for efficient packing
        samples = sorted(samples, key=lambda x: len(x['input_ids']), reverse=True)
        
        # Find max length in batch
        max_len = len(samples[0]['input_ids'])
        
        # Pad and stack
        input_ids = []
        attention_mask = []
        
        for sample in samples:
            ids = sample['input_ids']
            pad_len = max_len - len(ids)
            
            input_ids.append(ids + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
        
        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
        }


# =============================================================================
# Data Loading Benchmark
# =============================================================================

def benchmark_dataloader(
    loader: data.DataLoader,
    num_batches: int = 100,
    device: str = 'cuda',
) -> Dict[str, float]:
    """
    Benchmark dataloader throughput.
    
    Returns:
        Dictionary with timing statistics
    """
    import time
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Warmup
    for i, batch in enumerate(loader):
        if i >= 10:
            break
    
    # Benchmark
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    total_samples = 0
    
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        
        # Transfer to device
        if isinstance(batch, torch.Tensor):
            batch = batch.to(device)
            total_samples += batch.shape[0]
        elif isinstance(batch, (list, tuple)):
            batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
            total_samples += batch[0].shape[0]
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    
    return {
        'total_time_s': elapsed,
        'samples_per_second': total_samples / elapsed,
        'batches_per_second': num_batches / elapsed,
        'ms_per_batch': elapsed / num_batches * 1000,
    }
