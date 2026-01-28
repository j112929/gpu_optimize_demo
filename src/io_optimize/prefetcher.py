"""
GPU Prefetcher - Overlap data loading with GPU computation.

This module provides prefetching utilities that transfer the next batch
to GPU while the current batch is being processed, hiding data transfer latency.
"""

from typing import Any, Iterator, Optional, Tuple

import torch
from torch.utils.data import DataLoader


class GPUPrefetcher:
    """
    Prefetch data to GPU to overlap data loading with computation.
    
    This class wraps a DataLoader and prefetches the next batch to GPU
    using a separate CUDA stream, effectively hiding the data transfer
    latency behind computation.
    
    Example:
        >>> loader = DataLoader(dataset, batch_size=32)
        >>> prefetcher = GPUPrefetcher(loader)
        >>> 
        >>> for inputs, targets in prefetcher:
        ...     outputs = model(inputs)
        ...     loss = criterion(outputs, targets)
    """
    
    def __init__(
        self,
        loader: DataLoader,
        device: Optional[torch.device] = None,
        fp16: bool = False,
    ):
        """
        Initialize GPU prefetcher.
        
        Args:
            loader: PyTorch DataLoader
            device: Target GPU device (defaults to current device)
            fp16: Convert data to float16 for mixed precision
        """
        self.loader = loader
        self.device = device or torch.device('cuda')
        self.fp16 = fp16
        
        # CUDA stream for async data transfer
        self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        
        self._iterator: Optional[Iterator] = None
        self._next_batch: Optional[Tuple] = None
        self._stop_iteration = False
    
    def __iter__(self) -> "GPUPrefetcher":
        """Start iteration and prefetch first batch."""
        self._iterator = iter(self.loader)
        self._stop_iteration = False
        self._prefetch()
        return self
    
    def __next__(self) -> Tuple[torch.Tensor, ...]:
        """Get next batch from prefetch buffer."""
        if self._stop_iteration:
            raise StopIteration
        
        # Wait for prefetch to complete
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        
        # Get prefetched batch
        batch = self._next_batch
        
        if batch is None:
            raise StopIteration
        
        # Record that the batch is being used
        if self.stream is not None:
            for item in batch:
                if isinstance(item, torch.Tensor) and item.is_cuda:
                    item.record_stream(torch.cuda.current_stream())
        
        # Prefetch next batch
        self._prefetch()
        
        return batch
    
    def __len__(self) -> int:
        """Return number of batches."""
        return len(self.loader)
    
    def _prefetch(self) -> None:
        """Prefetch next batch asynchronously."""
        try:
            batch = next(self._iterator)
        except StopIteration:
            self._next_batch = None
            self._stop_iteration = True
            return
        
        # Transfer to GPU using separate stream
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                self._next_batch = self._to_cuda(batch)
        else:
            self._next_batch = self._to_cuda(batch)
    
    def _to_cuda(self, batch: Any) -> Any:
        """Transfer batch to GPU with optional FP16 conversion."""
        if isinstance(batch, torch.Tensor):
            tensor = batch.to(self.device, non_blocking=True)
            if self.fp16 and tensor.dtype == torch.float32:
                tensor = tensor.half()
            return tensor
        
        elif isinstance(batch, (list, tuple)):
            converted = [self._to_cuda(item) for item in batch]
            return type(batch)(converted)
        
        elif isinstance(batch, dict):
            return {key: self._to_cuda(value) for key, value in batch.items()}
        
        else:
            return batch


class CUDAPrefetcher:
    """
    Advanced CUDA prefetcher with mean/std normalization.
    
    Performs on-GPU normalization during prefetch, saving CPU cycles
    and memory bandwidth.
    
    Example:
        >>> mean = torch.tensor([0.485, 0.456, 0.406])
        >>> std = torch.tensor([0.229, 0.224, 0.225])
        >>> prefetcher = CUDAPrefetcher(loader, mean=mean, std=std)
        >>> 
        >>> for inputs, targets in prefetcher:
        ...     outputs = model(inputs)  # inputs already normalized
    """
    
    def __init__(
        self,
        loader: DataLoader,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        fp16: bool = False,
    ):
        """
        Initialize CUDA prefetcher.
        
        Args:
            loader: PyTorch DataLoader
            mean: Normalization mean (per channel)
            std: Normalization std (per channel)
            device: Target GPU device
            fp16: Use float16 precision
        """
        self.loader = loader
        self.device = device or torch.device('cuda')
        self.fp16 = fp16
        
        # Move normalization parameters to GPU
        self.mean = None
        self.std = None
        
        if mean is not None:
            self.mean = mean.view(1, -1, 1, 1).to(self.device)
            if fp16:
                self.mean = self.mean.half()
        
        if std is not None:
            self.std = std.view(1, -1, 1, 1).to(self.device)
            if fp16:
                self.std = self.std.half()
        
        # CUDA stream for async operations
        self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        
        self._iterator: Optional[Iterator] = None
        self._next_input: Optional[torch.Tensor] = None
        self._next_target: Optional[torch.Tensor] = None
        self._stop_iteration = False
    
    def __iter__(self) -> "CUDAPrefetcher":
        """Start iteration."""
        self._iterator = iter(self.loader)
        self._stop_iteration = False
        self._prefetch()
        return self
    
    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get next preprocessed batch."""
        if self._stop_iteration:
            raise StopIteration
        
        # Wait for prefetch
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        
        input_data = self._next_input
        target = self._next_target
        
        if input_data is None:
            raise StopIteration
        
        # Record stream usage
        if self.stream is not None:
            if input_data is not None and input_data.is_cuda:
                input_data.record_stream(torch.cuda.current_stream())
            if target is not None and isinstance(target, torch.Tensor) and target.is_cuda:
                target.record_stream(torch.cuda.current_stream())
        
        # Prefetch next
        self._prefetch()
        
        return input_data, target
    
    def __len__(self) -> int:
        """Return number of batches."""
        return len(self.loader)
    
    def _prefetch(self) -> None:
        """Prefetch and preprocess next batch."""
        try:
            input_data, target = next(self._iterator)
        except StopIteration:
            self._next_input = None
            self._next_target = None
            self._stop_iteration = True
            return
        
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                self._process_batch(input_data, target)
        else:
            self._process_batch(input_data, target)
    
    def _process_batch(
        self,
        input_data: torch.Tensor,
        target: torch.Tensor
    ) -> None:
        """Transfer and preprocess batch."""
        # Transfer to GPU
        self._next_input = input_data.to(self.device, non_blocking=True)
        self._next_target = target.to(self.device, non_blocking=True)
        
        # Convert to float and normalize
        if self._next_input.dtype == torch.uint8:
            self._next_input = self._next_input.float().div_(255.0)
        
        # Apply FP16 conversion
        if self.fp16:
            self._next_input = self._next_input.half()
        
        # Apply normalization
        if self.mean is not None:
            self._next_input = self._next_input.sub_(self.mean)
        if self.std is not None:
            self._next_input = self._next_input.div_(self.std)


class AsyncPrefetcher:
    """
    Multi-buffer async prefetcher for maximum throughput.
    
    Uses multiple prefetch buffers to ensure data is always ready.
    """
    
    def __init__(
        self,
        loader: DataLoader,
        device: Optional[torch.device] = None,
        num_buffers: int = 2,
    ):
        """
        Initialize async prefetcher.
        
        Args:
            loader: PyTorch DataLoader
            device: Target GPU device
            num_buffers: Number of prefetch buffers
        """
        self.loader = loader
        self.device = device or torch.device('cuda')
        self.num_buffers = num_buffers
        
        # Create streams for each buffer
        self.streams = [
            torch.cuda.Stream() if torch.cuda.is_available() else None
            for _ in range(num_buffers)
        ]
        
        self._buffers: list = [None] * num_buffers
        self._iterator: Optional[Iterator] = None
        self._current_buffer = 0
        self._exhausted = False
    
    def __iter__(self) -> "AsyncPrefetcher":
        """Start iteration and fill buffers."""
        self._iterator = iter(self.loader)
        self._current_buffer = 0
        self._exhausted = False
        
        # Fill all buffers
        for i in range(self.num_buffers):
            self._prefetch_to_buffer(i)
        
        return self
    
    def __next__(self) -> Any:
        """Get next batch."""
        if self._exhausted and self._buffers[self._current_buffer] is None:
            raise StopIteration
        
        # Wait for current buffer
        if self.streams[self._current_buffer] is not None:
            torch.cuda.current_stream().wait_stream(self.streams[self._current_buffer])
        
        batch = self._buffers[self._current_buffer]
        
        if batch is None:
            raise StopIteration
        
        # Start prefetching into this buffer
        self._prefetch_to_buffer(self._current_buffer)
        
        # Move to next buffer
        self._current_buffer = (self._current_buffer + 1) % self.num_buffers
        
        return batch
    
    def __len__(self) -> int:
        """Return number of batches."""
        return len(self.loader)
    
    def _prefetch_to_buffer(self, buffer_idx: int) -> None:
        """Prefetch next batch to specified buffer."""
        if self._exhausted:
            self._buffers[buffer_idx] = None
            return
        
        try:
            batch = next(self._iterator)
        except StopIteration:
            self._exhausted = True
            self._buffers[buffer_idx] = None
            return
        
        stream = self.streams[buffer_idx]
        if stream is not None:
            with torch.cuda.stream(stream):
                self._buffers[buffer_idx] = self._to_cuda(batch)
        else:
            self._buffers[buffer_idx] = self._to_cuda(batch)
    
    def _to_cuda(self, batch: Any) -> Any:
        """Transfer batch to GPU."""
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device, non_blocking=True)
        elif isinstance(batch, (list, tuple)):
            return type(batch)([self._to_cuda(item) for item in batch])
        elif isinstance(batch, dict):
            return {key: self._to_cuda(value) for key, value in batch.items()}
        return batch
