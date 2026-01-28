"""
Overlap Optimizer - Overlap computation with communication.

This module provides tools for optimizing distributed training by
overlapping gradient computation with gradient synchronization.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn


@dataclass
class GradientBucket:
    """Bucket of gradients for batched communication."""
    
    index: int
    size_bytes: int
    parameters: List[nn.Parameter] = field(default_factory=list)
    
    # Flattened gradient buffer
    buffer: Optional[torch.Tensor] = None
    
    # Communication handle
    handle: Optional[Any] = None
    
    # Timing info
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    @property
    def num_parameters(self) -> int:
        return len(self.parameters)
    
    @property
    def is_ready(self) -> bool:
        """Check if all gradients in bucket are computed."""
        return all(p.grad is not None for p in self.parameters)
    
    def flatten_gradients(self) -> torch.Tensor:
        """Flatten all gradients into a single tensor."""
        if not self.is_ready:
            raise RuntimeError("Not all gradients are computed")
        
        grads = [p.grad.view(-1) for p in self.parameters]
        self.buffer = torch.cat(grads)
        return self.buffer
    
    def unflatten_gradients(self) -> None:
        """Copy flattened buffer back to individual gradients."""
        if self.buffer is None:
            return
        
        offset = 0
        for param in self.parameters:
            numel = param.grad.numel()
            param.grad.copy_(self.buffer[offset:offset + numel].view_as(param.grad))
            offset += numel


class OverlapOptimizer:
    """
    Optimizer wrapper that overlaps gradient sync with backward pass.
    
    Implements gradient bucketing and asynchronous all-reduce to hide
    communication latency behind computation.
    
    Example:
        >>> model = DistributedDataParallel(model)
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        >>> overlap_opt = OverlapOptimizer(optimizer, model)
        >>> 
        >>> # Training loop
        >>> loss = model(input).sum()
        >>> overlap_opt.backward(loss)
        >>> overlap_opt.step()
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        model: nn.Module,
        bucket_size_mb: float = 25.0,
        gradient_predivide_factor: Optional[float] = None,
        process_group: Optional[Any] = None,
    ):
        """
        Initialize overlap optimizer.
        
        Args:
            optimizer: PyTorch optimizer
            model: Model to optimize
            bucket_size_mb: Size of gradient buckets in MB
            gradient_predivide_factor: Factor to divide gradients before reduce
            process_group: Process group for distributed ops
        """
        self.optimizer = optimizer
        self.model = model
        self.bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)
        self.gradient_predivide_factor = gradient_predivide_factor
        self.process_group = process_group
        
        # World size for averaging
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        # Build buckets
        self.buckets: List[GradientBucket] = []
        self._build_buckets()
        
        # Parameter to bucket mapping
        self._param_to_bucket: Dict[nn.Parameter, int] = {}
        for i, bucket in enumerate(self.buckets):
            for param in bucket.parameters:
                self._param_to_bucket[param] = i
        
        # Track which buckets have pending communication
        self._pending_buckets: List[int] = []
        
        # Gradient computation tracking
        self._grads_computed: Dict[int, int] = defaultdict(int)
        
        # Install gradient hooks
        self._hooks: List[Any] = []
        self._install_hooks()
    
    def _build_buckets(self) -> None:
        """Build gradient buckets from model parameters."""
        current_bucket_size = 0
        current_params: List[nn.Parameter] = []
        bucket_index = 0
        
        # Iterate parameters in reverse order (typical backward order)
        for param in reversed(list(self.model.parameters())):
            if not param.requires_grad:
                continue
            
            param_size = param.numel() * param.element_size()
            
            # Start new bucket if current is full
            if current_bucket_size + param_size > self.bucket_size_bytes and current_params:
                self.buckets.append(GradientBucket(
                    index=bucket_index,
                    size_bytes=current_bucket_size,
                    parameters=current_params,
                ))
                bucket_index += 1
                current_params = []
                current_bucket_size = 0
            
            current_params.append(param)
            current_bucket_size += param_size
        
        # Add remaining parameters
        if current_params:
            self.buckets.append(GradientBucket(
                index=bucket_index,
                size_bytes=current_bucket_size,
                parameters=current_params,
            ))
        
        # Reverse buckets to match backward order
        self.buckets = list(reversed(self.buckets))
        for i, bucket in enumerate(self.buckets):
            bucket.index = i
    
    def _install_hooks(self) -> None:
        """Install gradient computation hooks."""
        for param in self.model.parameters():
            if not param.requires_grad:
                continue
            
            # Register hook to be called when gradient is computed
            hook = param.register_hook(self._make_hook(param))
            self._hooks.append(hook)
    
    def _make_hook(self, param: nn.Parameter) -> Callable:
        """Create gradient hook for a parameter."""
        bucket_idx = self._param_to_bucket.get(param)
        if bucket_idx is None:
            return lambda grad: grad
        
        bucket = self.buckets[bucket_idx]
        
        def hook(grad: torch.Tensor) -> torch.Tensor:
            # Track gradient computation
            self._grads_computed[bucket_idx] += 1
            
            # Check if bucket is ready
            if self._grads_computed[bucket_idx] == bucket.num_parameters:
                self._start_bucket_reduction(bucket)
            
            return grad
        
        return hook
    
    def _start_bucket_reduction(self, bucket: GradientBucket) -> None:
        """Start asynchronous all-reduce for a bucket."""
        if not dist.is_initialized():
            return
        
        # Flatten gradients
        buffer = bucket.flatten_gradients()
        
        # Pre-divide if configured
        if self.gradient_predivide_factor is not None:
            buffer.div_(self.gradient_predivide_factor)
        
        # Start async all-reduce
        bucket.handle = dist.all_reduce(
            buffer,
            op=dist.ReduceOp.SUM,
            group=self.process_group,
            async_op=True,
        )
        
        self._pending_buckets.append(bucket.index)
    
    def backward(self, loss: torch.Tensor) -> None:
        """
        Perform backward pass with overlapped communication.
        
        Args:
            loss: Loss tensor to backpropagate
        """
        # Reset tracking
        self._grads_computed.clear()
        self._pending_buckets.clear()
        
        # Backward pass - hooks will trigger async all-reduce
        loss.backward()
        
        # Wait for all pending reductions
        self._wait_for_all()
    
    def _wait_for_all(self) -> None:
        """Wait for all pending communications to complete."""
        for bucket_idx in self._pending_buckets:
            bucket = self.buckets[bucket_idx]
            
            if bucket.handle is not None:
                bucket.handle.wait()
                
                # Average gradients
                if self.gradient_predivide_factor is None:
                    bucket.buffer.div_(self.world_size)
                else:
                    factor = self.world_size / self.gradient_predivide_factor
                    bucket.buffer.div_(factor)
                
                # Copy back to individual gradients
                bucket.unflatten_gradients()
        
        self._pending_buckets.clear()
    
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """
        Perform optimizer step.
        
        Args:
            closure: Optional closure for optimizers that require it
            
        Returns:
            Loss value if closure is provided
        """
        # Ensure all communications are complete
        self._wait_for_all()
        
        return self.optimizer.step(closure)
    
    def zero_grad(self, set_to_none: bool = False) -> None:
        """Zero gradients."""
        self.optimizer.zero_grad(set_to_none=set_to_none)


class GradientAccumulator:
    """
    Gradient accumulator for micro-batching with communication overlap.
    
    Accumulates gradients over multiple micro-batches before performing
    a single synchronized all-reduce.
    
    Example:
        >>> accumulator = GradientAccumulator(model, accumulation_steps=4)
        >>> 
        >>> for batch in dataloader:
        ...     loss = model(batch) / 4
        ...     if accumulator.accumulate(loss):
        ...         optimizer.step()
        ...         optimizer.zero_grad()
    """
    
    def __init__(
        self,
        model: nn.Module,
        accumulation_steps: int = 1,
        process_group: Optional[Any] = None,
    ):
        """
        Initialize gradient accumulator.
        
        Args:
            model: Model to accumulate gradients for
            accumulation_steps: Number of steps before sync
            process_group: Process group for distributed ops
        """
        self.model = model
        self.accumulation_steps = accumulation_steps
        self.process_group = process_group
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        self._current_step = 0
    
    def accumulate(self, loss: torch.Tensor) -> bool:
        """
        Accumulate gradients from a loss.
        
        Args:
            loss: Loss tensor (should be pre-divided by accumulation_steps)
            
        Returns:
            True if gradients are ready for optimizer step
        """
        # Backward pass
        loss.backward()
        
        self._current_step += 1
        
        # Check if ready for sync
        if self._current_step >= self.accumulation_steps:
            self._sync_gradients()
            self._current_step = 0
            return True
        
        return False
    
    def _sync_gradients(self) -> None:
        """Synchronize gradients across processes."""
        if not dist.is_initialized() or self.world_size == 1:
            return
        
        # All-reduce all gradients
        for param in self.model.parameters():
            if param.grad is not None:
                dist.all_reduce(
                    param.grad,
                    op=dist.ReduceOp.SUM,
                    group=self.process_group,
                )
                param.grad.div_(self.world_size)
    
    @property
    def should_sync(self) -> bool:
        """Check if next step will trigger sync."""
        return self._current_step + 1 >= self.accumulation_steps


class FusedAllReduce:
    """
    Fused all-reduce for multiple small tensors.
    
    Concatenates multiple tensors into a single buffer for more
    efficient communication.
    """
    
    def __init__(self, process_group: Optional[Any] = None):
        """
        Initialize fused all-reduce.
        
        Args:
            process_group: Process group for distributed ops
        """
        self.process_group = process_group
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
    
    def __call__(
        self,
        tensors: List[torch.Tensor],
        async_op: bool = False,
    ) -> Optional[Any]:
        """
        Perform fused all-reduce.
        
        Args:
            tensors: List of tensors to reduce
            async_op: Whether to perform asynchronously
            
        Returns:
            Handle if async_op=True, else None
        """
        if not dist.is_initialized() or self.world_size == 1:
            return None
        
        # Record original shapes and dtypes
        shapes = [t.shape for t in tensors]
        dtypes = [t.dtype for t in tensors]
        
        # Flatten all tensors
        flat_tensors = [t.view(-1).float() for t in tensors]
        sizes = [t.numel() for t in flat_tensors]
        
        # Concatenate
        buffer = torch.cat(flat_tensors)
        
        # All-reduce
        handle = dist.all_reduce(
            buffer,
            op=dist.ReduceOp.SUM,
            group=self.process_group,
            async_op=async_op,
        )
        
        if async_op:
            return FusedAllReduceHandle(
                handle=handle,
                buffer=buffer,
                tensors=tensors,
                shapes=shapes,
                dtypes=dtypes,
                sizes=sizes,
                world_size=self.world_size,
            )
        
        # Copy back to original tensors
        buffer.div_(self.world_size)
        offset = 0
        for i, tensor in enumerate(tensors):
            tensor.copy_(buffer[offset:offset + sizes[i]].view(shapes[i]).to(dtypes[i]))
            offset += sizes[i]
        
        return None


@dataclass
class FusedAllReduceHandle:
    """Handle for async fused all-reduce."""
    
    handle: Any
    buffer: torch.Tensor
    tensors: List[torch.Tensor]
    shapes: List[torch.Size]
    dtypes: List[torch.dtype]
    sizes: List[int]
    world_size: int
    
    def wait(self) -> None:
        """Wait for operation to complete and copy results."""
        self.handle.wait()
        
        self.buffer.div_(self.world_size)
        offset = 0
        for i, tensor in enumerate(self.tensors):
            tensor.copy_(
                self.buffer[offset:offset + self.sizes[i]]
                .view(self.shapes[i])
                .to(self.dtypes[i])
            )
            offset += self.sizes[i]
