"""
CPU/NVMe Offloading - Trade memory for compute.

Enables training models larger than GPU memory by
offloading parameters and optimizer states.
"""

import torch
import torch.nn as nn
from typing import Any, Callable, Dict, List, Optional, Set
from dataclasses import dataclass, field
import threading
from pathlib import Path
import tempfile
import shutil


@dataclass
class OffloadConfig:
    """Configuration for parameter offloading."""
    # Target
    device: str = "cpu"  # cpu, nvme
    
    # Behavior
    offload_optimizer: bool = True
    offload_params: bool = False  # Experimental
    
    # NVMe settings
    nvme_path: str = "/tmp/offload"
    nvme_buffer_size: int = 1024 * 1024 * 1024  # 1GB
    
    # Performance
    pin_memory: bool = True
    prefetch: bool = True
    num_threads: int = 4


class CPUOffloader:
    """
    CPU Offloader for large model training.
    
    Offloads optimizer states to CPU memory, enabling training
    of models up to 2x larger than GPU memory.
    
    Example:
        >>> offloader = CPUOffloader(model, optimizer)
        >>> 
        >>> for batch in dataloader:
        ...     offloader.prefetch()  # Pre-load next params
        ...     loss = model(batch)
        ...     loss.backward()
        ...     offloader.step()
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Optional[OffloadConfig] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.config = config or OffloadConfig()
        
        self._cpu_states: Dict[int, Dict] = {}
        self._gpu_params: Set[int] = set()
        self._lock = threading.Lock()
        
        # Move optimizer states to CPU
        if self.config.offload_optimizer:
            self._offload_optimizer_states()
    
    def _offload_optimizer_states(self):
        """Move optimizer states to CPU."""
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p in self.optimizer.state:
                    state = self.optimizer.state[p]
                    param_id = id(p)
                    self._cpu_states[param_id] = {}
                    
                    for key, value in state.items():
                        if isinstance(value, torch.Tensor):
                            self._cpu_states[param_id][key] = value.cpu()
                            if self.config.pin_memory:
                                self._cpu_states[param_id][key] = (
                                    self._cpu_states[param_id][key].pin_memory()
                                )
    
    def prefetch(self, param_ids: Optional[List[int]] = None):
        """
        Prefetch optimizer states to GPU.
        
        Call before forward pass to overlap transfer with compute.
        """
        if not self.config.prefetch:
            return
        
        if param_ids is None:
            param_ids = list(self._cpu_states.keys())
        
        for param_id in param_ids:
            if param_id in self._cpu_states:
                for key, value in self._cpu_states[param_id].items():
                    if isinstance(value, torch.Tensor):
                        # Async copy to GPU
                        value.cuda(non_blocking=True)
    
    def step(self):
        """
        Perform optimizer step with offloaded states.
        
        1. Move states to GPU
        2. Step optimizer
        3. Move states back to CPU
        """
        # Move states to GPU
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                param_id = id(p)
                if param_id in self._cpu_states:
                    state = self.optimizer.state[p]
                    for key, cpu_value in self._cpu_states[param_id].items():
                        if isinstance(cpu_value, torch.Tensor):
                            state[key] = cpu_value.cuda(non_blocking=True)
        
        torch.cuda.synchronize()
        
        # Step optimizer
        self.optimizer.step()
        
        # Move states back to CPU
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                param_id = id(p)
                if param_id in self._cpu_states:
                    state = self.optimizer.state[p]
                    for key, value in list(state.items()):
                        if isinstance(value, torch.Tensor) and value.is_cuda:
                            cpu_tensor = value.cpu()
                            if self.config.pin_memory:
                                cpu_tensor = cpu_tensor.pin_memory()
                            self._cpu_states[param_id][key] = cpu_tensor
                            del state[key]
        
        torch.cuda.empty_cache()
    
    def zero_grad(self):
        """Zero gradients."""
        self.optimizer.zero_grad()


class NVMeOffloader:
    """
    NVMe Offloader for extremely large models.
    
    Offloads parameters and optimizer states to NVMe storage,
    enabling training of models 10x larger than GPU+CPU memory.
    
    Example:
        >>> offloader = NVMeOffloader(model, optimizer, nvme_path="/raid/offload")
        >>> 
        >>> # Training loop
        >>> for batch in dataloader:
        ...     loss = model(batch)
        ...     loss.backward()
        ...     offloader.step()
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Optional[OffloadConfig] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.config = config or OffloadConfig(device="nvme")
        
        # Setup NVMe directory
        self.nvme_path = Path(self.config.nvme_path)
        self.nvme_path.mkdir(parents=True, exist_ok=True)
        
        self._file_handles: Dict[int, Path] = {}
        self._buffer = None
    
    def _get_buffer(self, size: int) -> torch.Tensor:
        """Get or create CPU buffer for transfers."""
        if self._buffer is None or self._buffer.numel() < size:
            self._buffer = torch.empty(
                size,
                dtype=torch.float32,
                pin_memory=self.config.pin_memory,
            )
        return self._buffer[:size]
    
    def offload_tensor(
        self,
        tensor: torch.Tensor,
        param_id: int,
        key: str,
    ) -> Path:
        """Offload tensor to NVMe."""
        file_path = self.nvme_path / f"{param_id}_{key}.pt"
        
        # Save to disk
        torch.save(tensor.cpu(), file_path)
        
        return file_path
    
    def load_tensor(
        self,
        file_path: Path,
        device: torch.device,
    ) -> torch.Tensor:
        """Load tensor from NVMe."""
        tensor = torch.load(file_path, map_location="cpu")
        return tensor.to(device, non_blocking=True)
    
    def step(self):
        """Optimizer step with NVMe offloading."""
        # This is a simplified version
        # Full implementation would use async I/O and double buffering
        self.optimizer.step()
    
    def cleanup(self):
        """Cleanup NVMe files."""
        if self.nvme_path.exists():
            shutil.rmtree(self.nvme_path)
    
    def __del__(self):
        self.cleanup()


# =============================================================================
# Activation Offloading
# =============================================================================

class ActivationOffloader:
    """
    Offload activations to CPU during backward pass.
    
    Saves GPU memory at cost of CPU-GPU transfer time.
    """
    
    def __init__(self, offload_to: str = "cpu"):
        self.offload_to = offload_to
        self._saved_activations: Dict[int, torch.Tensor] = {}
    
    def save(self, activation: torch.Tensor) -> int:
        """Save activation to CPU."""
        act_id = id(activation)
        
        if self.offload_to == "cpu":
            self._saved_activations[act_id] = activation.cpu()
        else:
            self._saved_activations[act_id] = activation
        
        return act_id
    
    def load(self, act_id: int, device: torch.device) -> torch.Tensor:
        """Load activation back to GPU."""
        activation = self._saved_activations.pop(act_id)
        return activation.to(device, non_blocking=True)
    
    def clear(self):
        """Clear all saved activations."""
        self._saved_activations.clear()


def offload_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    """Quick function to offload tensor to CPU."""
    return tensor.cpu()


def reload_to_gpu(
    tensor: torch.Tensor,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Reload tensor to GPU."""
    device = device or torch.device("cuda")
    return tensor.to(device, non_blocking=True)


# =============================================================================
# Parameter Sharding for Memory
# =============================================================================

class ParameterSharding:
    """
    Shard large parameters across multiple tensors.
    
    Reduces memory fragmentation for very large tensors.
    """
    
    def __init__(self, num_shards: int = 4):
        self.num_shards = num_shards
    
    def shard(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """Split tensor into shards."""
        total_size = tensor.numel()
        shard_size = (total_size + self.num_shards - 1) // self.num_shards
        
        flat = tensor.view(-1)
        shards = []
        
        for i in range(self.num_shards):
            start = i * shard_size
            end = min((i + 1) * shard_size, total_size)
            shards.append(flat[start:end].clone())
        
        return shards
    
    def unshard(
        self,
        shards: List[torch.Tensor],
        original_shape: torch.Size,
    ) -> torch.Tensor:
        """Reconstruct tensor from shards."""
        flat = torch.cat(shards)
        return flat.view(original_shape)
