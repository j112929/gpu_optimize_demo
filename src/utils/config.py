"""
Configuration management for GPU optimization tools.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class ProfilerConfig:
    """Profiler configuration."""
    output_dir: str = "./profiler_traces"
    profile_cuda: bool = True
    profile_cpu: bool = True
    profile_memory: bool = True
    export_chrome_trace: bool = True
    export_tensorboard: bool = True


@dataclass
class DataLoaderConfig:
    """DataLoader configuration."""
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True


@dataclass
class DistributedConfig:
    """Distributed training configuration."""
    backend: str = "nccl"
    world_size: int = 1
    local_rank: int = 0
    bucket_size_mb: float = 25.0
    gradient_predivide_factor: Optional[float] = None


@dataclass
class Config:
    """Main configuration container."""
    
    # Sub-configs
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    
    # General settings
    seed: int = 42
    debug: bool = False
    device: str = "cuda"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create config from dictionary."""
        config = cls()
        
        if "profiler" in data:
            config.profiler = ProfilerConfig(**data["profiler"])
        
        if "dataloader" in data:
            config.dataloader = DataLoaderConfig(**data["dataloader"])
        
        if "distributed" in data:
            config.distributed = DistributedConfig(**data["distributed"])
        
        for key in ["seed", "debug", "device"]:
            if key in data:
                setattr(config, key, data[key])
        
        return config
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load config from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data or {})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "profiler": {
                "output_dir": self.profiler.output_dir,
                "profile_cuda": self.profiler.profile_cuda,
                "profile_cpu": self.profiler.profile_cpu,
                "profile_memory": self.profiler.profile_memory,
                "export_chrome_trace": self.profiler.export_chrome_trace,
                "export_tensorboard": self.profiler.export_tensorboard,
            },
            "dataloader": {
                "batch_size": self.dataloader.batch_size,
                "num_workers": self.dataloader.num_workers,
                "pin_memory": self.dataloader.pin_memory,
                "prefetch_factor": self.dataloader.prefetch_factor,
                "persistent_workers": self.dataloader.persistent_workers,
            },
            "distributed": {
                "backend": self.distributed.backend,
                "world_size": self.distributed.world_size,
                "local_rank": self.distributed.local_rank,
                "bucket_size_mb": self.distributed.bucket_size_mb,
            },
            "seed": self.seed,
            "debug": self.debug,
            "device": self.device,
        }
    
    def save(self, path: str) -> None:
        """Save config to YAML file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


def load_config(path: Optional[str] = None) -> Config:
    """
    Load configuration from file or return defaults.
    
    Args:
        path: Optional path to YAML config file
        
    Returns:
        Config object
    """
    if path and Path(path).exists():
        return Config.from_yaml(path)
    return Config()
