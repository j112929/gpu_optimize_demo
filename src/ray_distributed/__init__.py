"""
Ray Distributed Module - Scalable distributed computing utilities.

This module provides Ray-based distributed computing capabilities:
- Distributed data preprocessing
- Multi-node training orchestration
- Hyperparameter tuning with Ray Tune
- Model serving with Ray Serve
"""

from src.ray_distributed.data_parallel import (
    RayDataLoader,
    RayDataset,
    distributed_map,
    distributed_preprocess,
)
from src.ray_distributed.trainer import (
    RayTrainer,
    RayTrainerConfig,
    distributed_train,
)
from src.ray_distributed.tuner import (
    RayTuner,
    TuneConfig,
    hyperparameter_search,
)
from src.ray_distributed.serve import (
    RayModelServer,
    deploy_model,
    batch_inference,
)
from src.ray_distributed.cluster import (
    RayCluster,
    get_cluster_info,
    scale_cluster,
)

__all__ = [
    # Data Parallel
    "RayDataLoader",
    "RayDataset",
    "distributed_map",
    "distributed_preprocess",
    # Trainer
    "RayTrainer",
    "RayTrainerConfig",
    "distributed_train",
    # Tuner
    "RayTuner",
    "TuneConfig",
    "hyperparameter_search",
    # Serve
    "RayModelServer",
    "deploy_model",
    "batch_inference",
    # Cluster
    "RayCluster",
    "get_cluster_info",
    "scale_cluster",
]
