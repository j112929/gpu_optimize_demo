#!/usr/bin/env python3
"""
Example: Ray Distributed Computing

Demonstrates Ray for distributed training, tuning, and serving.

Usage:
    python examples/ray_distributed.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset


def demo_cluster():
    """Demonstrate cluster management."""
    from src.ray_distributed.cluster import RayCluster, get_cluster_info
    
    print("\n" + "=" * 60)
    print("RAY CLUSTER DEMO")
    print("=" * 60)
    
    cluster = RayCluster()
    cluster.connect()
    cluster.print_status()


def demo_data_parallel():
    """Demonstrate distributed data processing."""
    from src.ray_distributed.data_parallel import (
        RayDataset, distributed_map, distributed_preprocess
    )
    
    print("\n" + "=" * 60)
    print("DISTRIBUTED DATA PROCESSING DEMO")
    print("=" * 60)
    
    # Create sample data
    data = list(range(1000))
    
    # Distributed map
    print("\n📊 Distributed Map:")
    
    def square(x):
        return x ** 2
    
    results = distributed_map(data[:100], square, batch_size=10)
    print(f"   Input: {data[:5]}...")
    print(f"   Output: {results[:5]}...")
    
    # Ray Dataset
    print("\n📊 Ray Dataset:")
    ds = RayDataset.from_items(data)
    print(f"   Count: {ds.count()}")
    
    # Map operation
    ds_squared = ds.map(lambda x: {"value": x ** 2})
    print(f"   After map: {list(ds_squared.raw.take(3))}")
    
    # Filter
    ds_filtered = ds.filter(lambda x: x > 500)
    print(f"   After filter (>500): {ds_filtered.count()} items")


def demo_trainer():
    """Demonstrate distributed training."""
    from src.ray_distributed.trainer import RayTrainer, RayTrainerConfig
    
    print("\n" + "=" * 60)
    print("DISTRIBUTED TRAINING DEMO")
    print("=" * 60)
    
    # Simple model
    def create_model():
        return nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
    
    # Create dummy dataset
    X = torch.randn(1000, 10)
    y = torch.randint(0, 2, (1000,))
    train_dataset = TensorDataset(X, y)
    
    print("\n📊 Training Configuration:")
    config = RayTrainerConfig(
        num_workers=2,
        use_gpu=torch.cuda.is_available(),
        num_epochs=3,
        batch_size=32,
    )
    print(f"   Workers: {config.num_workers}")
    print(f"   Use GPU: {config.use_gpu}")
    print(f"   Epochs: {config.num_epochs}")
    
    print("\n   [Training would run here with ray.train.TorchTrainer]")
    print("   Features: auto-sharding, fault tolerance, checkpointing")


def demo_tuner():
    """Demonstrate hyperparameter tuning."""
    from src.ray_distributed.tuner import (
        RayTuner, TuneConfig, get_transformer_search_space
    )
    
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING DEMO")
    print("=" * 60)
    
    print("\n📊 Sample Search Spaces:")
    
    # Transformer search space
    transformer_space = get_transformer_search_space()
    print("\n   Transformer hyperparameters:")
    for key, value in list(transformer_space.items())[:4]:
        print(f"     {key}: {type(value).__name__}")
    
    print("\n📊 Tuning Configuration:")
    config = TuneConfig(
        num_samples=10,
        search_alg="optuna",
        scheduler="asha",
        metric="loss",
        mode="min",
    )
    print(f"   Samples: {config.num_samples}")
    print(f"   Search: {config.search_alg}")
    print(f"   Scheduler: {config.scheduler}")
    print(f"   Metric: {config.metric} ({config.mode})")
    
    print("\n   [Tuning would run here with ray.tune]")
    print("   Features: Optuna, ASHA early stopping, PBT")


def demo_serve():
    """Demonstrate model serving."""
    from src.ray_distributed.serve import RayModelServer, ServeConfig
    
    print("\n" + "=" * 60)
    print("MODEL SERVING DEMO")
    print("=" * 60)
    
    # Create simple model
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 2),
    )
    
    print("\n📊 Serve Configuration:")
    config = ServeConfig(
        num_replicas=2,
        max_concurrent_queries=100,
        ray_actor_options={"num_gpus": 0.5},
        autoscaling_config={
            "min_replicas": 1,
            "max_replicas": 10,
            "target_num_ongoing_requests_per_replica": 5,
        },
    )
    print(f"   Replicas: {config.num_replicas}")
    print(f"   Max queries: {config.max_concurrent_queries}")
    print(f"   Auto-scaling: {config.autoscaling_config['min_replicas']}-{config.autoscaling_config['max_replicas']}")
    
    print("\n   [Server would deploy here with ray.serve]")
    print("   Features: auto-scaling, batching, health checks")


def demo_batch_inference():
    """Demonstrate batch inference."""
    print("\n" + "=" * 60)
    print("BATCH INFERENCE DEMO")
    print("=" * 60)
    
    print("\n📊 Batch Inference Setup:")
    print("   - 4 GPU actors")
    print("   - 1000 samples")
    print("   - Batch size: 32")
    print("   - Parallel processing across GPUs")
    
    print("\n   [Batch inference would run here]")
    print("   Features: multi-GPU parallel, actor pooling")


def main():
    print("=" * 60)
    print("RAY DISTRIBUTED COMPUTING DEMO")
    print("=" * 60)
    
    try:
        import ray
        print(f"Ray version: {ray.__version__}")
    except ImportError:
        print("Ray not installed. Install with: pip install 'ray[default]'")
        return
    
    demo_cluster()
    demo_data_parallel()
    demo_trainer()
    demo_tuner()
    demo_serve()
    demo_batch_inference()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
✅ RayDataset: Distributed data loading
✅ RayTrainer: Multi-node GPU training  
✅ RayTuner: Hyperparameter optimization
✅ RayServe: Auto-scaling model serving
✅ Batch Inference: Multi-GPU parallel
    """)
    
    # Cleanup
    import ray
    if ray.is_initialized():
        ray.shutdown()


if __name__ == "__main__":
    main()
