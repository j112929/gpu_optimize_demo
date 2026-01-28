"""
Ray Tuner - Hyperparameter optimization with Ray Tune.

Provides efficient hyperparameter search with multiple algorithms.
"""

import ray
from ray import tune
from ray.tune import TuneConfig as RayTuneConfig
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.hyperopt import HyperOptSearch
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
import torch.nn as nn


@dataclass
class TuneConfig:
    """Configuration for hyperparameter tuning."""
    # Search space
    search_space: Dict[str, Any] = field(default_factory=dict)
    
    # Search algorithm
    search_alg: str = "optuna"  # optuna, hyperopt, random
    
    # Scheduler
    scheduler: str = "asha"  # asha, pbt, fifo
    
    # Resources
    num_samples: int = 10
    max_concurrent_trials: int = 4
    resources_per_trial: Dict[str, float] = field(default_factory=lambda: {"cpu": 2, "gpu": 1})
    
    # Optimization
    metric: str = "loss"
    mode: str = "min"  # min or max
    
    # Early stopping
    grace_period: int = 1
    reduction_factor: int = 4
    
    # Storage
    storage_path: str = "./ray_tune_results"


class RayTuner:
    """
    Ray Tune-based hyperparameter optimizer.
    
    Features:
    - Multiple search algorithms (Optuna, HyperOpt, random)
    - Early stopping with ASHA scheduler
    - Population-based training
    - Automatic checkpointing
    
    Example:
        >>> config = TuneConfig(
        ...     search_space={
        ...         "lr": tune.loguniform(1e-5, 1e-2),
        ...         "batch_size": tune.choice([16, 32, 64]),
        ...     },
        ...     num_samples=20,
        ... )
        >>> tuner = RayTuner(config)
        >>> best_config = tuner.fit(train_fn)
    """
    
    def __init__(self, config: TuneConfig):
        self.config = config
        self._results = None
        
        if not ray.is_initialized():
            ray.init()
    
    def _get_search_alg(self):
        """Get search algorithm."""
        if self.config.search_alg == "optuna":
            return OptunaSearch(metric=self.config.metric, mode=self.config.mode)
        elif self.config.search_alg == "hyperopt":
            return HyperOptSearch(metric=self.config.metric, mode=self.config.mode)
        else:
            return None  # Random search
    
    def _get_scheduler(self):
        """Get trial scheduler."""
        if self.config.scheduler == "asha":
            return ASHAScheduler(
                metric=self.config.metric,
                mode=self.config.mode,
                grace_period=self.config.grace_period,
                reduction_factor=self.config.reduction_factor,
            )
        elif self.config.scheduler == "pbt":
            return PopulationBasedTraining(
                metric=self.config.metric,
                mode=self.config.mode,
            )
        else:
            return None
    
    def fit(
        self,
        train_fn: Callable,
        param_space: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Run hyperparameter search.
        
        Args:
            train_fn: Training function that accepts a config dict
            param_space: Optional override for search space
            
        Returns:
            Best configuration found
        """
        search_space = param_space or self.config.search_space
        
        tuner = tune.Tuner(
            train_fn,
            param_space=search_space,
            tune_config=RayTuneConfig(
                num_samples=self.config.num_samples,
                max_concurrent_trials=self.config.max_concurrent_trials,
                search_alg=self._get_search_alg(),
                scheduler=self._get_scheduler(),
            ),
            run_config=ray.train.RunConfig(
                storage_path=self.config.storage_path,
            ),
        )
        
        self._results = tuner.fit()
        
        return self.best_config
    
    @property
    def best_config(self) -> Dict[str, Any]:
        """Get best configuration."""
        if self._results:
            return self._results.get_best_result().config
        return {}
    
    @property
    def best_result(self):
        """Get best result."""
        if self._results:
            return self._results.get_best_result()
        return None
    
    @property
    def all_results(self):
        """Get all results."""
        return self._results
    
    def get_dataframe(self):
        """Get results as pandas DataFrame."""
        if self._results:
            return self._results.get_dataframe()
        return None


def hyperparameter_search(
    train_fn: Callable,
    search_space: Dict[str, Any],
    num_samples: int = 10,
    metric: str = "loss",
    mode: str = "min",
    use_gpu: bool = True,
) -> Dict[str, Any]:
    """
    Simple API for hyperparameter search.
    
    Args:
        train_fn: Training function
        search_space: Hyperparameter search space
        num_samples: Number of trials
        metric: Metric to optimize
        mode: "min" or "max"
        use_gpu: Whether to use GPU
        
    Returns:
        Best configuration
        
    Example:
        >>> best = hyperparameter_search(
        ...     train_fn=my_train,
        ...     search_space={
        ...         "lr": tune.loguniform(1e-5, 1e-2),
        ...         "hidden_size": tune.choice([128, 256, 512]),
        ...     },
        ...     num_samples=20,
        ... )
    """
    config = TuneConfig(
        search_space=search_space,
        num_samples=num_samples,
        metric=metric,
        mode=mode,
        resources_per_trial={"cpu": 2, "gpu": 1 if use_gpu else 0},
    )
    
    tuner = RayTuner(config)
    return tuner.fit(train_fn)


# =============================================================================
# Common Search Spaces
# =============================================================================

def get_transformer_search_space() -> Dict[str, Any]:
    """Get common transformer hyperparameters."""
    return {
        "learning_rate": tune.loguniform(1e-5, 1e-3),
        "batch_size": tune.choice([8, 16, 32, 64]),
        "num_layers": tune.randint(4, 12),
        "hidden_size": tune.choice([256, 512, 768, 1024]),
        "num_heads": tune.choice([4, 8, 12, 16]),
        "dropout": tune.uniform(0.0, 0.3),
        "warmup_steps": tune.randint(100, 1000),
        "weight_decay": tune.loguniform(1e-5, 1e-2),
    }


def get_cnn_search_space() -> Dict[str, Any]:
    """Get common CNN hyperparameters."""
    return {
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "num_filters": tune.choice([32, 64, 128]),
        "kernel_size": tune.choice([3, 5, 7]),
        "dropout": tune.uniform(0.0, 0.5),
        "optimizer": tune.choice(["adam", "sgd", "adamw"]),
        "momentum": tune.uniform(0.8, 0.99),
    }
