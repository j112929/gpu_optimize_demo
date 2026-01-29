"""
Model Merging - Combine multiple fine-tuned models.

Provides:
- Linear interpolation merging
- SLERP (Spherical Linear Interpolation)
- TIES-Merging (Task Interpolation for Expert Selection)
- DARE (Drop And REscale)
- Model Soup (averaging multiple checkpoints)

These techniques allow combining capabilities from different fine-tuned models.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import copy


@dataclass
class MergeConfig:
    """Configuration for model merging."""
    
    method: str = "linear"           # "linear", "slerp", "ties", "dare", "soup"
    weights: Optional[List[float]] = None  # Weights for each model
    
    # TIES parameters
    ties_threshold: float = 0.2      # Threshold for pruning
    ties_sign_consensus: bool = True # Use sign consensus
    
    # DARE parameters
    dare_drop_rate: float = 0.9      # Fraction of deltas to drop
    dare_rescale: bool = True        # Rescale remaining deltas
    
    # SLERP parameters
    slerp_t: float = 0.5             # Interpolation factor
    

class ModelMerger:
    """
    Merge multiple fine-tuned models into a single model.
    
    Example:
        >>> merger = ModelMerger(base_model)
        >>> merged = merger.merge_models(
        >>>     [model_a, model_b, model_c],
        >>>     config=MergeConfig(method="ties", weights=[0.4, 0.3, 0.3]),
        >>> )
    """
    
    def __init__(self, base_model: Optional[nn.Module] = None):
        """
        Initialize merger.
        
        Args:
            base_model: Optional base/reference model for computing deltas
        """
        self.base_model = base_model
    
    def merge_models(
        self,
        models: List[nn.Module],
        config: Optional[MergeConfig] = None,
    ) -> nn.Module:
        """
        Merge multiple models using specified method.
        
        Args:
            models: List of models to merge
            config: Merge configuration
            
        Returns:
            Merged model
        """
        config = config or MergeConfig()
        
        if config.method == "linear":
            return self.linear_merge(models, config.weights)
        elif config.method == "slerp":
            if len(models) != 2:
                raise ValueError("SLERP requires exactly 2 models")
            return self.slerp_merge(models[0], models[1], config.slerp_t)
        elif config.method == "ties":
            return self.ties_merge(
                models, 
                config.weights, 
                config.ties_threshold,
                config.ties_sign_consensus,
            )
        elif config.method == "dare":
            return self.dare_merge(
                models,
                config.weights,
                config.dare_drop_rate,
                config.dare_rescale,
            )
        elif config.method == "soup":
            return self.model_soup(models)
        else:
            raise ValueError(f"Unknown merge method: {config.method}")
    
    def linear_merge(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
    ) -> nn.Module:
        """
        Linear interpolation of model weights.
        
        merged = w1 * model1 + w2 * model2 + ...
        
        Args:
            models: List of models
            weights: Weight for each model (defaults to equal)
            
        Returns:
            Merged model
        """
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        assert len(weights) == len(models), "Weights must match number of models"
        assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1"
        
        # Create output model
        merged = copy.deepcopy(models[0])
        merged_state = merged.state_dict()
        
        # Zero out
        for key in merged_state:
            merged_state[key] = torch.zeros_like(merged_state[key])
        
        # Weighted sum
        for model, weight in zip(models, weights):
            state = model.state_dict()
            for key in merged_state:
                merged_state[key] += weight * state[key].float()
        
        # Convert back to original dtype
        for key in merged_state:
            merged_state[key] = merged_state[key].to(models[0].state_dict()[key].dtype)
        
        merged.load_state_dict(merged_state)
        return merged
    
    def slerp_merge(
        self,
        model_a: nn.Module,
        model_b: nn.Module,
        t: float = 0.5,
    ) -> nn.Module:
        """
        Spherical Linear Interpolation (SLERP) of model weights.
        
        Better than linear interpolation for maintaining magnitude.
        
        Args:
            model_a: First model
            model_b: Second model
            t: Interpolation factor (0 = model_a, 1 = model_b)
            
        Returns:
            Merged model
        """
        merged = copy.deepcopy(model_a)
        
        state_a = model_a.state_dict()
        state_b = model_b.state_dict()
        merged_state = merged.state_dict()
        
        for key in merged_state:
            a = state_a[key].float().flatten()
            b = state_b[key].float().flatten()
            
            # Compute angle between vectors
            dot = torch.clamp(torch.dot(a, b) / (a.norm() * b.norm() + 1e-10), -1, 1)
            omega = torch.acos(dot)
            
            if omega.abs() < 1e-10:
                # Vectors are parallel, use linear interpolation
                merged_state[key] = ((1 - t) * a + t * b).view(state_a[key].shape)
            else:
                # SLERP
                sin_omega = torch.sin(omega)
                merged_state[key] = (
                    torch.sin((1 - t) * omega) / sin_omega * a +
                    torch.sin(t * omega) / sin_omega * b
                ).view(state_a[key].shape)
            
            merged_state[key] = merged_state[key].to(state_a[key].dtype)
        
        merged.load_state_dict(merged_state)
        return merged
    
    def ties_merge(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
        threshold: float = 0.2,
        sign_consensus: bool = True,
    ) -> nn.Module:
        """
        TIES-Merging: Task Interpolation for Expert Selection.
        
        1. Compute task vectors (deltas from base)
        2. Prune small values
        3. Resolve sign conflicts
        4. Merge
        
        Paper: https://arxiv.org/abs/2306.01708
        
        Args:
            models: List of fine-tuned models
            weights: Weight for each model
            threshold: Pruning threshold (fraction of values to keep)
            sign_consensus: Whether to use sign consensus
            
        Returns:
            Merged model
        """
        if self.base_model is None:
            raise ValueError("TIES requires a base model")
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        merged = copy.deepcopy(self.base_model)
        base_state = self.base_model.state_dict()
        merged_state = merged.state_dict()
        
        for key in merged_state:
            # 1. Compute task vectors (deltas)
            deltas = []
            for model in models:
                delta = model.state_dict()[key].float() - base_state[key].float()
                deltas.append(delta)
            
            # 2. Prune small values (keep only top-k by magnitude)
            pruned_deltas = []
            for delta in deltas:
                flat = delta.flatten()
                k = max(1, int(len(flat) * (1 - threshold)))
                threshold_val = torch.topk(flat.abs(), k).values[-1]
                pruned = torch.where(
                    delta.abs() >= threshold_val,
                    delta,
                    torch.zeros_like(delta),
                )
                pruned_deltas.append(pruned)
            
            # 3. Sign consensus
            if sign_consensus and len(pruned_deltas) > 1:
                # Compute majority sign
                signs = torch.stack([torch.sign(d) for d in pruned_deltas])
                sign_sum = signs.sum(dim=0)
                majority_sign = torch.sign(sign_sum)
                
                # Mask out values that disagree with majority
                final_deltas = []
                for delta in pruned_deltas:
                    agree = torch.sign(delta) == majority_sign
                    final_deltas.append(torch.where(agree, delta, torch.zeros_like(delta)))
            else:
                final_deltas = pruned_deltas
            
            # 4. Weighted merge
            merged_delta = torch.zeros_like(merged_state[key], dtype=torch.float32)
            for delta, weight in zip(final_deltas, weights):
                merged_delta += weight * delta
            
            merged_state[key] = (base_state[key].float() + merged_delta).to(base_state[key].dtype)
        
        merged.load_state_dict(merged_state)
        return merged
    
    def dare_merge(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
        drop_rate: float = 0.9,
        rescale: bool = True,
    ) -> nn.Module:
        """
        DARE: Drop And REscale merging.
        
        Randomly drops most delta values and rescales the rest.
        Surprisingly effective at combining model capabilities.
        
        Paper: https://arxiv.org/abs/2311.03099
        
        Args:
            models: List of fine-tuned models
            weights: Weight for each model
            drop_rate: Fraction of delta values to drop (0.9 = keep only 10%)
            rescale: Whether to rescale remaining values
            
        Returns:
            Merged model
        """
        if self.base_model is None:
            raise ValueError("DARE requires a base model")
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        merged = copy.deepcopy(self.base_model)
        base_state = self.base_model.state_dict()
        merged_state = merged.state_dict()
        
        for key in merged_state:
            merged_delta = torch.zeros_like(merged_state[key], dtype=torch.float32)
            
            for model, weight in zip(models, weights):
                # Compute delta
                delta = model.state_dict()[key].float() - base_state[key].float()
                
                # Random drop mask
                mask = torch.bernoulli(
                    torch.ones_like(delta) * (1 - drop_rate)
                )
                
                # Apply mask
                sparse_delta = delta * mask
                
                # Rescale to maintain expected value
                if rescale:
                    sparse_delta = sparse_delta / (1 - drop_rate + 1e-10)
                
                merged_delta += weight * sparse_delta
            
            merged_state[key] = (base_state[key].float() + merged_delta).to(base_state[key].dtype)
        
        merged.load_state_dict(merged_state)
        return merged
    
    def model_soup(
        self,
        models: List[nn.Module],
    ) -> nn.Module:
        """
        Model Soup: Simple averaging of model checkpoints.
        
        Often improves robustness and generalization.
        
        Paper: https://arxiv.org/abs/2203.05482
        
        Args:
            models: List of model checkpoints to average
            
        Returns:
            Averaged model
        """
        return self.linear_merge(models)  # Equal weighting


# =============================================================================
# Convenience Functions
# =============================================================================

def linear_merge(
    models: List[nn.Module],
    weights: Optional[List[float]] = None,
) -> nn.Module:
    """Quick linear merge of models."""
    return ModelMerger().linear_merge(models, weights)


def slerp_merge(
    model_a: nn.Module,
    model_b: nn.Module,
    t: float = 0.5,
) -> nn.Module:
    """Quick SLERP merge of two models."""
    return ModelMerger().slerp_merge(model_a, model_b, t)


def ties_merge(
    base_model: nn.Module,
    models: List[nn.Module],
    threshold: float = 0.2,
) -> nn.Module:
    """Quick TIES merge with base model."""
    return ModelMerger(base_model).ties_merge(models, threshold=threshold)


def dare_merge(
    base_model: nn.Module,
    models: List[nn.Module],
    drop_rate: float = 0.9,
) -> nn.Module:
    """Quick DARE merge with base model."""
    return ModelMerger(base_model).dare_merge(models, drop_rate=drop_rate)


# =============================================================================
# Merge Analysis Tools
# =============================================================================

def compute_model_similarity(
    model_a: nn.Module,
    model_b: nn.Module,
) -> Dict[str, float]:
    """
    Compute similarity metrics between two models.
    
    Returns:
        Dictionary with cosine similarity and L2 distance
    """
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    
    flat_a = torch.cat([p.flatten().float() for p in state_a.values()])
    flat_b = torch.cat([p.flatten().float() for p in state_b.values()])
    
    cosine = F.cosine_similarity(flat_a.unsqueeze(0), flat_b.unsqueeze(0)).item()
    l2_dist = (flat_a - flat_b).norm().item()
    
    return {
        "cosine_similarity": cosine,
        "l2_distance": l2_dist,
    }


def compute_task_vector_stats(
    base_model: nn.Module,
    finetuned_model: nn.Module,
) -> Dict[str, float]:
    """
    Analyze the task vector (delta from base to finetuned).
    
    Returns:
        Statistics about the task vector
    """
    base_state = base_model.state_dict()
    ft_state = finetuned_model.state_dict()
    
    total_delta = 0.0
    total_magnitude = 0.0
    num_params = 0
    
    for key in base_state:
        delta = (ft_state[key].float() - base_state[key].float()).abs()
        total_delta += delta.sum().item()
        total_magnitude += base_state[key].float().abs().sum().item()
        num_params += delta.numel()
    
    return {
        "mean_delta": total_delta / num_params,
        "relative_change": total_delta / (total_magnitude + 1e-10),
        "num_parameters": num_params,
    }


import torch.nn.functional as F
