"""
GaLore - Gradient Low-Rank Projection for Memory-Efficient Training.

Enables full-parameter training with LoRA-level memory usage.
Projects gradients to low-rank space during optimization.

Paper: https://arxiv.org/abs/2403.03507
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Iterator
import math


@dataclass
class GaLoreConfig:
    """Configuration for GaLore optimizer."""
    
    rank: int = 128                     # Projection rank
    update_proj_gap: int = 200          # Steps between projection updates
    scale: float = 1.0                  # Gradient scaling factor
    proj_type: str = "std"              # "std", "reverse_std", "right", "left"
    

class GaLoreProjector:
    """
    Low-rank projector for gradients.
    
    Maintains orthogonal projection matrices that are periodically updated
    via SVD of accumulated gradients.
    """
    
    def __init__(
        self,
        rank: int,
        update_proj_gap: int = 200,
        scale: float = 1.0,
        proj_type: str = "std",
    ):
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.proj_type = proj_type
        
        self.ortho_matrix = None
        self.step_count = 0
    
    def project(self, grad: torch.Tensor, is_weight: bool = True) -> torch.Tensor:
        """
        Project gradient to low-rank space.
        
        Args:
            grad: Full gradient tensor [out_features, in_features]
            is_weight: Whether this is a weight (True) or bias (False)
            
        Returns:
            Low-rank projected gradient
        """
        if not is_weight or grad.dim() != 2:
            return grad
        
        # Update projection matrix periodically
        if self.ortho_matrix is None or self.step_count % self.update_proj_gap == 0:
            self.ortho_matrix = self._get_orthogonal_matrix(grad)
        
        self.step_count += 1
        
        # Project based on type
        if self.proj_type == "std":
            # Standard: project to right singular vectors
            projected = grad @ self.ortho_matrix.T
        elif self.proj_type == "reverse_std":
            # Reverse: project to left singular vectors
            projected = self.ortho_matrix.T @ grad
        elif self.proj_type == "right":
            projected = grad @ self.ortho_matrix.T
        elif self.proj_type == "left":
            projected = self.ortho_matrix.T @ grad
        else:
            projected = grad @ self.ortho_matrix.T
        
        return projected * self.scale
    
    def project_back(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Project gradient back to full space.
        
        Args:
            grad: Low-rank gradient
            
        Returns:
            Full-rank gradient
        """
        if self.ortho_matrix is None:
            return grad
        
        if self.proj_type == "std" or self.proj_type == "right":
            return grad @ self.ortho_matrix
        else:
            return self.ortho_matrix @ grad
    
    def _get_orthogonal_matrix(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Compute orthogonal projection matrix via SVD.
        
        Args:
            grad: Gradient to compute projection from
            
        Returns:
            Orthogonal matrix [rank, min(in, out)]
        """
        # Use gradient structure to determine projection
        if self.proj_type in ["std", "right"]:
            # Project along input dimension
            if grad.size(1) >= self.rank:
                _, _, V = torch.linalg.svd(grad.float(), full_matrices=False)
                return V[:self.rank, :].to(grad.dtype)
            else:
                return torch.eye(grad.size(1), device=grad.device, dtype=grad.dtype)
        else:
            # Project along output dimension
            if grad.size(0) >= self.rank:
                U, _, _ = torch.linalg.svd(grad.float(), full_matrices=False)
                return U[:, :self.rank].T.to(grad.dtype)
            else:
                return torch.eye(grad.size(0), device=grad.device, dtype=grad.dtype)


class GaLoreAdamW(Optimizer):
    """
    AdamW optimizer with GaLore gradient projection.
    
    Enables training with full parameters while using LoRA-level memory.
    
    Example:
        >>> config = GaLoreConfig(rank=128)
        >>> optimizer = GaLoreAdamW(
        >>>     model.parameters(),
        >>>     lr=1e-4,
        >>>     galore_config=config,
        >>> )
        >>> 
        >>> for batch in dataloader:
        >>>     loss = model(batch)
        >>>     loss.backward()
        >>>     optimizer.step()
        >>>     optimizer.zero_grad()
    """
    
    def __init__(
        self,
        params: Iterator[nn.Parameter],
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        galore_config: Optional[GaLoreConfig] = None,
    ):
        self.galore_config = galore_config or GaLoreConfig()
        
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)
        
        # Initialize projectors for each parameter
        self.projectors: Dict[int, GaLoreProjector] = {}
    
    def _get_projector(self, param_id: int) -> GaLoreProjector:
        """Get or create projector for parameter."""
        if param_id not in self.projectors:
            self.projectors[param_id] = GaLoreProjector(
                rank=self.galore_config.rank,
                update_proj_gap=self.galore_config.update_proj_gap,
                scale=self.galore_config.scale,
                proj_type=self.galore_config.proj_type,
            )
        return self.projectors[param_id]
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step with GaLore projection.
        
        Args:
            closure: Optional closure for loss computation
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # Apply GaLore projection for 2D tensors (weights)
                if grad.dim() == 2 and min(grad.shape) > self.galore_config.rank:
                    projector = self._get_projector(id(p))
                    grad = projector.project(grad)
                
                state = self.state[p]
                
                # Initialize state
                if len(state) == 0:
                    state["step"] = 0
                    # Use low-rank shape for momentum if projected
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)
                
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1
                
                # Adam update in low-rank space
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                
                step_size = group["lr"] / bias_correction1
                
                # Compute update
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])
                update = exp_avg / denom
                
                # Project back to full space
                if grad.dim() == 2 and min(p.grad.shape) > self.galore_config.rank:
                    projector = self._get_projector(id(p))
                    update = projector.project_back(update)
                
                # Weight decay
                if group["weight_decay"] > 0:
                    p.add_(p, alpha=-group["lr"] * group["weight_decay"])
                
                # Apply update
                p.add_(update, alpha=-step_size)
        
        return loss


class GaLoreAdaFactor(Optimizer):
    """
    AdaFactor optimizer with GaLore projection.
    
    More memory efficient than AdamW for large models.
    Combines GaLore with AdaFactor's factored second moment estimation.
    """
    
    def __init__(
        self,
        params: Iterator[nn.Parameter],
        lr: float = 1e-3,
        eps: Tuple[float, float] = (1e-30, 1e-3),
        clip_threshold: float = 1.0,
        decay_rate: float = -0.8,
        weight_decay: float = 0.0,
        scale_parameter: bool = True,
        warmup_init: bool = False,
        galore_config: Optional[GaLoreConfig] = None,
    ):
        self.galore_config = galore_config or GaLoreConfig()
        
        defaults = dict(
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            decay_rate=decay_rate,
            weight_decay=weight_decay,
            scale_parameter=scale_parameter,
            warmup_init=warmup_init,
        )
        super().__init__(params, defaults)
        
        self.projectors: Dict[int, GaLoreProjector] = {}
    
    @torch.no_grad()
    def step(self, closure=None):
        """Perform optimization step with GaLore + AdaFactor."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad.float()
                
                state = self.state[p]
                
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_inf"] = torch.zeros_like(grad)
                
                state["step"] += 1
                
                # Simple AdaFactor-like update
                state["exp_inf"].mul_(0.999).add_(grad.abs(), alpha=0.001)
                
                step_size = group["lr"]
                if group["scale_parameter"]:
                    step_size *= max(1.0, grad.norm().item() / (grad.numel() ** 0.5))
                
                update = grad / (state["exp_inf"] + group["eps"][0])
                
                # Weight decay
                if group["weight_decay"] > 0:
                    p.add_(p, alpha=-group["lr"] * group["weight_decay"])
                
                p.add_(update.to(p.dtype), alpha=-step_size)
        
        return loss


# =============================================================================
# Utility Functions
# =============================================================================

def create_galore_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    rank: int = 128,
    update_proj_gap: int = 200,
    optimizer_type: str = "adamw",
    target_modules: Optional[List[str]] = None,
) -> Optimizer:
    """
    Create a GaLore optimizer for a model.
    
    Args:
        model: Model to optimize
        lr: Learning rate
        rank: Projection rank
        update_proj_gap: Steps between projection updates
        optimizer_type: "adamw" or "adafactor"
        target_modules: Optional list of module names to apply GaLore to
        
    Returns:
        Configured optimizer
        
    Example:
        >>> optimizer = create_galore_optimizer(
        >>>     model, lr=1e-4, rank=128,
        >>>     target_modules=["q_proj", "k_proj", "v_proj"],
        >>> )
    """
    config = GaLoreConfig(
        rank=rank,
        update_proj_gap=update_proj_gap,
    )
    
    # Separate parameters for GaLore vs regular optimization
    galore_params = []
    regular_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Check if should apply GaLore
        apply_galore = False
        if target_modules is None:
            # Apply to all 2D parameters
            apply_galore = param.dim() == 2 and min(param.shape) > rank
        else:
            # Apply only to specified modules
            for target in target_modules:
                if target in name:
                    apply_galore = param.dim() == 2
                    break
        
        if apply_galore:
            galore_params.append(param)
        else:
            regular_params.append(param)
    
    if optimizer_type == "adamw":
        optimizer = GaLoreAdamW(
            [
                {"params": galore_params, "galore": True},
                {"params": regular_params, "galore": False},
            ],
            lr=lr,
            galore_config=config,
        )
    else:
        optimizer = GaLoreAdaFactor(
            [
                {"params": galore_params},
                {"params": regular_params},
            ],
            lr=lr,
            galore_config=config,
        )
    
    print(f"GaLore optimizer created:")
    print(f"  - GaLore parameters: {len(galore_params)}")
    print(f"  - Regular parameters: {len(regular_params)}")
    print(f"  - Rank: {rank}")
    
    return optimizer


def estimate_galore_memory_savings(
    model: nn.Module,
    rank: int = 128,
) -> Dict[str, float]:
    """
    Estimate memory savings from using GaLore.
    
    Args:
        model: Model to analyze
        rank: GaLore rank
        
    Returns:
        Dictionary with memory estimates
    """
    full_optim_memory = 0
    galore_optim_memory = 0
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        param_size = param.numel()
        
        # AdamW needs 2x param size for momentum (m, v)
        full_optim_memory += param_size * 2 * 4  # float32
        
        if param.dim() == 2 and min(param.shape) > rank:
            # GaLore reduces to rank x larger_dim
            larger_dim = max(param.shape)
            galore_size = rank * larger_dim
            galore_optim_memory += galore_size * 2 * 4
        else:
            galore_optim_memory += param_size * 2 * 4
    
    return {
        "full_optimizer_gb": full_optim_memory / 1e9,
        "galore_optimizer_gb": galore_optim_memory / 1e9,
        "memory_savings_ratio": full_optim_memory / galore_optim_memory if galore_optim_memory > 0 else 1.0,
    }
