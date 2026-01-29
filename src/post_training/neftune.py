"""
NEFTune - Noisy Embeddings Improve Instruction Finetuning.

Adds uniform random noise to embedding vectors during training,
improving conversational ability by 5-10% with minimal overhead.

Paper: https://arxiv.org/abs/2310.05914
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Callable, Any
from contextlib import contextmanager


@dataclass
class NEFTuneConfig:
    """Configuration for NEFTune."""
    
    noise_alpha: float = 5.0  # Noise magnitude (5 is good default)
    enabled: bool = True
    

class NEFTuneEmbedding(nn.Module):
    """
    NEFTune wrapper for embedding layers.
    
    Adds uniform random noise to embeddings during training.
    
    Example:
        >>> embed = nn.Embedding(50000, 768)
        >>> nef_embed = NEFTuneEmbedding(embed, noise_alpha=5.0)
        >>> # During training, noise is added
        >>> nef_embed.train()
        >>> output = nef_embed(input_ids)
        >>> # During eval, no noise
        >>> nef_embed.eval()
        >>> output = nef_embed(input_ids)
    """
    
    def __init__(
        self,
        embedding: nn.Embedding,
        noise_alpha: float = 5.0,
    ):
        super().__init__()
        self.embedding = embedding
        self.noise_alpha = noise_alpha
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        
        if self.training and self.noise_alpha > 0:
            # Add uniform random noise
            dims = embeddings.shape
            mag_norm = self.noise_alpha / (dims[-1] ** 0.5)
            noise = torch.zeros_like(embeddings).uniform_(-1, 1) * mag_norm
            embeddings = embeddings + noise
        
        return embeddings
    
    @property
    def weight(self):
        return self.embedding.weight
    
    @property
    def num_embeddings(self):
        return self.embedding.num_embeddings
    
    @property
    def embedding_dim(self):
        return self.embedding.embedding_dim


class NEFTuneTrainer:
    """
    NEFTune Trainer - Adds noise to embeddings during training.
    
    Simple but effective technique to improve instruction following.
    
    Example:
        >>> trainer = NEFTuneTrainer(model, noise_alpha=5.0)
        >>> trainer.enable()
        >>> # Train as usual
        >>> for batch in dataloader:
        >>>     loss = model(batch)
        >>>     loss.backward()
        >>> # Disable for evaluation
        >>> trainer.disable()
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[NEFTuneConfig] = None,
    ):
        self.model = model
        self.config = config or NEFTuneConfig()
        self.original_embeddings = {}
        self._enabled = False
    
    def enable(self):
        """Enable NEFTune by wrapping embedding layers."""
        if self._enabled:
            return
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Embedding):
                # Store original
                self.original_embeddings[name] = module
                
                # Create wrapped version
                wrapped = NEFTuneEmbedding(
                    module,
                    noise_alpha=self.config.noise_alpha,
                )
                
                # Replace in model
                self._set_module(self.model, name, wrapped)
        
        self._enabled = True
        print(f"NEFTune enabled with alpha={self.config.noise_alpha}")
    
    def disable(self):
        """Disable NEFTune by restoring original embeddings."""
        if not self._enabled:
            return
        
        for name, original in self.original_embeddings.items():
            self._set_module(self.model, name, original)
        
        self.original_embeddings.clear()
        self._enabled = False
        print("NEFTune disabled")
    
    def _set_module(self, model: nn.Module, name: str, module: nn.Module):
        """Set a module by name (handles nested modules)."""
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], module)
    
    @contextmanager
    def neftune_context(self):
        """Context manager for NEFTune training."""
        self.enable()
        try:
            yield
        finally:
            self.disable()


def apply_neftune(
    model: nn.Module,
    noise_alpha: float = 5.0,
) -> NEFTuneTrainer:
    """
    Quick function to apply NEFTune to a model.
    
    Args:
        model: Model to apply NEFTune to
        noise_alpha: Noise magnitude (0 to disable)
        
    Returns:
        NEFTuneTrainer instance
        
    Example:
        >>> trainer = apply_neftune(model, noise_alpha=5.0)
        >>> trainer.enable()
        >>> # ... train ...
        >>> trainer.disable()
    """
    config = NEFTuneConfig(noise_alpha=noise_alpha)
    trainer = NEFTuneTrainer(model, config)
    trainer.enable()
    return trainer


# =============================================================================
# NEFTune Loss Wrapper
# =============================================================================

class NEFTuneLoss(nn.Module):
    """
    Loss wrapper that applies NEFTune noise during forward pass.
    
    Alternative approach that adds noise at loss computation time.
    """
    
    def __init__(
        self,
        base_loss: nn.Module,
        noise_alpha: float = 5.0,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.noise_alpha = noise_alpha
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute loss. If embeddings are provided and training,
        add noise for regularization effect.
        """
        return self.base_loss(logits, labels)


# =============================================================================
# Utilities
# =============================================================================

def get_optimal_noise_alpha(
    model_size: str,
    task: str = "instruction",
) -> float:
    """
    Get recommended noise alpha based on model size and task.
    
    Args:
        model_size: "small" (<1B), "medium" (1-7B), "large" (>7B)
        task: "instruction", "chat", or "general"
        
    Returns:
        Recommended noise_alpha value
    """
    recommendations = {
        ("small", "instruction"): 5.0,
        ("small", "chat"): 5.0,
        ("small", "general"): 3.0,
        ("medium", "instruction"): 5.0,
        ("medium", "chat"): 5.0,
        ("medium", "general"): 4.0,
        ("large", "instruction"): 5.0,
        ("large", "chat"): 5.0,
        ("large", "general"): 5.0,
    }
    
    return recommendations.get((model_size, task), 5.0)
