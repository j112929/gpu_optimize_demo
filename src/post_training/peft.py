"""
PEFT (Parameter-Efficient Fine-Tuning) Methods.

Provides various PEFT techniques:
- Adapter layers
- Prefix tuning
- Prompt tuning
- IA3 (Infused Adapter by Inhibiting and Amplifying)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import math


@dataclass
class PEFTConfig:
    """Base configuration for PEFT methods."""
    peft_type: str = "adapter"    # adapter, prefix, prompt, ia3
    
    # Common
    trainable_modules: List[str] = field(default_factory=list)


# =============================================================================
# Adapter Layers
# =============================================================================

@dataclass
class AdapterConfig(PEFTConfig):
    """Configuration for Adapter layers."""
    peft_type: str = "adapter"
    
    # Architecture
    bottleneck_dim: int = 64
    non_linearity: str = "gelu"   # relu, gelu, swish
    
    # Placement
    add_after_attention: bool = True
    add_after_ffn: bool = True
    
    # Initialization
    init_scale: float = 1e-3


class AdapterLayer(nn.Module):
    """
    Adapter layer for efficient fine-tuning.
    
    Architecture: h = h + f(h @ W_down) @ W_up
    
    Adds only ~1-5% parameters while achieving competitive performance.
    """
    
    def __init__(
        self,
        hidden_size: int,
        bottleneck_dim: int = 64,
        non_linearity: str = "gelu",
        init_scale: float = 1e-3,
    ):
        super().__init__()
        
        self.down_proj = nn.Linear(hidden_size, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, hidden_size)
        
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "swish": nn.SiLU(),
        }
        self.activation = activations.get(non_linearity, nn.GELU())
        
        # Initialize with small values
        nn.init.normal_(self.down_proj.weight, std=init_scale)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.normal_(self.up_proj.weight, std=init_scale)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual adapter
        h = self.down_proj(x)
        h = self.activation(h)
        h = self.up_proj(h)
        return x + h


class AdapterModel(nn.Module):
    """
    Wrapper to add Adapter layers to a transformer model.
    
    Example:
        >>> config = AdapterConfig(bottleneck_dim=64)
        >>> adapter_model = AdapterModel(base_model, config)
        >>> # Only adapters are trainable
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[AdapterConfig] = None,
    ):
        super().__init__()
        
        self.model = model
        self.config = config or AdapterConfig()
        self.adapters: Dict[str, AdapterLayer] = {}
        
        # Freeze base model
        for param in self.model.parameters():
            param.requires_grad = False
        
        self._add_adapters()
    
    def _add_adapters(self):
        """Add adapter layers to the model."""
        for name, module in self.model.named_modules():
            # Find attention and FFN outputs
            if self.config.add_after_attention and "attn" in name.lower():
                self._add_adapter_after(name, module)
            elif self.config.add_after_ffn and "mlp" in name.lower():
                self._add_adapter_after(name, module)
    
    def _add_adapter_after(self, name: str, module: nn.Module):
        """Add adapter after a module."""
        # Get hidden size from module
        hidden_size = None
        for child in module.children():
            if isinstance(child, nn.Linear):
                hidden_size = child.out_features
                break
        
        if hidden_size is None:
            return
        
        adapter = AdapterLayer(
            hidden_size,
            self.config.bottleneck_dim,
            self.config.non_linearity,
            self.config.init_scale,
        )
        
        self.adapters[name] = adapter
        self.add_module(f"adapter_{name.replace('.', '_')}", adapter)
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


# =============================================================================
# Prefix Tuning
# =============================================================================

@dataclass
class PrefixConfig(PEFTConfig):
    """Configuration for Prefix Tuning."""
    peft_type: str = "prefix"
    
    # Prefix
    prefix_length: int = 20
    prefix_projection: bool = True
    prefix_hidden_dim: int = 512
    
    # Model
    num_layers: int = 12
    num_heads: int = 12
    head_dim: int = 64


class PrefixTuning(nn.Module):
    """
    Prefix Tuning for transformer models.
    
    Prepends learnable prefix tokens to the key and value
    of attention layers. Very parameter-efficient.
    
    Example:
        >>> config = PrefixConfig(prefix_length=20, num_layers=12)
        >>> prefix = PrefixTuning(config)
        >>> 
        >>> # Get prefix for layer 0
        >>> past_key_values = prefix()
    """
    
    def __init__(self, config: PrefixConfig):
        super().__init__()
        
        self.config = config
        self.prefix_length = config.prefix_length
        
        # Prefix embedding
        if config.prefix_projection:
            # Use MLP to parameterize prefix
            self.prefix_embedding = nn.Embedding(
                config.prefix_length,
                config.prefix_hidden_dim,
            )
            self.prefix_mlp = nn.Sequential(
                nn.Linear(config.prefix_hidden_dim, config.prefix_hidden_dim),
                nn.Tanh(),
                nn.Linear(
                    config.prefix_hidden_dim,
                    config.num_layers * 2 * config.num_heads * config.head_dim,
                ),
            )
        else:
            # Direct prefix parameters
            total_dim = config.num_layers * 2 * config.num_heads * config.head_dim
            self.prefix_embedding = nn.Parameter(
                torch.randn(config.prefix_length, total_dim)
            )
            self.prefix_mlp = None
    
    def forward(
        self,
        batch_size: int = 1,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        Get prefix key-value pairs for all layers.
        
        Returns:
            Tuple of (key, value) pairs for each layer
        """
        if self.prefix_mlp is not None:
            prefix_idx = torch.arange(self.prefix_length, device=self.prefix_embedding.weight.device)
            prefix = self.prefix_embedding(prefix_idx)
            prefix = self.prefix_mlp(prefix)
        else:
            prefix = self.prefix_embedding
        
        # Reshape for layers
        prefix = prefix.view(
            self.prefix_length,
            self.config.num_layers,
            2,
            self.config.num_heads,
            self.config.head_dim,
        )
        
        # Expand for batch
        prefix = prefix.unsqueeze(0).expand(batch_size, -1, -1, -1, -1, -1)
        
        # Split into per-layer key-value pairs
        past_key_values = []
        for layer_idx in range(self.config.num_layers):
            key = prefix[:, :, layer_idx, 0]   # [batch, prefix_len, heads, dim]
            value = prefix[:, :, layer_idx, 1]
            past_key_values.append((key, value))
        
        return tuple(past_key_values)


# =============================================================================
# Prompt Tuning
# =============================================================================

@dataclass
class PromptConfig(PEFTConfig):
    """Configuration for Prompt Tuning."""
    peft_type: str = "prompt"
    
    # Prompt
    num_virtual_tokens: int = 20
    prompt_init: str = "random"   # random, text
    prompt_init_text: Optional[str] = None
    
    # Model
    hidden_size: int = 768
    tokenizer: Optional[Any] = None


class PromptTuning(nn.Module):
    """
    Prompt Tuning with learnable soft prompts.
    
    Prepends learnable embeddings to input, much simpler
    than prefix tuning but still effective.
    
    Example:
        >>> config = PromptConfig(num_virtual_tokens=20, hidden_size=768)
        >>> prompt = PromptTuning(config)
        >>> 
        >>> # Prepend to input embeddings
        >>> input_embeds = model.embed(input_ids)
        >>> input_embeds = prompt(input_embeds)
    """
    
    def __init__(self, config: PromptConfig):
        super().__init__()
        
        self.config = config
        
        # Learnable prompt embeddings
        self.prompt_embeddings = nn.Parameter(
            torch.randn(config.num_virtual_tokens, config.hidden_size)
        )
        
        # Initialize
        self._init_prompt()
    
    def _init_prompt(self):
        """Initialize prompt embeddings."""
        if self.config.prompt_init == "random":
            nn.init.normal_(self.prompt_embeddings, std=0.02)
        elif self.config.prompt_init == "text" and self.config.prompt_init_text:
            # Initialize from text embeddings
            # Requires tokenizer and embedding layer
            pass
    
    def forward(self, input_embeds: torch.Tensor) -> torch.Tensor:
        """
        Prepend prompt embeddings to input.
        
        Args:
            input_embeds: [batch, seq_len, hidden_size]
            
        Returns:
            [batch, num_virtual_tokens + seq_len, hidden_size]
        """
        batch_size = input_embeds.size(0)
        
        # Expand prompt for batch
        prompt = self.prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Prepend to input
        return torch.cat([prompt, input_embeds], dim=1)


# =============================================================================
# IA3 (Infused Adapter by Inhibiting and Amplifying)
# =============================================================================

@dataclass
class IA3Config(PEFTConfig):
    """Configuration for IA3."""
    peft_type: str = "ia3"
    
    # Target modules
    target_modules: List[str] = field(default_factory=lambda: [
        "k_proj", "v_proj", "down_proj"
    ])


class IA3Layer(nn.Module):
    """
    IA3 layer: learned rescaling vectors.
    
    Extremely parameter-efficient - only adds one vector per layer.
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        
        self.ia3_vector = nn.Parameter(torch.ones(hidden_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.ia3_vector


# =============================================================================
# Utilities
# =============================================================================

def count_trainable_params(model: nn.Module) -> Tuple[int, int]:
    """Count trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def freeze_model(model: nn.Module, except_patterns: Optional[List[str]] = None):
    """Freeze all parameters except those matching patterns."""
    except_patterns = except_patterns or []
    
    for name, param in model.named_parameters():
        should_freeze = True
        for pattern in except_patterns:
            if pattern in name:
                should_freeze = False
                break
        param.requires_grad = not should_freeze


def get_peft_model(
    model: nn.Module,
    peft_type: str = "lora",
    **kwargs,
) -> nn.Module:
    """
    Get PEFT model with specified method.
    
    Args:
        model: Base model
        peft_type: "lora", "adapter", "prefix", "prompt"
        **kwargs: Method-specific arguments
        
    Returns:
        PEFT-wrapped model
    """
    from src.post_training.lora import LoRAModel, LoRAConfig
    
    if peft_type == "lora":
        config = LoRAConfig(**kwargs)
        return LoRAModel(model, config)
    elif peft_type == "adapter":
        config = AdapterConfig(**kwargs)
        return AdapterModel(model, config)
    else:
        raise ValueError(f"Unknown PEFT type: {peft_type}")
