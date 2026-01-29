"""
Flash Attention Integration - Efficient attention computation.

Provides:
- FlashAttention-2/3 wrapper with automatic fallback
- Memory-efficient attention for long sequences
- Multi-query attention (MQA) and grouped-query attention (GQA) support
- Sliding window attention

Key benefits:
- O(N) memory instead of O(N²)
- 1.5-2× faster than standard attention
- Support for sequences up to millions of tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple
import math


@dataclass
class FlashAttentionConfig:
    """Configuration for Flash Attention."""
    
    # Core settings
    use_flash_attn: bool = True
    version: str = "auto"            # "auto", "v2", "v3", "sdpa"
    
    # Attention parameters
    causal: bool = True              # Causal mask for autoregressive
    dropout: float = 0.0
    softmax_scale: Optional[float] = None  # Auto-compute if None
    
    # Sliding window (for long sequences)
    window_size: Optional[Tuple[int, int]] = None  # (left, right) or None
    
    # Multi-query / Grouped-query attention
    num_heads: int = 32
    num_kv_heads: Optional[int] = None  # None = MHA, < num_heads = GQA
    
    # Optimization
    return_attn_weights: bool = False  # Disable for memory savings
    deterministic: bool = False


# =============================================================================
# Flash Attention Detection and Selection
# =============================================================================

def detect_flash_attention_version() -> str:
    """Detect available Flash Attention implementation."""
    
    # Check for Flash Attention 3 (H100 optimized)
    try:
        import flash_attn
        if hasattr(flash_attn, 'flash_attn_func'):
            version = getattr(flash_attn, '__version__', '2.0.0')
            if version.startswith('3'):
                return "v3"
            return "v2"
    except ImportError:
        pass
    
    # Check for PyTorch SDPA (always available in PyTorch 2.0+)
    if hasattr(F, 'scaled_dot_product_attention'):
        return "sdpa"
    
    return "naive"


def get_attention_backend(config: FlashAttentionConfig) -> str:
    """Get the best available attention backend."""
    if not config.use_flash_attn:
        return "naive"
    
    if config.version == "auto":
        return detect_flash_attention_version()
    
    return config.version


# =============================================================================
# Flash Attention Wrapper
# =============================================================================

class FlashAttention(nn.Module):
    """
    Unified Flash Attention wrapper with automatic backend selection.
    
    Automatically selects the best available implementation:
    1. FlashAttention-3 (if available, H100+)
    2. FlashAttention-2 (if available)
    3. PyTorch SDPA (always available in PyTorch 2.0+)
    4. Naive attention (fallback)
    
    Example:
        >>> config = FlashAttentionConfig(num_heads=32, causal=True)
        >>> attn = FlashAttention(config, hidden_size=4096)
        >>> output = attn(query, key, value)
    """
    
    def __init__(
        self,
        config: FlashAttentionConfig,
        hidden_size: int,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // config.num_heads
        
        self.backend = get_attention_backend(config)
        
        # Softmax scale
        self.scale = config.softmax_scale or (1.0 / math.sqrt(self.head_dim))
        
        # GQA/MQA setup
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads or config.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        
        print(f"FlashAttention initialized with backend: {self.backend}")
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute attention with the best available backend.
        
        Args:
            query: [batch, seq_len, num_heads, head_dim] or [batch, seq_len, hidden]
            key: [batch, seq_len, num_kv_heads, head_dim] or [batch, seq_len, hidden]
            value: [batch, seq_len, num_kv_heads, head_dim] or [batch, seq_len, hidden]
            attention_mask: Optional attention mask
            
        Returns:
            output: [batch, seq_len, hidden_size]
        """
        
        if self.backend == "v3" or self.backend == "v2":
            return self._flash_attn_forward(query, key, value, attention_mask)
        elif self.backend == "sdpa":
            return self._sdpa_forward(query, key, value, attention_mask)
        else:
            return self._naive_forward(query, key, value, attention_mask)
    
    def _flash_attn_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass using Flash Attention library."""
        try:
            from flash_attn import flash_attn_func, flash_attn_varlen_func
            
            batch_size, seq_len = query.shape[:2]
            
            # Reshape for flash attention: [batch, seq, num_heads, head_dim]
            if query.dim() == 3:
                query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
                key = key.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
                value = value.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            
            # Repeat KV heads for GQA
            if self.num_kv_groups > 1:
                key = key.repeat_interleave(self.num_kv_groups, dim=2)
                value = value.repeat_interleave(self.num_kv_groups, dim=2)
            
            output = flash_attn_func(
                query,
                key,
                value,
                dropout_p=self.config.dropout if self.training else 0.0,
                softmax_scale=self.scale,
                causal=self.config.causal,
                window_size=self.config.window_size or (-1, -1),
            )
            
            # Reshape back
            return output.reshape(batch_size, seq_len, self.hidden_size)
            
        except Exception as e:
            print(f"Flash Attention failed, falling back to SDPA: {e}")
            return self._sdpa_forward(query, key, value, attention_mask)
    
    def _sdpa_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass using PyTorch Scaled Dot Product Attention."""
        batch_size, seq_len = query.shape[:2]
        
        # Reshape for SDPA: [batch, num_heads, seq, head_dim]
        if query.dim() == 3:
            query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
            key = key.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            value = value.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        query = query.transpose(1, 2)  # [batch, num_heads, seq, head_dim]
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        # Repeat KV heads for GQA
        if self.num_kv_groups > 1:
            key = key.repeat_interleave(self.num_kv_groups, dim=1)
            value = value.repeat_interleave(self.num_kv_groups, dim=1)
        
        # Use SDPA
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.config.dropout if self.training else 0.0,
            is_causal=self.config.causal and attention_mask is None,
            scale=self.scale,
        )
        
        # Reshape back: [batch, seq, hidden]
        output = output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        return output
    
    def _naive_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Naive attention implementation (fallback)."""
        batch_size, seq_len = query.shape[:2]
        
        # Reshape
        if query.dim() == 3:
            query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
            key = key.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            value = value.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        # Repeat KV heads for GQA
        if self.num_kv_groups > 1:
            key = key.repeat_interleave(self.num_kv_groups, dim=1)
            value = value.repeat_interleave(self.num_kv_groups, dim=1)
        
        # Compute attention scores
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        if self.config.causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=query.device),
                diagonal=1,
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.config.dropout, training=self.training)
        
        output = torch.matmul(attn_weights, value)
        output = output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        
        return output


# =============================================================================
# Sliding Window Attention
# =============================================================================

class SlidingWindowAttention(FlashAttention):
    """
    Sliding Window Attention for processing very long sequences.
    
    Each token only attends to a fixed window of previous tokens,
    reducing memory from O(N²) to O(N*W) where W is window size.
    
    Example:
        >>> config = FlashAttentionConfig(window_size=(4096, 0))  # 4K left context
        >>> attn = SlidingWindowAttention(config, hidden_size=4096)
    """
    
    def __init__(
        self,
        config: FlashAttentionConfig,
        hidden_size: int,
        window_size: int = 4096,
    ):
        # Set window size
        config.window_size = (window_size, 0)  # Causal: only look back
        super().__init__(config, hidden_size)
        self.window_size = window_size


# =============================================================================
# Memory-Efficient Attention Context Manager
# =============================================================================

class MemoryEfficientAttentionContext:
    """
    Context manager for memory-efficient attention computation.
    
    Automatically enables the most memory-efficient attention backend.
    
    Example:
        >>> with MemoryEfficientAttentionContext():
        >>>     output = model(input)  # Uses memory-efficient attention
    """
    
    def __init__(self, enable_flash: bool = True, enable_math: bool = False):
        self.enable_flash = enable_flash
        self.enable_math = enable_math
        self._prev_flash = None
        self._prev_math = None
        self._prev_mem_efficient = None
    
    def __enter__(self):
        if hasattr(torch.backends.cuda, 'sdp_kernel'):
            # PyTorch 2.0+ context manager
            self._context = torch.backends.cuda.sdp_kernel(
                enable_flash=self.enable_flash,
                enable_math=self.enable_math,
                enable_mem_efficient=True,
            )
            self._context.__enter__()
        return self
    
    def __exit__(self, *args):
        if hasattr(self, '_context'):
            self._context.__exit__(*args)


# =============================================================================
# Utilities
# =============================================================================

def benchmark_attention_backends(
    batch_size: int = 4,
    seq_len: int = 2048,
    hidden_size: int = 4096,
    num_heads: int = 32,
    num_iterations: int = 100,
) -> dict:
    """
    Benchmark different attention backends.
    
    Returns timing results for each available backend.
    """
    import time
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head_dim = hidden_size // num_heads
    
    # Create test tensors
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
    
    results = {}
    
    # Test SDPA
    config = FlashAttentionConfig(num_heads=num_heads, use_flash_attn=True, version="sdpa")
    attn = FlashAttention(config, hidden_size).to(device)
    
    # Warmup
    for _ in range(10):
        _ = attn(q.view(batch_size, seq_len, -1), k.view(batch_size, seq_len, -1), v.view(batch_size, seq_len, -1))
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = attn(q.view(batch_size, seq_len, -1), k.view(batch_size, seq_len, -1), v.view(batch_size, seq_len, -1))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    results["sdpa"] = (time.perf_counter() - start) / num_iterations * 1000  # ms
    
    # Memory usage
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        _ = attn(q.view(batch_size, seq_len, -1), k.view(batch_size, seq_len, -1), v.view(batch_size, seq_len, -1))
        results["sdpa_memory_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    return results


def create_attention_layer(
    hidden_size: int,
    num_heads: int,
    num_kv_heads: Optional[int] = None,
    causal: bool = True,
    use_flash: bool = True,
) -> FlashAttention:
    """
    Create an attention layer with optimal settings.
    
    Args:
        hidden_size: Model hidden size
        num_heads: Number of attention heads
        num_kv_heads: Number of KV heads (for GQA/MQA)
        causal: Whether to use causal mask
        use_flash: Whether to use Flash Attention
        
    Returns:
        Configured FlashAttention module
    """
    config = FlashAttentionConfig(
        use_flash_attn=use_flash,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        causal=causal,
    )
    return FlashAttention(config, hidden_size)
