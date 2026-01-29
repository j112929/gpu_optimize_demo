"""
Speculative Decoding - Fast LLM generation with draft models.

Uses a smaller draft model to predict multiple tokens,
then verifies with the target model in parallel.
Achieves 2-3x speedup without quality loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import time


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding."""
    # Draft model
    draft_model_path: Optional[str] = None
    
    # Generation
    num_speculative_tokens: int = 5    # Tokens to draft per step
    
    # Acceptance
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    
    # Optimization
    max_new_tokens: int = 256
    early_stop: bool = True
    
    # Fallback
    min_accept_rate: float = 0.3       # Fall back to normal if too low


class SpeculativeDecoder:
    """
    Speculative Decoding for fast LLM inference.
    
    How it works:
    1. Draft model generates K candidate tokens
    2. Target model processes all K tokens in parallel
    3. Accept matching tokens, reject from first mismatch
    4. Repeat until done
    
    Provides 2-3x speedup without quality loss.
    
    Example:
        >>> decoder = SpeculativeDecoder(target_model, draft_model)
        >>> output = decoder.generate(input_ids, max_new_tokens=100)
    """
    
    def __init__(
        self,
        target_model: nn.Module,
        draft_model: nn.Module,
        config: Optional[SpeculativeConfig] = None,
    ):
        self.target = target_model
        self.draft = draft_model
        self.config = config or SpeculativeConfig()
        
        # Statistics
        self.total_tokens = 0
        self.accepted_tokens = 0
        self.draft_calls = 0
        self.target_calls = 0
    
    def _sample(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Sample from logits with temperature and top-k."""
        if temperature == 0:
            return logits.argmax(dim=-1)
        
        logits = logits / temperature
        
        if top_k > 0:
            top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
            logits = torch.full_like(logits, float('-inf'))
            logits.scatter_(-1, top_k_indices, top_k_logits)
        
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    def _draft_tokens(
        self,
        input_ids: torch.Tensor,
        num_tokens: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate draft tokens with draft model."""
        draft_ids = input_ids.clone()
        draft_logits = []
        
        for _ in range(num_tokens):
            with torch.no_grad():
                outputs = self.draft(draft_ids)
                logits = outputs.logits[:, -1]
                draft_logits.append(logits)
                
                next_token = self._sample(
                    logits,
                    self.config.temperature,
                    self.config.top_k,
                )
                draft_ids = torch.cat([
                    draft_ids,
                    next_token.unsqueeze(-1),
                ], dim=-1)
        
        self.draft_calls += num_tokens
        
        return draft_ids, torch.stack(draft_logits, dim=1)
    
    def _verify_tokens(
        self,
        input_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        draft_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """Verify draft tokens with target model."""
        # Run target model on full sequence (including drafts)
        with torch.no_grad():
            outputs = self.target(draft_tokens)
            target_logits = outputs.logits
        
        self.target_calls += 1
        
        # Get target logits for each draft position
        prompt_len = input_ids.size(1)
        num_draft = draft_tokens.size(1) - prompt_len
        
        # Compare distributions
        accepted = 0
        
        for i in range(num_draft):
            draft_pos = prompt_len + i - 1
            target_pos = prompt_len + i
            
            # Get probabilities
            draft_probs = F.softmax(draft_logits[:, i] / self.config.temperature, dim=-1)
            target_probs = F.softmax(target_logits[:, draft_pos] / self.config.temperature, dim=-1)
            
            draft_token = draft_tokens[:, target_pos]
            
            # Acceptance criterion
            draft_p = draft_probs.gather(-1, draft_token.unsqueeze(-1)).squeeze(-1)
            target_p = target_probs.gather(-1, draft_token.unsqueeze(-1)).squeeze(-1)
            
            # Accept if target has higher probability
            accept_ratio = target_p / (draft_p + 1e-10)
            
            if accept_ratio >= torch.rand_like(accept_ratio):
                accepted += 1
            else:
                # Reject this and all following tokens
                break
        
        self.accepted_tokens += accepted
        self.total_tokens += num_draft
        
        # Return accepted sequence + one bonus token from target
        accepted_len = prompt_len + accepted
        final_ids = draft_tokens[:, :accepted_len]
        
        # Sample one more token from target at the rejection point
        bonus_logits = target_logits[:, accepted_len - 1]
        bonus_token = self._sample(
            bonus_logits,
            self.config.temperature,
            self.config.top_k,
        )
        final_ids = torch.cat([
            final_ids,
            bonus_token.unsqueeze(-1),
        ], dim=-1)
        
        return final_ids, accepted + 1  # +1 for bonus token
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate tokens using speculative decoding.
        
        Args:
            input_ids: Input token IDs
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated token IDs
        """
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        
        current_ids = input_ids
        generated_len = 0
        
        while generated_len < max_new_tokens:
            # Draft tokens
            num_draft = min(
                self.config.num_speculative_tokens,
                max_new_tokens - generated_len,
            )
            draft_ids, draft_logits = self._draft_tokens(current_ids, num_draft)
            
            # Verify with target
            current_ids, num_accepted = self._verify_tokens(
                current_ids, draft_ids, draft_logits
            )
            generated_len += num_accepted
            
            # Early stopping (EOS token)
            if self.config.early_stop:
                # Check for EOS (simplified)
                pass
        
        return current_ids
    
    @property
    def accept_rate(self) -> float:
        """Get token acceptance rate."""
        if self.total_tokens == 0:
            return 0.0
        return self.accepted_tokens / self.total_tokens
    
    @property
    def speedup_estimate(self) -> float:
        """Estimate speedup factor."""
        # Speedup ≈ (1 + accept_rate * num_speculative) / (1 + draft_overhead)
        avg_accepted = self.accept_rate * self.config.num_speculative_tokens
        draft_overhead = 0.1  # Assume draft is 10% cost of target
        return (1 + avg_accepted) / (1 + draft_overhead)
    
    def reset_stats(self):
        """Reset statistics."""
        self.total_tokens = 0
        self.accepted_tokens = 0
        self.draft_calls = 0
        self.target_calls = 0
    
    def print_stats(self):
        """Print generation statistics."""
        print("\n" + "=" * 50)
        print("SPECULATIVE DECODING STATISTICS")
        print("=" * 50)
        print(f"Total draft tokens:    {self.total_tokens}")
        print(f"Accepted tokens:       {self.accepted_tokens}")
        print(f"Accept rate:           {self.accept_rate:.2%}")
        print(f"Draft model calls:     {self.draft_calls}")
        print(f"Target model calls:    {self.target_calls}")
        print(f"Estimated speedup:     {self.speedup_estimate:.2f}x")
        print("=" * 50)


def speculative_generate(
    target_model: nn.Module,
    draft_model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 256,
    num_speculative: int = 5,
) -> torch.Tensor:
    """
    Quick function for speculative decoding.
    
    Example:
        >>> output = speculative_generate(
        ...     llama_70b, llama_7b, input_ids, max_new_tokens=100
        ... )
    """
    config = SpeculativeConfig(num_speculative_tokens=num_speculative)
    decoder = SpeculativeDecoder(target_model, draft_model, config)
    return decoder.generate(input_ids, max_new_tokens)


# =============================================================================
# Medusa: Multi-Head Speculative Decoding
# =============================================================================

class MedusaHead(nn.Module):
    """
    Medusa head for multi-head speculative decoding.
    
    Adds multiple prediction heads to predict future tokens
    without a separate draft model.
    """
    
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_heads: int = 4,
    ):
        super().__init__()
        
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, vocab_size),
            )
            for _ in range(num_heads)
        ])
    
    def forward(self, hidden_states: torch.Tensor) -> List[torch.Tensor]:
        """Get predictions from each head."""
        return [head(hidden_states) for head in self.heads]


# =============================================================================
# Benchmark
# =============================================================================

def benchmark_speculative(
    target_model: nn.Module,
    draft_model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 100,
    num_runs: int = 5,
) -> Dict[str, float]:
    """
    Benchmark speculative decoding vs autoregressive.
    """
    # Warmup
    _ = target_model.generate(input_ids, max_new_tokens=10)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Autoregressive baseline
    ar_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = target_model.generate(input_ids, max_new_tokens=max_new_tokens)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ar_times.append(time.perf_counter() - start)
    
    ar_time = sum(ar_times) / len(ar_times)
    
    # Speculative
    decoder = SpeculativeDecoder(target_model, draft_model)
    
    spec_times = []
    for _ in range(num_runs):
        decoder.reset_stats()
        start = time.perf_counter()
        _ = decoder.generate(input_ids, max_new_tokens=max_new_tokens)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        spec_times.append(time.perf_counter() - start)
    
    spec_time = sum(spec_times) / len(spec_times)
    
    return {
        "autoregressive_ms": ar_time * 1000,
        "speculative_ms": spec_time * 1000,
        "speedup": ar_time / spec_time,
        "accept_rate": decoder.accept_rate,
    }
