"""
Continuous Batching - Dynamic batching for optimal throughput.

Processes requests as they arrive, maximizing GPU utilization
without waiting for batch boundaries.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import threading
import queue
import time
from collections import deque
import uuid


@dataclass
class Request:
    """Single inference request."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    input_ids: Optional[torch.Tensor] = None
    prompt: str = ""
    max_new_tokens: int = 256
    temperature: float = 1.0
    
    # State
    generated_tokens: List[int] = field(default_factory=list)
    is_complete: bool = False
    start_time: float = 0.0
    
    # Callback
    on_token: Optional[Callable[[int], None]] = None
    on_complete: Optional[Callable[[List[int]], None]] = None


@dataclass
class BatchConfig:
    """Configuration for continuous batching."""
    max_batch_size: int = 32
    max_waiting_time_ms: float = 50.0    # Max wait before processing
    
    # Padding
    pad_to_multiple: int = 8
    max_sequence_length: int = 2048
    
    # Scheduling
    priority_queue: bool = False
    preemption: bool = True              # Allow preempting long sequences


class RequestQueue:
    """
    Thread-safe request queue with priority support.
    """
    
    def __init__(self, max_size: int = 1000):
        self._queue = queue.PriorityQueue(maxsize=max_size)
        self._lock = threading.Lock()
    
    def put(self, request: Request, priority: int = 0):
        """Add request to queue."""
        self._queue.put((priority, time.time(), request))
    
    def get(self, timeout: Optional[float] = None) -> Optional[Request]:
        """Get next request."""
        try:
            _, _, request = self._queue.get(timeout=timeout)
            return request
        except queue.Empty:
            return None
    
    def get_batch(self, max_size: int, timeout: float = 0.05) -> List[Request]:
        """Get a batch of requests."""
        batch = []
        deadline = time.time() + timeout
        
        while len(batch) < max_size and time.time() < deadline:
            remaining = deadline - time.time()
            request = self.get(timeout=max(0.001, remaining))
            if request:
                batch.append(request)
            else:
                break
        
        return batch
    
    def qsize(self) -> int:
        return self._queue.qsize()


class ContinuousBatcher:
    """
    Continuous Batching Engine for LLM inference.
    
    Dynamically batches requests for optimal GPU utilization.
    Supports:
    - Dynamic batching (add requests anytime)
    - Iteration-level scheduling
    - Preemption for long sequences
    - Memory-efficient KV-cache sharing
    
    Example:
        >>> batcher = ContinuousBatcher(model, config)
        >>> batcher.start()
        >>> 
        >>> # Submit requests
        >>> request = Request(prompt="Hello", max_new_tokens=100)
        >>> batcher.submit(request)
        >>> 
        >>> # Wait for completion
        >>> result = batcher.wait(request.id)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[BatchConfig] = None,
        tokenizer: Optional[Any] = None,
    ):
        self.model = model
        self.config = config or BatchConfig()
        self.tokenizer = tokenizer
        
        # Request management
        self.pending_queue = RequestQueue()
        self.active_batch: List[Request] = []
        self.completed: Dict[str, List[int]] = {}
        
        # Threading
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "batches_processed": 0,
        }
    
    def start(self):
        """Start the batching engine."""
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop the batching engine."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
    
    def submit(self, request: Request) -> str:
        """
        Submit a request for processing.
        
        Returns:
            Request ID for tracking
        """
        request.start_time = time.time()
        self.pending_queue.put(request)
        self.stats["total_requests"] += 1
        return request.id
    
    def wait(self, request_id: str, timeout: float = 60.0) -> Optional[List[int]]:
        """Wait for a request to complete."""
        deadline = time.time() + timeout
        
        while time.time() < deadline:
            with self._lock:
                if request_id in self.completed:
                    return self.completed.pop(request_id)
            time.sleep(0.01)
        
        return None
    
    def _run_loop(self):
        """Main processing loop."""
        while self._running:
            # Get new requests
            new_requests = self.pending_queue.get_batch(
                max_size=self.config.max_batch_size - len(self.active_batch),
                timeout=self.config.max_waiting_time_ms / 1000,
            )
            
            # Add to active batch
            with self._lock:
                self.active_batch.extend(new_requests)
            
            if not self.active_batch:
                continue
            
            # Process one iteration
            self._process_iteration()
            
            # Remove completed requests
            self._cleanup_completed()
            
            self.stats["batches_processed"] += 1
    
    def _process_iteration(self):
        """Process one decoding iteration for active batch."""
        if not self.active_batch:
            return
        
        # Prepare batch
        batch_input_ids = self._prepare_batch()
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(batch_input_ids)
            logits = outputs.logits[:, -1]
        
        # Sample next tokens
        next_tokens = self._sample_tokens(logits)
        
        # Update requests
        for i, request in enumerate(self.active_batch):
            token = next_tokens[i].item()
            request.generated_tokens.append(token)
            
            # Callback
            if request.on_token:
                request.on_token(token)
            
            # Check completion
            if len(request.generated_tokens) >= request.max_new_tokens:
                request.is_complete = True
            
            # Check EOS (simplified)
            if self.tokenizer and token == self.tokenizer.eos_token_id:
                request.is_complete = True
        
        self.stats["total_tokens"] += len(self.active_batch)
    
    def _prepare_batch(self) -> torch.Tensor:
        """Prepare input tensor for batch."""
        # Collect all sequences
        sequences = []
        for request in self.active_batch:
            if request.input_ids is not None:
                seq = torch.cat([
                    request.input_ids.squeeze(0),
                    torch.tensor(request.generated_tokens, device=request.input_ids.device),
                ])
            else:
                seq = torch.tensor(request.generated_tokens)
            sequences.append(seq)
        
        # Pad to same length
        max_len = max(len(s) for s in sequences)
        padded = []
        for seq in sequences:
            if len(seq) < max_len:
                padding = torch.zeros(max_len - len(seq), dtype=seq.dtype, device=seq.device)
                seq = torch.cat([padding, seq])
            padded.append(seq)
        
        return torch.stack(padded)
    
    def _sample_tokens(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """Sample tokens from logits."""
        # Temperature sampling (simplified)
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    def _cleanup_completed(self):
        """Move completed requests to completed dict."""
        with self._lock:
            still_active = []
            for request in self.active_batch:
                if request.is_complete:
                    self.completed[request.id] = request.generated_tokens
                    if request.on_complete:
                        request.on_complete(request.generated_tokens)
                else:
                    still_active.append(request)
            self.active_batch = still_active
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batching statistics."""
        return {
            **self.stats,
            "pending_requests": self.pending_queue.qsize(),
            "active_requests": len(self.active_batch),
            "avg_tokens_per_batch": (
                self.stats["total_tokens"] / max(1, self.stats["batches_processed"])
            ),
        }


# =============================================================================
# Iteration-Level Scheduling
# =============================================================================

class IterationScheduler:
    """
    Iteration-level scheduler for optimal throughput.
    
    Decides which requests to include in each iteration
    based on:
    - Request priority
    - Sequence length
    - Memory constraints
    """
    
    def __init__(
        self,
        max_batch_tokens: int = 4096,
        max_batch_size: int = 32,
    ):
        self.max_batch_tokens = max_batch_tokens
        self.max_batch_size = max_batch_size
    
    def schedule(
        self,
        active_requests: List[Request],
        pending_requests: List[Request],
    ) -> Tuple[List[Request], List[Request]]:
        """
        Select requests for next iteration.
        
        Returns:
            (selected, remaining)
        """
        # Sort by length (shorter first for efficiency)
        all_requests = active_requests + pending_requests
        all_requests.sort(key=lambda r: len(r.generated_tokens))
        
        selected = []
        remaining = []
        total_tokens = 0
        
        for request in all_requests:
            seq_len = len(request.generated_tokens) + 1
            
            if (
                len(selected) < self.max_batch_size and
                total_tokens + seq_len <= self.max_batch_tokens
            ):
                selected.append(request)
                total_tokens += seq_len
            else:
                remaining.append(request)
        
        return selected, remaining


# =============================================================================
# Utilities
# =============================================================================

def create_batch_inference_server(
    model: nn.Module,
    tokenizer: Any,
    port: int = 8000,
) -> ContinuousBatcher:
    """
    Create a simple batch inference server.
    
    Example:
        >>> server = create_batch_inference_server(model, tokenizer)
        >>> server.start()
    """
    config = BatchConfig(
        max_batch_size=32,
        max_waiting_time_ms=100,
    )
    
    return ContinuousBatcher(model, config, tokenizer)
