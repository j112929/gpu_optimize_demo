"""
Advanced Inference Optimizations Demo

Demonstrates cutting-edge inference techniques:
1. Speculative Decoding (Draft Model speedup)
2. Prefix Caching (Radix Attention for reusable prompts)
3. Chunked Prefill (Split-wise inference optimization)
4. FP8 KV Cache (Memory efficiency)

Usage:
    python examples/advanced_inference.py --mode speculative
    python examples/advanced_inference.py --mode interactive  # Radix Attention
    python examples/advanced_inference.py --mode chunking
    python examples/advanced_inference.py --mode fp8
    python examples/advanced_inference.py --mode all
"""

import argparse
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import time
import sys
import os

# Add src to path
sys.path.append(os.getcwd())

from src.inference.speculative import SpeculativeDecoder, benchmark_speculative
from src.inference.kv_cache import PrefixCacheManager, FP8KVCache
from src.inference.batching import Request, ChunkedPrefillScheduler

# =============================================================================
# Helper: Simple Autoregressive Logic
# =============================================================================

class SimpleModel(nn.Module):
    def __init__(self, vocab_size=1000, dim=256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([nn.Linear(dim, dim) for _ in range(2)])
        self.head = nn.Linear(dim, vocab_size)
        
    def forward(self, x):
        h = self.emb(x)
        for block in self.blocks:
            h = block(h) + h
        logits = self.head(h)
        from collections import namedtuple
        Output = namedtuple('Output', ['logits'])
        return Output(logits=logits)
        
    def generate(self, input_ids, max_new_tokens=10):
        # Used for baseline benchmarking
        curr = input_ids
        for _ in range(max_new_tokens):
            out = self(curr)
            next_token = torch.argmax(out.logits[:, -1], dim=-1, keepdim=True)
            curr = torch.cat([curr, next_token], dim=1)
        return curr


# =============================================================================
# 1. Speculative Decoding
# =============================================================================

def speculative_example():
    print("\n" + "="*60)
    print("🚀 Speculative Decoding Demo")
    print("="*60)
    
    # Setup mock models
    # Target model is "large" (simulated delay), draft is "small"
    vocab_size = 1000
    target_model = SimpleModel(vocab_size, 1024)
    draft_model = SimpleModel(vocab_size, 256) # 4x smaller params implies faster
    
    # Input
    input_ids = torch.randint(0, vocab_size, (1, 10))
    
    print("Benchmarking...")
    stats = benchmark_speculative(
        target_model, 
        draft_model, 
        input_ids, 
        max_new_tokens=50,
        num_runs=3
    )
    
    print("\nResults:")
    print(f"  Autoregressive Time: {stats['autoregressive_ms']:.2f} ms")
    print(f"  Speculative Time:    {stats['speculative_ms']:.2f} ms")
    print(f"  Speedup:             {stats['speedup']:.2f}x")
    print(f"  Acceptance Rate:     {stats['accept_rate']:.2%}")
    print("\nNote: Speedup is simulated. Real gains depend on model size gap and hardware.")


# =============================================================================
# 2. Prefix Caching (Radix Attention)
# =============================================================================

def prefix_caching_example():
    print("\n" + "="*60)
    print("🧠 Radix Attention (Prefix Caching) Demo")
    print("="*60)
    
    manager = PrefixCacheManager(block_size=4, max_blocks=100)
    
    # Scenario: Multi-turn Chat
    system_prompt = [1, 2, 3, 4, 5, 6, 7, 8] # "You are a helpful assistant..."
    
    # Request 1: User asks question
    user_q1 = [101, 102]
    full_req1 = system_prompt + user_q1
    
    print(f"Request 1: {full_req1}")
    hits, blocks = manager.check_cache(full_req1)
    if hits == 0:
        print("  Cache Miss. Processing full prompt...")
        # Simulate processing allocation
        new_blocks = manager.paged_cache.allocate(len(full_req1) // 4 + 1)
        manager.cache_request(full_req1, new_blocks)
        print(f"  Cached {len(full_req1)} tokens in {len(new_blocks)} blocks.")
    
    # Request 2: New user question with SAME system prompt
    user_q2 = [201, 202]
    full_req2 = system_prompt + user_q2
    
    print(f"\nRequest 2: {full_req2} (Shares system prompt)")
    hits, blocks = manager.check_cache(full_req2)
    
    if hits > 0:
        print(f"  ✅ Cache Hit! Matched prefix length: {hits}")
        print(f"  Reuse ratio: {hits / len(full_req2):.1%}")
        print("  Only need to compute: ", full_req2[hits:])
    else:
        print("  ❌ Cache Miss (Unexpected)")


# =============================================================================
# 3. Chunked Prefill
# =============================================================================

def chunking_example():
    print("\n" + "="*60)
    print("🧩 Chunked Prefill Demo")
    print("="*60)
    
    # Create scheduler
    scheduler = ChunkedPrefillScheduler(chunk_size=10)
    
    # Create requests
    # 1. Long prefill request
    req_long = Request(input_ids=torch.zeros(1, 100), prompt="Long Document")
    # 2. Short decode request (already generating)
    req_decode = Request(generated_tokens=[1, 2, 3], max_new_tokens=10)
    
    active_requests = [req_long, req_decode]
    
    # Simulate step budget of 15 tokens (GPU capacity)
    budget = 15
    print(f"Step Budget: {budget} tokens\n")
    
    print(f"Scheduling step 1...")
    selected, chunks = scheduler.schedule_step(active_requests, budget)
    
    for req, chunk_meta in zip(selected, chunks):
        req_id, start, length = chunk_meta
        type_str = "Decode" if start == -1 else f"Prefill Chunk ({start}-{start+length})"
        print(f"  Request {req_id[:4]} ({req.prompt if req.prompt else 'Decode'}): {type_str} - {length} tokens")
    
    # Validate result
    # Should see Decode (1 token) + Prefill Chunk (14 tokens) = 15 total
    total_scheduled = sum(c[2] for c in chunks)
    print(f"  Total Scheduled: {total_scheduled} / {budget}")
    
    print("\nScheduling step 2...")
    selected, chunks = scheduler.schedule_step(active_requests, budget)
    for req, chunk_meta in zip(selected, chunks):
        req_id, start, length = chunk_meta
        type_str = "Decode" if start == -1 else f"Prefill Chunk ({start}-{start+length})"
        print(f"  Request {req_id[:4]}: {type_str} - {length} tokens")


# =============================================================================
# 4. FP8 KV Cache & Triton Kernel
# =============================================================================

def fp8_example():
    print("\n" + "="*60)
    print("💾 FP8 KV Cache & Triton Kernel Demo")
    print("="*60)
    
    # 1. FP8 KV Cache Awareness
    try:
        from src.inference.kv_cache import FP8KVCache
        cache = FP8KVCache(num_blocks=1000, block_size=16, hidden_dim=4096, num_layers=32)
        
        # Simulate Allocation
        cache.allocate(100)
        
        mem_mb = cache.get_memory_usage()
        # Comparison
        fp16_mem_mb = 1000 * 16 * 4096 * 2 * 2 / (1024**2) # FP16 = 2 bytes
        
        print(f"Allocated 1000 blocks")
        print(f"  FP16 Memory (standard): {fp16_mem_mb:.2f} MB")
        print(f"  FP8 Memory (quantized): {mem_mb:.2f} MB")
        print(f"  Savings: {fp16_mem_mb / mem_mb:.1f}x")
    except ImportError:
        print("FP8KVCache not found")

    # 2. Triton FP8 Kernel Demo
    print("\n--- Triton FP8 Flash Decoding ---")
    try:
        from src.triton_kernels import flash_decode_fp8_triton, FlashInferWrapper
        
        # Check support
        if not torch.cuda.is_available():
            print("  Skipping kernel execution (No CUDA)")
            return

        # Prepare dummy data
        batch, heads, seq, dim = 2, 8, 128, 64
        q = torch.randn(batch, heads, 1, dim, device="cuda").half() # Simulate FP8 by casting if needed
        k = torch.randn(batch, heads, seq, dim, device="cuda").half()
        v = torch.randn(batch, heads, seq, dim, device="cuda").half()
        
        print(f"  Input: Q={q.shape}, K={k.shape}, kv_len={seq}")
        
        # Run Kernel (Wrapper handles fallback if not H100)
        wrapper = FlashInferWrapper()
        
        print("  Running FlashInferWrapper (Dispatching to FlashInfer or Triton)...")
        # Simulate scales
        out = wrapper.decode_fp8(q, k, v, 1.0, 1.0, 1.0)
        
        print(f"  Output shape: {out.shape}")
        print("  ✅ Kernel execution successful (or fallback used)")
        
    except ImportError as e:
        print(f"  Error importing Triton kernels: {e}")
    except Exception as e:
         print(f"  Error running kernel: {e}")


# =============================================================================
# 5. Tensor Parallel Inference
# =============================================================================

def tp_example():
    print("\n" + "="*60)
    print("🤝 Tensor Parallel Inference Demo")
    print("="*60)
    
    try:
        from src.inference.tp_worker import ParallelInferenceEngine
        
        print("Starting TP Engine (World Size = 2)...")
        print("Note: This spawns 2 worker processes.")
        
        # Check if we can run this (need multiprocessing support)
        import torch.multiprocessing as mp
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass # context already set
            
        engine = ParallelInferenceEngine(world_size=2)
        engine.start()
        engine.join()
        
        print("\n✅ TP Demo Complete")
        
    except Exception as e:
        print(f"TP Demo Error: {e}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all", choices=["speculative", "interactive", "chunking", "fp8", "tp", "all"])
    args = parser.parse_args()
    
    examples = {
        "speculative": speculative_example,
        "interactive": prefix_caching_example,
        "chunking": chunking_example,
        "fp8": fp8_example,
        "tp": tp_example,
    }
    
    if args.mode == "all":
        print("Running ALL examples (TP might fail if mixed with others due to MP context)...")
        for name, func in examples.items():
            func()
    else:
        examples[args.mode]()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
