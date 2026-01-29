"""
Tensor Parallel Inference Worker.

Implements a distributed worker for multi-GPU inference.
Uses "Single Program Multiple Data" (SPMD) paradigm where:
- Rank 0 (Driver) sends input tokens via Broadcast
- All Ranks (Workers) execute model forward pass in parallel
- Tensor Parallel layers handle AllReduce/AllGather internally
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from src.training.parallelism import ColumnParallelLinear, RowParallelLinear
from src.inference.kv_cache import PagedKVCache

# =============================================================================
# Messages
# =============================================================================

@dataclass
class InferenceCommand:
    """Command sent from Driver to Workers."""
    op: str  # "init", "forward", "step", "stop"
    data: Any = None


# =============================================================================
# TP Model Wrapper (Mock for Demo)
# =============================================================================

class ParallelTransformerLayer(torch.nn.Module):
    """
    A single Tensor Parallel Transformer Layer.
    MLP: ColumnParallel(fc1) -> Gelu -> RowParallel(fc2)
    """
    def __init__(self, hidden_size: int, tp_group: Any):
        super().__init__()
        self.tp_group = tp_group
        
        # MLP: 4x expansion
        intermediate_size = hidden_size * 4
        
        # [H, 4H/P]
        self.fc1 = ColumnParallelLinear(
            hidden_size, intermediate_size, gather_output=False, tp_group=tp_group
        )
        # [4H/P, H]
        self.fc2 = RowParallelLinear(
            intermediate_size, hidden_size, input_is_parallel=True, tp_group=tp_group
        )
        self.act = torch.nn.GELU()
        
    def forward(self, x):
        # x: [Batch, Seq, Hidden] (Replicated on all ranks)
        
        # 1. FC1 (Split Col) -> [Batch, Seq, 4H/P] (Per rank)
        h = self.fc1(x)
        h = self.act(h)
        
        # 2. FC2 (Split Row) -> [Batch, Seq, H] (AllReduced, Replicated)
        output = self.fc2(h)
        
        # Residual
        return x + output


# =============================================================================
# Worker Implementation
# =============================================================================

class TPInferenceWorker:
    """
    Worker process running on a single GPU.
    """
    def __init__(self, rank: int, world_size: int, hidden_size: int = 1024):
        self.rank = rank
        self.world_size = world_size
        self.hidden_size = hidden_size
        self.device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")
        self.tp_group = None
        self.model = None
        self.kv_cache = None

    def init_process_group(self):
        """Initialize distributed environment."""
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        # Use gloo for CPU demo, nccl for GPU
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend, rank=self.rank, world_size=self.world_size)
        
        # Create TP group (all ranks)
        self.tp_group = dist.new_group(list(range(self.world_size)))
        print(f"Worker {self.rank}: Process group initialized.")

    def init_model(self):
        """Initialize TP model shard."""
        torch.manual_seed(42) # Ensure shared weights are same init
        self.model = ParallelTransformerLayer(self.hidden_size, self.tp_group).to(self.device)
        self.kv_cache = PagedKVCache(100, 16, self.hidden_size, 1) # simple cache
        print(f"Worker {self.rank}: Model initialized.")

    def run_loop(self):
        """Main event loop."""
        self.init_process_group()
        self.init_model()
        
        while True:
            # 1. Receive Command Broadcast from Rank 0
            # For simplicity in demo, we pickle obj via a broadcast list
            # In optimized engines, we use custom C++ layout or just broadcast tensor data
            
            cmd_list = [None]
            dist.broadcast_object_list(cmd_list, src=0)
            cmd: InferenceCommand = cmd_list[0]
            
            if cmd.op == "stop":
                break
            elif cmd.op == "forward":
                self.step(cmd.data)
        
        print(f"Worker {self.rank}: Stopping.")
        dist.destroy_process_group()

    def step(self, input_ids: torch.Tensor):
        """Execute one forward step."""
        input_ids = input_ids.to(self.device)
        
        # Embedding (Mock: assume we have embeddings)
        # In real TP, embedding is also Row/Col parallel
        x = torch.randn(
            input_ids.shape[0], input_ids.shape[1], self.hidden_size, 
            device=self.device
        )
        
        # TP Forward
        with torch.no_grad():
            output = self.model(x)
            
        # Log only on rank 0
        if self.rank == 0:
            # print(f"Rank 0 Output mean: {output.mean().item()}")
            pass


# =============================================================================
# Engine (Driver)
# =============================================================================

class ParallelInferenceEngine:
    """
    Orchestrates TP Workers.
    Runs on the main process (which acts as Rank 0 usually, or manages processes).
    """
    def __init__(self, world_size: int = 2):
        self.world_size = world_size
        self.processes = []
        
    def start(self):
        """Start worker processes."""
        mp.set_start_method("spawn", force=True)
        
        # Spawn workers
        # Note: Rank 0 is also a worker in this simplified SPMD design
        for rank in range(self.world_size):
            p = mp.Process(target=self._worker_entry, args=(rank, self.world_size))
            p.start()
            self.processes.append(p)
            
    @staticmethod
    def _worker_entry(rank, world_size):
        worker = TPInferenceWorker(rank, world_size)
        if rank == 0:
            # Rank 0 acts as Controller + Worker 0
            # It drives the loop by broadcasting to itself and others
            worker.init_process_group()
            worker.init_model()
            
            # Simple interaction loop simulation
            try:
                # 1. Forward
                print("\n[Controller] Sending Forward Command...")
                input_data = torch.randint(0, 100, (1, 10))
                cmd = InferenceCommand("forward", input_data)
                
                # Broadcast payload
                dist.broadcast_object_list([cmd], src=0)
                
                # Execute local work (as Worker 0)
                worker.step(input_data)
                
                time.sleep(1)
                
                # 2. Stop
                print("[Controller] Sending Stop Command...")
                stop_cmd = InferenceCommand("stop")
                dist.broadcast_object_list([stop_cmd], src=0)
                
            except Exception as e:
                print(f"Controller Error: {e}")
            finally:
                dist.destroy_process_group()
        else:
            # Other ranks just listen
            worker.run_loop()

    def join(self):
        for p in self.processes:
            p.join()


# =============================================================================
# Standalone Test
# =============================================================================

def test_tp_inference():
    print("Initializing TP Engine with World Size 2...")
    engine = ParallelInferenceEngine(world_size=2)
    engine.start()
    engine.join()
    print("TP Inference Test Complete.")

if __name__ == "__main__":
    test_tp_inference()
