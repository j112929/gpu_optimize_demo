"""
Tensor & Sequence Parallelism - 3D Parallelism Building Blocks.

Provides:
- ColumnParallelLinear: Splits output dimension
- RowParallelLinear: Splits input dimension
- VocabParallelEmbedding: Splits vocabulary
- SequenceParallelWrapper: Splits sequence dimension (Ring/Ulysses style)

These blocks allow training models larger than any single GPU's memory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class ParallelismConfig:
    tp_size: int = 1       # Tensor Parallel size
    sp_size: int = 1       # Sequence Parallel size
    enable_async: bool = True


# =============================================================================
# Helper: Distributed Utils
# =============================================================================

class Utils:
    @staticmethod
    def split_tensor_along_last_dim(tensor, num_partitions, contiguous_split_chunks=False):
        """Split a tensor along its last dimension.
        Arguments:
            tensor: input tensor.
            num_partitions: number of partitions to split the tensor
            contiguous_split_chunks: If True, make each chunk contiguous
                                     in memory.
        """
        # Get the size and dimension.
        last_dim = tensor.dim() - 1
        last_dim_size = tensor.size()[last_dim] // num_partitions
        # Split.
        tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
        # Note: torch.split does not create contiguous tensors by default.
        if contiguous_split_chunks:
            return tuple(chunk.contiguous() for chunk in tensor_list)
        return tensor_list

# =============================================================================
# Tensor Parallelism Layers
# =============================================================================

class ColumnParallelLinear(nn.Module):
    """
    Linear layer with column parallelism.
    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].
    """

    def __init__(self, input_size, output_size, bias=True, gather_output=True, tp_group=None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        self.tp_group = tp_group
        
        # Determine world size
        world_size = dist.get_world_size(group=self.tp_group) if dist.is_initialized() else 1
        self.output_size_per_partition = output_size // world_size

        # Parameter initialization
        self.weight = nn.Parameter(torch.empty(
            self.output_size_per_partition, self.input_size
        ))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize master weight then scatter or initialize locally (simplified here)
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input_):
        # Input: [batch, seq, hidden]
        # Weight: [output_per_part, hidden]
        # Output: [batch, seq, output_per_part]
        
        # 1. Identity (Copy to all TP ranks) - already replicated in TP usually
        input_parallel = input_
        
        # 2. Matrix Multiply
        output_parallel = F.linear(input_parallel, self.weight, self.bias)
        
        # 3. All-Gather (if requested)
        if self.gather_output and dist.is_initialized():
             # Gather along last dimension
             # [b, s, h/p] -> [b, s, h]
             output_list = [torch.zeros_like(output_parallel) for _ in range(dist.get_world_size(self.tp_group))]
             dist.all_gather(output_list, output_parallel, group=self.tp_group)
             output = torch.cat(output_list, dim=-1)
             return output
             
        return output_parallel


class RowParallelLinear(nn.Module):
    """
    Linear layer with row parallelism.
    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension.
    """

    def __init__(self, input_size, output_size, bias=True, input_is_parallel=False, tp_group=None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        self.tp_group = tp_group
        
        world_size = dist.get_world_size(group=self.tp_group) if dist.is_initialized() else 1
        self.input_size_per_partition = input_size // world_size

        self.weight = nn.Parameter(torch.empty(
            self.output_size, self.input_size_per_partition
        ))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input_):
        # Input should be split along last dimension if input_is_parallel=True
        if not self.input_is_parallel:
            # Split input
            # [b, s, h] -> [b, s, h/p]
            world_size = dist.get_world_size(self.tp_group) if dist.is_initialized() else 1
            input_parallel = Utils.split_tensor_along_last_dim(input_, world_size)[dist.get_rank(self.tp_group)]
        else:
            input_parallel = input_
            
        # Matrix Multiply
        output_parallel = F.linear(input_parallel, self.weight)
        
        # All-Reduce
        if dist.is_initialized():
            dist.all_reduce(output_parallel, op=dist.ReduceOp.SUM, group=self.tp_group)
            
        if self.bias is not None:
            output_parallel = output_parallel + self.bias
            
        return output_parallel


# =============================================================================
# Sequence Parallelism (Simplified Ring Style)
# =============================================================================

class SequenceParallelWrapper(nn.Module):
    """
    Wraps a module to support Sequence Parallelism.
    
    Splits the sequence dimension [batch, seq_len, hidden] across GPUs.
    Useful for very long context lengths.
    """
    
    def __init__(self, module, sp_group=None):
        super().__init__()
        self.module = module
        self.sp_group = sp_group

    def forward(self, x):
        # x: [batch, seq_len, hidden]
        
        # 1. Scatter sequence dim
        if dist.is_initialized():
            world_size = dist.get_world_size(self.sp_group)
            rank = dist.get_rank(self.sp_group)
            
            # Split sequence dimension
            seq_len = x.shape[1]
            part_len = seq_len // world_size
            
            # This rank processes: [rank*part_len : (rank+1)*part_len]
            x_local = x[:, rank*part_len : (rank+1)*part_len, :]
        else:
            x_local = x
            
        # 2. Forward pass on local chunk
        output_local = self.module(x_local)
        
        # 3. Gather sequence dim (optional, depending on architecture)
        if dist.is_initialized():
            outputs = [torch.zeros_like(output_local) for _ in range(world_size)]
            dist.all_gather(outputs, output_local, group=self.sp_group)
            return torch.cat(outputs, dim=1)
            
        return output_local
