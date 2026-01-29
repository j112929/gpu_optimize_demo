"""
FP8 Attention Kernel using Triton.

Optimized for H100/MI300+ hardware with FP8 tensor core support.
Implements FlashDecoding style attention for inference.
"""

import torch
import triton
import triton.language as tl
from typing import Optional, Tuple

def is_fp8_supported():
    """Check if FP8 is supported on this hardware."""
    try:
        if not torch.cuda.is_available():
            return False
        gpu_props = torch.cuda.get_device_properties(0)
        # Hopper (sm_90) or newer required for full FP8 support
        return gpu_props.major >= 9
    except:
        return False

# =============================================================================
# Triton FP8 Flash Decoding Kernel
# =============================================================================

@triton.jit
def _flash_decode_fp8_kernel(
    Q, K, V, Out,
    L, # Logsumexp storage for backward or verification
    q_scale, k_scale, v_scale, # Quantization scales
    stride_qz, stride_qh, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_on, stride_ok,
    Z, H, N_CTX, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Flash Decoding Kernel for FP8.
    
    Q: [Z, H, 1, D] - FP8
    K: [Z, H, N, D] - FP8
    V: [Z, H, N, D] - FP8
    """
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    off_z = off_hz // H
    off_h = off_hz % H
    
    # Q offsets
    q_offset = off_z * stride_qz + off_h * stride_qh
    Q_ptr = Q + q_offset + tl.arange(0, BLOCK_DMODEL) * stride_qk
    
    # Load Q and its scale
    # Triton loads FP8 as int8, need to convert to fp32 for compute (simulated here)
    # On H100, we would use proper fp8 matmul intrinsics (WGMMA)
    q_val = tl.load(Q_ptr)
    qs = tl.load(q_scale)
    q = q_val * qs # Dequantize on the fly for calculating softmax scores
    
    # Initialize Accumulators
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    l_i = tl.zeros([1], dtype=tl.float32) - float('inf')
    
    # KV Loop
    for start_n in range(0, N_CTX, BLOCK_N):
        # K offsets
        k_offset = off_z * stride_kz + off_h * stride_kh + \
                   (start_n + tl.arange(0, BLOCK_N)[:, None]) * stride_kn + \
                   tl.arange(0, BLOCK_DMODEL)[None, :] * stride_kk
                   
        # Load K (FP8)
        k_val = tl.load(K + k_offset) 
        ks = tl.load(k_scale)
        k = k_val * ks
        
        # QK^T
        qk = tl.sum(q[None, :] * k, axis=1) # [BLOCK_N]
        qk *= 1.44269504 # log2(e) for exp2
        
        # Softmax logic (Online scale adjustment)
        m_i = tl.max(qk, 0)
        alpha = tl.exp2(m_i - l_i)
        l_i = tl.maximum(l_i, m_i) + tl.log2(tl.sum(tl.exp2(qk - m_i)))
        
        # Update accumulator
        acc *= alpha
        
        # Load V (FP8)
        v_offset = off_z * stride_vz + off_h * stride_vh + \
                   (start_n + tl.arange(0, BLOCK_N)[:, None]) * stride_vn + \
                   tl.arange(0, BLOCK_DMODEL)[None, :] * stride_vk
        
        v_val = tl.load(V + v_offset)
        vs = tl.load(v_scale)
        v = v_val * vs
        
        # Attention * V
        p = tl.exp2(qk - m_i)
        acc += tl.sum(p[:, None] * v, axis=0)
        
    # Write output
    out_offset = off_z * stride_oz + off_h * stride_oh + \
                 tl.arange(0, BLOCK_DMODEL) * stride_ok
                 
    tl.store(Out + out_offset, acc / tl.exp2(l_i))


def flash_decode_fp8_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_scale: float = 1.0,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
):
    """
    Run FP8 Flash Decoding using Triton.
    
    Args:
        q: [Batch, Heads, 1, Dim] (FP8)
        k: [Batch, Heads, SeqLen, Dim] (FP8)
        v: [Batch, Heads, SeqLen, Dim] (FP8)
    """
    batch, heads, _, dim = q.shape
    seq_len = k.shape[2]
    
    # Checks
    assert dim in {64, 128}, "Only dim 64/128 supported for now"
    
    # Allocate output
    out = torch.empty((batch, heads, 1, dim), device=q.device, dtype=torch.float16)
    l_sum = torch.empty((batch, heads), device=q.device, dtype=torch.float32)
    
    # Convert scales to tensors
    qs = torch.tensor([q_scale], device=q.device, dtype=torch.float32)
    ks = torch.tensor([k_scale], device=q.device, dtype=torch.float32)
    vs = torch.tensor([v_scale], device=q.device, dtype=torch.float32)
    
    # Config
    BLOCK_N = 128
    grid = (1, batch * heads)
    
    _flash_decode_fp8_kernel[grid](
        q, k, v, out, l_sum,
        qs, ks, vs,
        q.stride(0), q.stride(1), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        batch, heads, seq_len,
        BLOCK_DMODEL=dim,
        BLOCK_N=BLOCK_N,
    )
    
    return out


# =============================================================================
# FlashInfer Integration (Wrapper)
# =============================================================================

class FlashInferWrapper:
    """
    Wrapper for FlashInfer library (SOTA kernel library).
    Falls back to Triton or PyTorch implementation if not available.
    """
    
    def __init__(self, workspace_buffer: Optional[torch.Tensor] = None):
        self.backend = "none"
        try:
            import flashinfer
            self.flashinfer = flashinfer
            self.backend = "flashinfer"
            # Initialize Workspace if needed
            if workspace_buffer is None:
                # 128MB default workspace
                self.workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
            else:
                self.workspace = workspace_buffer
        except ImportError:
            pass
            
    def decode(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        kv_layout: str = "NHD", # NHD or HND
    ):
        """
        Run batch decode.
        
        Args:
            q: [Batch, NumQHeads, HeadDim]
            k_cache: [Batch, SeqLen, NumKVHeads, HeadDim]
            v_cache: [Batch, SeqLen, NumKVHeads, HeadDim]
        """
        if self.backend == "flashinfer":
            return self.flashinfer.batch_decode_with_padded_kv_cache(
                q, k_cache, v_cache, kv_layout=kv_layout
            )
        else:
            # Fallback to PyTorch / Triton
            # Simplified SDPA
            if kv_layout == "NHD":
                k = k_cache.transpose(1, 2) # [B, H, S, D]
                v = v_cache.transpose(1, 2)
                q = q.unsqueeze(2)          # [B, H, 1, D]
            
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
                out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            return out.squeeze(2)

    def decode_fp8(self, q, k, v, q_scale, k_scale, v_scale):
        """FP8 Decode dispatch."""
        if self.backend == "flashinfer" and hasattr(self.flashinfer, "batch_decode_with_fp8_kv_cache"):
             # Hypothetical API, FlashInfer is evolving fast
             pass
             
        # Use our local Triton kernel
        if q.dtype == torch.float8_e4m3fn or is_fp8_supported():
             return flash_decode_fp8_triton(q, k, v, q_scale, k_scale, v_scale)
        else:
             # Dequantize and run standard
             # Just for functional correctness in demo
             q_f = q.float() * q_scale
             k_f = k.float() * k_scale
             v_f = v.float() * v_scale
             return self.decode(q_f.view(q.shape[0], q.shape[1], -1), 
                                k_f.permute(0, 2, 1, 3), 
                                v_f.permute(0, 2, 1, 3), kv_layout="NHD")
