"""
FlagGems backend for FlashAttention
This module provides adapters to use FlagGems' Triton-based flash attention
implementation as a drop-in replacement for the CUDA version.
"""

import logging
import sys
import os
from typing import Optional, Tuple

import torch

# Add FlagGems to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../FlagGems/src'))

try:
    from flag_gems.ops.attention import flash_attention_forward, flash_attn_varlen_func
    from flag_gems.ops.flash_api import mha_fwd, mha_varlan_fwd
    FLAGGEMS_AVAILABLE = True
except ImportError as e:
    FLAGGEMS_AVAILABLE = False
    print(f"Warning: Could not import FlagGems: {e}")

logger = logging.getLogger(__name__)


def _flaggems_flash_attn_forward(
    q, k, v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size_left=-1,
    window_size_right=-1,
    softcap=0.0,
    alibi_slopes=None,
    return_softmax=False,
):
    """
    Adapter for flash_attn_func using FlagGems backend.
    
    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k: (batch_size, seqlen, nheads_k, headdim)
        v: (batch_size, seqlen, nheads_k, headdim)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
        causal: bool. Whether to apply causal attention mask.
        window_size: (left, right). Sliding window attention.
        alibi_slopes: (nheads,) or (batch_size, nheads)
        deterministic: bool. Whether to use deterministic implementation.
        return_attn_probs: bool. Whether to return attention probabilities.
    
    Returns:
        out: (batch_size, seqlen, nheads, headdim)
        softmax_lse: (batch_size, nheads, seqlen) if return_attn_probs=True
    """
    if not FLAGGEMS_AVAILABLE:
        raise RuntimeError("FlagGems is not available. Please install it first.")
    
    batch_size, seqlen_q, num_heads, head_dim = q.shape
    _, seqlen_k, num_heads_k, _ = k.shape
    
    # Convert window_size format
    window_left = window_size_left if window_size_left >= 0 else None
    window_right = window_size_right if window_size_right >= 0 else None
    
    # Call FlagGems implementation
    out, lse, _, _, p = flash_attention_forward(
        query=q,
        key=k,
        value=v,
        cumulative_sequence_length_q=None,
        cumulative_sequence_length_k=None,
        max_q=seqlen_q,
        max_k=seqlen_k,
        dropout_p=dropout_p,
        is_causal=causal,
        return_debug_mask=return_softmax,
        scale=softmax_scale,
        softcap=0.0,
        window_size_left=window_left,
        window_size_right=window_right,
        seqused_k=None,
        alibi_slopes=alibi_slopes,
        disable_splitkv=False,
    )
    
    if return_softmax:
        # Return (out, softmax_lse, S_dmask, rng_state)
        # Create dummy values for missing outputs
        rng_state = torch.empty(2, dtype=torch.int64, device=q.device)
        S_dmask = None  # Dummy for dropout mask
        return out, lse, S_dmask, rng_state
    else:
        # Return (out, softmax_lse, S_dmask, rng_state)
        rng_state = torch.empty(2, dtype=torch.int64, device=q.device)
        return out, lse, None, rng_state


def _flash_attn_varlen_forward(
    q, k, v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    """
    Adapter for flash_attn_varlen_func using FlagGems backend.
    
    Arguments:
        q: (total_q, nheads, headdim)
        k: (total_k, nheads_k, headdim)
        v: (total_k, nheads_k, headdim)
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32
        max_seqlen_q: int
        max_seqlen_k: int
        dropout_p: float
        softmax_scale: float
        causal: bool
        window_size: (left, right)
        alibi_slopes: (nheads,) or (batch_size, nheads)
        deterministic: bool
        return_attn_probs: bool
    
    Returns:
        out: (total_q, nheads, headdim)
        softmax_lse: (nheads, total_q) if return_attn_probs=True
    """
    if not FLAGGEMS_AVAILABLE:
        raise RuntimeError("FlagGems is not available. Please install it first.")
    
    # Call FlagGems varlen implementation
    result = flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        max_seqlen_q=max_seqlen_q,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=max_seqlen_k,
        cu_seqlens_k=cu_seqlens_k,
        seqused_k=None,
        q_v=None,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        softcap=0.0,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
        block_table=None,
        return_softmax_lse=return_attn_probs,
        out=None,
        fa_version=2,
    )
    
    if return_attn_probs:
        return result
    else:
        return result


def _flash_attn_qkvpacked_forward(
    qkv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    """
    Adapter for flash_attn_qkvpacked_func using FlagGems backend.
    
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, headdim)
        Other arguments same as _flash_attn_forward
    
    Returns:
        out: (batch_size, seqlen, nheads, headdim)
    """
    # Unpack qkv
    q, k, v = qkv.unbind(dim=2)
    
    # Call standard forward
    return _flash_attn_forward(
        q, k, v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
    )


def _flash_attn_kvpacked_forward(
    q, kv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    """
    Adapter for flash_attn_kvpacked_func using FlagGems backend.
    
    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        kv: (batch_size, seqlen, 2, nheads_k, headdim)
        Other arguments same as _flash_attn_forward
    
    Returns:
        out: (batch_size, seqlen, nheads, headdim)
    """
    # Unpack kv
    k, v = kv.unbind(dim=2)
    
    # Call standard forward
    return _flash_attn_forward(
        q, k, v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
    )


def _flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens=None,
    cache_batch_idx=None,
    cache_leftpad=None,
    block_table=None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    rotary_interleaved=True,
    alibi_slopes=None,
):
    """
    Adapter for flash_attn_with_kvcache using FlagGems backend.
    
    Note: This is a simplified implementation. Full KV cache support
    would require additional work in FlagGems.
    """
    raise NotImplementedError(
        "KV cache support is not yet implemented in FlagGems backend. "
        "This feature requires additional development."
    )


# Backward functions - not implemented yet
def _flash_attn_backward(*args, **kwargs):
    raise NotImplementedError(
        "Backward pass is not yet implemented in FlagGems backend. "
        "This feature requires additional development."
    )


def _flash_attn_varlen_backward(*args, **kwargs):
    raise NotImplementedError(
        "Backward pass is not yet implemented in FlagGems backend. "
        "This feature requires additional development."
    )