# API Reference - FlashAttention-Plus

## Core Functions

### flash_attn_func

Main attention function compatible with the original FlashAttention API.

```python
flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor, 
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False
) -> torch.Tensor
```

**Parameters:**
- `q`: Query tensor of shape `(batch_size, seqlen, nheads, headdim)`
- `k`: Key tensor of shape `(batch_size, seqlen, nheads_k, headdim)`
- `v`: Value tensor of shape `(batch_size, seqlen, nheads_k, headdim)`
- `dropout_p`: Dropout probability (0.0 to 1.0)
- `softmax_scale`: Scaling factor for QK^T. Default: `1/sqrt(headdim)`
- `causal`: Whether to apply causal masking
- `window_size`: (left, right) for sliding window attention. Default: no windowing
- `softcap`: Softcapping value for attention scores
- `alibi_slopes`: ALiBi slopes for position bias
- `deterministic`: Whether to use deterministic algorithms
- `return_attn_probs`: Whether to return attention probabilities

**Returns:**
- Output tensor of shape `(batch_size, seqlen, nheads, headdim)`
- If `return_attn_probs=True`, also returns attention probabilities

### flash_attn_qkvpacked_func

Attention function for packed QKV format.

```python
flash_attn_qkvpacked_func(
    qkv: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False
) -> torch.Tensor
```

**Parameters:**
- `qkv`: Packed QKV tensor of shape `(batch_size, seqlen, 3, nheads, headdim)`
- Other parameters same as `flash_attn_func`

### flash_attn_kvpacked_func

Attention function with packed KV format.

```python
flash_attn_kvpacked_func(
    q: torch.Tensor,
    kv: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False
) -> torch.Tensor
```

**Parameters:**
- `q`: Query tensor of shape `(batch_size, seqlen, nheads, headdim)`
- `kv`: Packed KV tensor of shape `(batch_size, seqlen, 2, nheads_k, headdim)`
- Other parameters same as `flash_attn_func`

## Environment Variables

### FLASH_ATTENTION_USE_FLAGGEMS

Controls which backend to use for attention computation.

- `"TRUE"` (default): Use FlagGems/Triton backend
- `"FALSE"`: Attempt to use original CUDA backend

Example:
```python
import os
os.environ["FLASH_ATTENTION_USE_FLAGGEMS"] = "TRUE"
```

## Module Classes

### FlashAttention

PyTorch module wrapper for flash attention.

```python
class FlashAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        causal: bool = False,
        window_size: Tuple[int, int] = (-1, -1),
        alibi_slopes: Optional[torch.Tensor] = None
    )
```

**Methods:**
- `forward(q, k, v)`: Compute attention
- `reset_parameters()`: Reset module parameters

### FlashMHA

Multi-head attention module using FlashAttention.

```python
class FlashMHA(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        causal: bool = False,
        layer_idx: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    )
```

## Data Types and Constraints

### Supported Data Types
- `torch.float16` (FP16)
- `torch.bfloat16` (BF16)

### Tensor Requirements
- Must be contiguous in memory
- Must be on CUDA device
- Sequence length must be divisible by certain block sizes (typically 128)

### Shape Constraints
- `headdim` must be one of: 32, 40, 64, 80, 96, 128, 160, 192, 224, 256
- `nheads_k` must divide `nheads` evenly (for MQA/GQA)

## Error Handling

Common exceptions:
- `RuntimeError`: Invalid tensor shapes or unsupported configurations
- `ImportError`: FlagGems not properly installed
- `AssertionError`: Constraint violations

Example error handling:
```python
try:
    output = flash_attn_func(q, k, v, causal=True)
except RuntimeError as e:
    print(f"FlashAttention error: {e}")
    # Fallback to standard attention
    output = standard_attention(q, k, v)
```