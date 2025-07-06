# FlashAttention with FlagGems Backend

This is a modified version of FlashAttention that uses FlagGems' Triton-based implementation instead of CUDA kernels. This allows FlashAttention to run on a wider variety of hardware platforms beyond just NVIDIA GPUs.

## Key Changes

1. **Removed CUDA Dependencies**: All CUDA C++ code has been replaced with FlagGems' Triton implementation
2. **Hardware Agnostic**: Can potentially run on any hardware supported by Triton
3. **Same API**: Maintains the same Python API as the original FlashAttention

## Installation

```bash
# Make sure FlagGems is installed
cd /path/to/FlagGems
pip install -e .

# Install flash-attention-plus
cd /path/to/flash-attention-plus
pip install -e .
```

## Usage

The API remains the same as the original FlashAttention:

```python
from flash_attn import flash_attn_func

# Basic usage
output = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)

# QKV packed format
from flash_attn import flash_attn_qkvpacked_func
output = flash_attn_qkvpacked_func(qkv, dropout_p=0.0, causal=True)
```

## Environment Variables

- `FLASH_ATTENTION_USE_FLAGGEMS`: Set to "TRUE" (default) to use FlagGems backend, "FALSE" to attempt using CUDA backend

## Testing

Run the test script to verify functionality:

```bash
python test_flash_attention_flaggems.py
```

## Limitations

1. **Backward Pass**: The backward pass is not yet implemented in the FlagGems backend
2. **KV Cache**: KV cache functionality is not yet supported
3. **Performance**: May not match the performance of the highly optimized CUDA kernels
4. **Some Features**: Advanced features like block-sparse attention are not yet implemented

## Original FlashAttention

This is based on the original FlashAttention by Tri Dao. See the original repository at:
https://github.com/Dao-AILab/flash-attention