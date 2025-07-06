# Migration Guide: Flash-Attention → Flash-Attention-Plus

## Overview

Flash-Attention-Plus is a drop-in replacement for the original Flash-Attention that uses FlagGems' Triton implementation instead of NVIDIA CUDA. This guide will help you migrate your existing code.

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install torch>=2.0
pip install triton>=3.0
pip install einops

# Install FlagGems
cd ~/.code/library/FlagGems
pip install -e .

# Install Flash-Attention-Plus
cd ~/.code/library/flash-attention-plus
pip install -e .
```

### 2. Code Changes

The migration requires minimal code changes:

```python
# Add this before importing flash_attn
import os
os.environ["FLASH_ATTENTION_USE_FLAGGEMS"] = "TRUE"

# Then use flash_attn as usual
from flash_attn import flash_attn_func
```

That's it! Your existing code should work with the FlagGems backend.

## Detailed Comparison

### Original Flash-Attention
```python
import torch
from flash_attn import flash_attn_func

q = torch.randn(2, 1024, 16, 64, device='cuda', dtype=torch.float16)
k = torch.randn(2, 1024, 16, 64, device='cuda', dtype=torch.float16)
v = torch.randn(2, 1024, 16, 64, device='cuda', dtype=torch.float16)

output = flash_attn_func(q, k, v, causal=True)
```

### Flash-Attention-Plus
```python
import os
os.environ["FLASH_ATTENTION_USE_FLAGGEMS"] = "TRUE"

import torch
from flash_attn import flash_attn_func

q = torch.randn(2, 1024, 16, 64, device='cuda', dtype=torch.float16)
k = torch.randn(2, 1024, 16, 64, device='cuda', dtype=torch.float16)
v = torch.randn(2, 1024, 16, 64, device='cuda', dtype=torch.float16)

output = flash_attn_func(q, k, v, causal=True)
```

## Important Differences

### 1. Data Type Requirements

FlagGems requires **fp16** or **bf16** tensors:

```python
# Convert existing tensors
q = q.half()  # to fp16
# or
q = q.to(torch.bfloat16)  # to bf16
```

### 2. Feature Support

| Feature | Original | Flash-Attention-Plus |
|---------|----------|---------------------|
| Basic attention | ✅ | ✅ |
| QKV packed | ✅ | ✅ |
| Causal mask | ✅ | ✅ |
| Custom scaling | ✅ | ✅ |
| Sliding window | ✅ | ✅ |
| Dropout | ✅ | ⚠️ Interface only |
| ALiBi slopes | ✅ | ⚠️ Limited |
| KV cache | ✅ | ❌ Not implemented |
| Backward pass | ✅ | ❌ Not implemented |
| Variable length | ✅ | ❌ Not implemented |

### 3. Performance Considerations

- **First run**: May be slower due to Triton kernel compilation
- **Subsequent runs**: Use cached compiled kernels
- **Memory usage**: Similar to original implementation
- **Accuracy**: Slight numerical differences (< 0.002) due to different implementations

## Common Issues and Solutions

### Issue 1: "FlagGems not available"
**Solution**: Ensure FlagGems is properly installed:
```bash
cd ~/.code/library/FlagGems
pip install -e .
```

### Issue 2: "FlashAttention only support fp16 and bf16"
**Solution**: Convert your tensors:
```python
q = q.half()  # or q.to(torch.bfloat16)
k = k.half()
v = v.half()
```

### Issue 3: ImportError with Triton
**Solution**: Upgrade Triton:
```bash
pip install --upgrade triton>=3.0
```

## Switching Between Backends

You can easily switch between backends:

```python
# Use FlagGems backend
os.environ["FLASH_ATTENTION_USE_FLAGGEMS"] = "TRUE"

# Use original CUDA backend (if available)
os.environ["FLASH_ATTENTION_USE_FLAGGEMS"] = "FALSE"
```

## Example: Transformer Model Migration

```python
import os
os.environ["FLASH_ATTENTION_USE_FLAGGEMS"] = "TRUE"

import torch
import torch.nn as nn
from flash_attn import flash_attn_func

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        B, N, C = x.shape
        
        # Generate QKV
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        
        # Ensure fp16 for FlagGems
        q, k, v = q.half(), k.half(), v.half()
        
        # Flash Attention
        out = flash_attn_func(q, k, v, causal=True)
        
        # Project back
        out = out.reshape(B, N, C)
        return self.proj(out.float())
```

## Performance Tips

1. **Use fp16 instead of bf16** when possible - it's generally faster
2. **Batch operations** to amortize kernel launch overhead
3. **Keep sequence lengths** as multiples of 128 for better performance
4. **Warm up** with a few iterations to compile Triton kernels

## Future Roadmap

- [ ] Backward pass implementation
- [ ] KV cache support
- [ ] Variable length sequence support
- [ ] Further performance optimizations

## Getting Help

- Check the `examples/` directory for more usage examples
- File issues at the GitHub repository
- Refer to FlagGems documentation for backend-specific details

## Conclusion

Flash-Attention-Plus provides a hardware-agnostic alternative to the original Flash-Attention with minimal code changes required. While some advanced features are not yet implemented, it covers the most common use cases and provides good performance with the benefit of not requiring NVIDIA-specific toolchains.