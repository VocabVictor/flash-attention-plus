# FlashAttention-Plus

**A hardware-agnostic implementation of FlashAttention using FlagGems/Triton backend**

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Overview

FlashAttention-Plus is a drop-in replacement for the original [FlashAttention](https://github.com/Dao-AILab/flash-attention) that replaces NVIDIA CUDA kernels with [FlagGems](https://github.com/FlagOpen/FlagGems)' Triton implementation. This enables FlashAttention to run on a broader range of hardware while maintaining API compatibility.

**Key Features:**

- 🚀 **Hardware-agnostic**: Uses Triton instead of CUDA-specific code
- 🔄 **API Compatible**: Drop-in replacement for original FlashAttention
- ⚡ **High Performance**: Leverages optimized Triton kernels from FlagGems
- 🎯 **Easy Integration**: Minimal code changes required

## Why FlashAttention-Plus?

The original FlashAttention implementation provides excellent performance but is limited to NVIDIA GPUs due to its CUDA-specific kernels. FlashAttention-Plus addresses this limitation by using FlagGems' Triton-based implementation, which can potentially run on various hardware accelerators while maintaining the same API.

## Quick Example

```python
import os
import torch

# Enable FlagGems backend
os.environ["FLASH_ATTENTION_USE_FLAGGEMS"] = "TRUE"

from flash_attn import flash_attn_func

# Create tensors (must be fp16 or bf16)
batch_size, seq_len, num_heads, head_dim = 2, 1024, 16, 64
q = torch.randn(batch_size, seq_len, num_heads, head_dim, 
                device='cuda', dtype=torch.float16)
k = torch.randn(batch_size, seq_len, num_heads, head_dim, 
                device='cuda', dtype=torch.float16)
v = torch.randn(batch_size, seq_len, num_heads, head_dim, 
                device='cuda', dtype=torch.float16)

# Run flash attention
output = flash_attn_func(q, k, v, causal=True)
print(f"Output shape: {output.shape}")
```

## Getting Started

- [Installation Guide](installation.md) - Set up FlashAttention-Plus
- [Usage Guide](usage.md) - Learn how to use FlashAttention-Plus
- [Migration Guide](migration.md) - Migrate from original FlashAttention
- [API Reference](api.md) - Detailed API documentation

## Project Status

This project is in active development. Current limitations include:

- ❌ Backward pass not yet implemented
- ❌ KV cache support pending
- ❌ Variable length sequences not supported
- ⚠️ Dropout interface exists but may not be fully functional

See our [Roadmap](#roadmap) for upcoming features.

## Roadmap

- [ ] Implement backward pass support
- [ ] Add KV cache functionality
- [ ] Support variable length sequences
- [ ] Performance optimizations
- [ ] Comprehensive benchmarks
- [ ] Support for more hardware backends

## License

This project maintains the same BSD 3-Clause License as the original FlashAttention. See [LICENSE](https://github.com/VocabVictor/flash-attention-plus/blob/main/LICENSE) for details.

## Acknowledgments

- Original FlashAttention by [Tri Dao](https://tridao.me/) and team
- [FlagGems](https://github.com/FlagOpen/FlagGems) team for the Triton kernels
- [OpenAI Triton](https://github.com/openai/triton) for the GPU programming language