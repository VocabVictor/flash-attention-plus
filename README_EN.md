# FlashAttention-Plus

**A hardware-agnostic implementation of FlashAttention using FlagGems/Triton backend**

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

[📖 中文文档](README.md) | [📖 Chinese Documentation](README_CN.md)

## Overview

FlashAttention-Plus is a drop-in replacement for the original [FlashAttention](https://github.com/Dao-AILab/flash-attention) that replaces NVIDIA CUDA kernels with [FlagGems](https://github.com/FlagOpen/FlagGems)' Triton implementation. This enables FlashAttention to run on a broader range of hardware while maintaining API compatibility.

**Key Features:**
- 🚀 **Hardware-agnostic**: Uses Triton instead of CUDA-specific code
- 🔄 **API Compatible**: Drop-in replacement for original FlashAttention
- ⚡ **High Performance**: Leverages optimized Triton kernels from FlagGems
- 🎯 **Easy Integration**: Minimal code changes required

## Installation

### Prerequisites

```bash
# PyTorch with CUDA support
pip install torch>=2.0.0

# Triton (required for FlagGems)
pip install triton>=3.0.0

# Other dependencies
pip install einops
```

### Install FlagGems

```bash
cd ~/.code/library/FlagGems  # or your preferred location
git clone https://github.com/FlagOpen/FlagGems.git
cd FlagGems
pip install -e .
```

### Install FlashAttention-Plus

```bash
git clone https://github.com/VocabVictor/flash-attention-plus.git
cd flash-attention-plus
pip install -e .
```

## Usage

### Quick Start

Current version uses FlagGems backend directly:

```python
import torch
from flash_attn import flash_attn_func

# Create tensors (must be fp16 or bf16)
q = torch.randn(2, 1024, 16, 64, device='cuda', dtype=torch.float16)
k = torch.randn(2, 1024, 16, 64, device='cuda', dtype=torch.float16)
v = torch.randn(2, 1024, 16, 64, device='cuda', dtype=torch.float16)

# Run flash attention
output = flash_attn_func(q, k, v, causal=True)
```

### Supported Functions

```python
# Standard attention
output = flash_attn_func(q, k, v, causal=True)

# QKV packed format
qkv = torch.randn(2, 1024, 3, 16, 64, device='cuda', dtype=torch.float16)
output = flash_attn_qkvpacked_func(qkv, causal=True)

# Variable length sequences
output = flash_attn_varlen_func(q_varlen, k_varlen, v_varlen, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
```

## Current Status

### ✅ Implemented Features
- **Forward Pass**: Fully implemented
  - Standard attention (flash_attn_func)
  - QKV packed format (flash_attn_qkvpacked_func)
  - KV packed format (flash_attn_kvpacked_func)
  - Variable-length sequences (flash_attn_varlen_func)
- **API Compatibility**: Full compatibility with original Flash Attention
- **Error Handling**: Proper error messages for unimplemented features

### ❌ Missing Features
- **Backward Pass**: Not implemented (inference only)
- **KV Cache**: Under development

## Performance

Performance varies by hardware and configuration. First run may be slower due to Triton kernel compilation, but subsequent runs use cached kernels.

## Documentation

For detailed documentation, please refer to:
- `README_CN.md` - Comprehensive Chinese documentation
- `IMPLEMENTATION_PLAN_CN.md` - Detailed implementation plan
- `TASK_CHECKLIST_CN.md` - Development task checklist
- `PROJECT_STATUS_CN.md` - Current project status

## License

This project maintains the same BSD 3-Clause License as the original FlashAttention. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Original FlashAttention by [Tri Dao](https://tridao.me/) and team
- [FlagGems](https://github.com/FlagOpen/FlagGems) team for the Triton kernels
- [OpenAI Triton](https://github.com/openai/triton) for the GPU programming language

## Roadmap

- [ ] Implement backward pass support
- [ ] Add KV cache functionality
- [ ] Performance optimizations
- [ ] Comprehensive benchmarks
- [ ] Support for more hardware backends

For detailed development plan, see `IMPLEMENTATION_PLAN_CN.md`.