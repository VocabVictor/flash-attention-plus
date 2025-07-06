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

Simply set an environment variable to use the FlagGems backend:

```python
import os
os.environ["FLASH_ATTENTION_USE_FLAGGEMS"] = "TRUE"

# Then use flash_attn as usual
from flash_attn import flash_attn_func

# Create tensors (must be fp16 or bf16)
q = torch.randn(2, 1024, 16, 64, device='cuda', dtype=torch.float16)
k = torch.randn(2, 1024, 16, 64, device='cuda', dtype=torch.float16)
v = torch.randn(2, 1024, 16, 64, device='cuda', dtype=torch.float16)

# Run flash attention
output = flash_attn_func(q, k, v, causal=True)
```

### Switching Between Backends

```python
# Use FlagGems backend (hardware-agnostic)
os.environ["FLASH_ATTENTION_USE_FLAGGEMS"] = "TRUE"

# Use original CUDA backend (if available)
os.environ["FLASH_ATTENTION_USE_FLAGGEMS"] = "FALSE"
```

## Key Differences from Original FlashAttention

### What's Changed
- **Backend**: CUDA kernels → FlagGems/Triton kernels
- **Hardware Support**: NVIDIA-only → Hardware-agnostic
- **Installation**: No CUDA compilation required

### What's Preserved
- ✅ API compatibility
- ✅ Core functionality
- ✅ Model support (BERT, GPT, LLaMA, etc.)
- ✅ Performance characteristics

### Current Limitations
- ❌ Backward pass not yet implemented
- ❌ KV cache support pending
- ❌ Variable length sequences not supported
- ⚠️ Dropout interface exists but may not be fully functional

## Examples

See the [examples](examples/) directory for more usage examples:
- [basic_usage.py](examples/basic_usage.py) - Basic FlashAttention usage
- [migration_guide.py](examples/migration_guide.py) - Migration from original FlashAttention

## Performance

Performance varies by hardware and configuration. First run may be slower due to Triton kernel compilation, but subsequent runs use cached kernels.

## Migration Guide

For detailed migration instructions from the original FlashAttention, see [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md).

## Technical Details

For more information about the FlagGems integration, see [README_FLAGGEMS.md](README_FLAGGEMS.md).

## Citation

If you use FlashAttention-Plus in your research, please cite both this project and the original FlashAttention:

```bibtex
@misc{flashattentionplus2024,
  author = {Wu, Zhongheng},
  title = {FlashAttention-Plus: Hardware-agnostic FlashAttention using FlagGems/Triton},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/VocabVictor/flash-attention-plus}
}
```

Original FlashAttention papers:
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://tridao.me/publications/flash2/flash2.pdf)

## License

This project maintains the same BSD 3-Clause License as the original FlashAttention. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Original FlashAttention by [Tri Dao](https://tridao.me/) and team
- [FlagGems](https://github.com/FlagOpen/FlagGems) team for the Triton kernels
- [OpenAI Triton](https://github.com/openai/triton) for the GPU programming language

## Author

**Zhongheng Wu**  
Nanjing University

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Roadmap

- [ ] Implement backward pass support
- [ ] Add KV cache functionality
- [ ] Support variable length sequences
- [ ] Performance optimizations
- [ ] Comprehensive benchmarks
- [ ] Support for more hardware backends