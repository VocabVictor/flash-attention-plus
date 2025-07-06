# Installation Guide

This guide will help you install FlashAttention-Plus and its dependencies.

## Prerequisites

Before installing FlashAttention-Plus, ensure you have:

- Python 3.8 or higher
- PyTorch 2.0 or higher with CUDA support
- CUDA 11.6 or higher (for GPU support)

## Step 1: Install Core Dependencies

```bash
# PyTorch with CUDA support
pip install torch>=2.0.0

# Triton (required for FlagGems)
pip install triton>=3.0.0

# Other dependencies
pip install einops
```

!!! note "Triton Version"
    FlagGems requires Triton 3.0 or higher. If you have PyTorch pre-installed with an older Triton version, you may need to upgrade:
    ```bash
    pip install --upgrade triton
    ```

## Step 2: Install FlagGems

FlagGems provides the Triton-based Flash Attention implementation:

```bash
# Clone FlagGems repository
git clone https://github.com/FlagOpen/FlagGems.git
cd FlagGems

# Install in development mode
pip install -e .
```

## Step 3: Install FlashAttention-Plus

```bash
# Clone the repository
git clone https://github.com/VocabVictor/flash-attention-plus.git
cd flash-attention-plus

# Install in development mode
pip install -e .
```

## Verify Installation

To verify that FlashAttention-Plus is installed correctly:

```python
import torch
from flash_attn import flash_attn_func

# Check if import works
print("FlashAttention-Plus imported successfully!")

# Test with a simple example
q = torch.randn(1, 64, 8, 32, device='cuda', dtype=torch.float16)
k = torch.randn(1, 64, 8, 32, device='cuda', dtype=torch.float16)
v = torch.randn(1, 64, 8, 32, device='cuda', dtype=torch.float16)

output = flash_attn_func(q, k, v)
print(f"Test passed! Output shape: {output.shape}")
```

## Troubleshooting

### Common Issues

#### 1. NumPy Version Conflict

If you encounter NumPy version issues:
```bash
pip install numpy==1.26.4
```

#### 2. Triton Import Errors

If you see errors related to Triton imports:
```bash
# Upgrade Triton to the latest version
pip install --upgrade triton
```

#### 3. CUDA Not Available

Ensure PyTorch is installed with CUDA support:
```python
import torch
print(torch.cuda.is_available())  # Should return True
```

### Environment Setup

For a clean installation, we recommend using a conda environment:

```bash
# Create a new environment
conda create -n flash-attn-plus python=3.10
conda activate flash-attn-plus

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Continue with the installation steps above
```

## Next Steps

- Check out the [Usage Guide](usage.md) to learn how to use FlashAttention-Plus
- See [Examples](examples.md) for practical code samples
- Read the [Migration Guide](migration.md) if you're coming from the original FlashAttention