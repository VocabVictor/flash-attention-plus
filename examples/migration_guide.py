#!/usr/bin/env python3
"""
Migration guide from original Flash-Attention to Flash-Attention-Plus
"""

import os
import torch

print("=== Flash-Attention to Flash-Attention-Plus Migration Guide ===\n")

print("1. Environment Setup:")
print("   Original Flash-Attention requires NVIDIA CUDA compilation.")
print("   Flash-Attention-Plus uses FlagGems Triton backend - no compilation needed!")
print()

print("2. Installation:")
print("   # Original")
print("   pip install flash-attn --no-build-isolation")
print()
print("   # Flash-Attention-Plus")
print("   pip install triton>=3.0")
print("   pip install -e /path/to/FlagGems")
print("   pip install -e /path/to/flash-attention-plus")
print()

print("3. Code Changes:")
print("   Minimal! Just set an environment variable:")
print()
print("   import os")
print('   os.environ["FLASH_ATTENTION_USE_FLAGGEMS"] = "TRUE"')
print()
print("   # Then use flash_attn as usual")
print("   from flash_attn import flash_attn_func")
print()

print("4. Data Type Requirements:")
print("   FlagGems requires fp16 or bf16 tensors:")
print("   tensor = tensor.half()  # Convert to fp16")
print("   # or")
print("   tensor = tensor.to(torch.bfloat16)  # Convert to bf16")
print()

print("5. Feature Support:")
print("   ✅ Supported:")
print("      - Basic flash attention")
print("      - QKV packed format")
print("      - Causal masking")
print("      - Custom scaling")
print("      - Sliding window attention")
print()
print("   ⚠️  Limited Support:")
print("      - Dropout (interface exists but may not be fully implemented)")
print("      - ALiBi slopes")
print()
print("   ❌ Not Yet Supported:")
print("      - KV cache")
print("      - Backward pass (gradients)")
print("      - Variable length sequences")
print()

print("6. Performance Considerations:")
print("   - First run may be slower due to Triton compilation")
print("   - Subsequent runs will use cached kernels")
print("   - Performance may vary compared to CUDA implementation")
print()

# Example code comparison
print("7. Example Code Comparison:")
print()
print("Original Flash-Attention:")
print("-" * 40)
original_code = '''
import torch
from flash_attn import flash_attn_func

# Create tensors
q = torch.randn(2, 1024, 16, 64, device='cuda', dtype=torch.float16)
k = torch.randn(2, 1024, 16, 64, device='cuda', dtype=torch.float16)
v = torch.randn(2, 1024, 16, 64, device='cuda', dtype=torch.float16)

# Run flash attention
output = flash_attn_func(q, k, v, causal=True)
'''
print(original_code)

print("\nFlash-Attention-Plus:")
print("-" * 40)
plus_code = '''
import os
import torch

# Enable FlagGems backend
os.environ["FLASH_ATTENTION_USE_FLAGGEMS"] = "TRUE"

from flash_attn import flash_attn_func

# Create tensors (same as before!)
q = torch.randn(2, 1024, 16, 64, device='cuda', dtype=torch.float16)
k = torch.randn(2, 1024, 16, 64, device='cuda', dtype=torch.float16)
v = torch.randn(2, 1024, 16, 64, device='cuda', dtype=torch.float16)

# Run flash attention (same API!)
output = flash_attn_func(q, k, v, causal=True)
'''
print(plus_code)

print("\n8. Troubleshooting:")
print("   - If you see 'FlagGems not available', check FlagGems installation")
print("   - For 'fp16/bf16 required' errors, convert your tensors")
print("   - If performance is poor, ensure Triton cache is enabled")
print()

print("9. Switching Back:")
print("   To use original CUDA backend, simply set:")
print('   os.environ["FLASH_ATTENTION_USE_FLAGGEMS"] = "FALSE"')
print()

print("For more examples, see the examples/ directory!")
print("Happy coding with hardware-agnostic Flash Attention! 🚀")