# Migration Guide

This guide helps you migrate from the original FlashAttention to FlashAttention-Plus.

## Quick Migration

The migration process is straightforward since FlashAttention-Plus maintains API compatibility:

### Step 1: Install FlashAttention-Plus

Follow the [Installation Guide](installation.md) to install FlashAttention-Plus and its dependencies.

### Step 2: Enable FlagGems Backend

Add this line before importing flash_attn:

```python
import os
os.environ["FLASH_ATTENTION_USE_FLAGGEMS"] = "TRUE"
```

### Step 3: No Code Changes Required

Your existing code should work without modifications:

```python
# Original code - no changes needed!
from flash_attn import flash_attn_func

output = flash_attn_func(q, k, v, causal=True)
```

## Detailed Migration

### For Training Scripts

```python
# Before: Original FlashAttention
import torch
from flash_attn import flash_attn_func

# After: FlashAttention-Plus
import os
os.environ["FLASH_ATTENTION_USE_FLAGGEMS"] = "TRUE"  # Add this line

import torch
from flash_attn import flash_attn_func  # Same import
```

### For Model Definitions

If you have custom attention modules:

```python
# The module definition remains the same
class MyFlashAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        # No changes needed
        
    def forward(self, x):
        # flash_attn_func works the same way
        return flash_attn_func(q, k, v, causal=self.causal)
```

## Feature Compatibility

| Feature | Original FlashAttention | FlashAttention-Plus |
|---------|------------------------|--------------------|
| Forward Pass | ✅ | ✅ |
| Backward Pass | ✅ | ❌ (Coming soon) |
| Causal Masking | ✅ | ✅ |
| Dropout | ✅ | ⚠️ (Limited) |
| Custom Softmax Scale | ✅ | ✅ |
| FP16/BF16 Support | ✅ | ✅ |
| Variable Length | ✅ | ❌ (Coming soon) |
| KV Cache | ✅ | ❌ (Coming soon) |

## Common Migration Scenarios

### Scenario 1: Research Projects

For research projects using FlashAttention:

1. Install FlashAttention-Plus alongside your existing setup
2. Set the environment variable at the beginning of your script
3. Run your experiments as usual

### Scenario 2: Production Systems

For production deployments:

1. Test thoroughly with your specific workloads
2. Monitor performance metrics
3. Keep the ability to switch back:

```python
USE_FLAGGEMS = os.getenv("USE_FLAGGEMS", "true").lower() == "true"

if USE_FLAGGEMS:
    os.environ["FLASH_ATTENTION_USE_FLAGGEMS"] = "TRUE"
else:
    os.environ["FLASH_ATTENTION_USE_FLAGGEMS"] = "FALSE"
```

### Scenario 3: Multi-GPU Training

FlashAttention-Plus works with distributed training:

```python
# Works with DDP, FSDP, etc.
model = YourModel()
model = DDP(model)

# FlashAttention-Plus will work across all GPUs
```

## Troubleshooting Migration Issues

### Issue: Performance Regression

**Solution**: First run may be slower due to Triton compilation. Run a warmup:

```python
# Warmup run
for _ in range(3):
    _ = flash_attn_func(q[:1], k[:1], v[:1])

# Actual computation
output = flash_attn_func(q, k, v)
```

### Issue: Backward Pass Not Working

**Current Limitation**: Backward pass is not yet implemented. For training, you may need to keep using the original FlashAttention for now.

### Issue: Import Errors

**Solution**: Ensure all dependencies are properly installed:

```bash
pip install --upgrade triton
pip install -e /path/to/FlagGems
pip install -e /path/to/flash-attention-plus
```

## Rollback Plan

If you need to rollback to the original FlashAttention:

```python
# Simply set the environment variable to FALSE
os.environ["FLASH_ATTENTION_USE_FLAGGEMS"] = "FALSE"
```

Or uninstall FlashAttention-Plus:

```bash
pip uninstall flash-attn-plus
```

## Next Steps

- Check [Examples](examples.md) for working code samples
- Read [Technical Details](technical.md) for implementation details
- Report issues on [GitHub](https://github.com/VocabVictor/flash-attention-plus/issues)