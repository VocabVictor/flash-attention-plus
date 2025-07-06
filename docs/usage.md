# Usage Guide

This guide covers how to use FlashAttention-Plus in your projects.

## Basic Usage

### Enabling FlagGems Backend

FlashAttention-Plus uses an environment variable to switch between backends:

```python
import os

# Enable FlagGems backend (default)
os.environ["FLASH_ATTENTION_USE_FLAGGEMS"] = "TRUE"

# Or disable to use original CUDA backend (if available)
os.environ["FLASH_ATTENTION_USE_FLAGGEMS"] = "FALSE"
```

### Simple Example

```python
import torch
from flash_attn import flash_attn_func

# Create input tensors
batch_size = 2
seq_length = 1024
num_heads = 16
head_dim = 64

# Note: Inputs must be fp16 or bf16
q = torch.randn(batch_size, seq_length, num_heads, head_dim, 
                device='cuda', dtype=torch.float16)
k = torch.randn(batch_size, seq_length, num_heads, head_dim, 
                device='cuda', dtype=torch.float16)
v = torch.randn(batch_size, seq_length, num_heads, head_dim, 
                device='cuda', dtype=torch.float16)

# Apply flash attention
output = flash_attn_func(q, k, v, causal=True)
```

## Advanced Usage

### With Dropout

```python
# Apply attention with dropout
output = flash_attn_func(q, k, v, dropout_p=0.1, causal=True)
```

!!! warning "Dropout Support"
    Dropout interface is available but may not be fully functional in the current version.

### Custom Softmax Scale

```python
# Custom scaling factor
scale = 1.0 / math.sqrt(head_dim)
output = flash_attn_func(q, k, v, softmax_scale=scale, causal=True)
```

### Non-Causal Attention

```python
# For bidirectional attention (e.g., BERT)
output = flash_attn_func(q, k, v, causal=False)
```

## Input Requirements

### Data Types

FlashAttention-Plus requires inputs to be in half-precision format:

- `torch.float16` (fp16)
- `torch.bfloat16` (bf16)

```python
# Convert to fp16 if needed
q = q.to(torch.float16)
k = k.to(torch.float16)
v = v.to(torch.float16)
```

### Tensor Shape

Input tensors should have the shape: `[batch_size, seq_length, num_heads, head_dim]`

### Device

All tensors must be on CUDA device:

```python
q = q.to('cuda')
k = k.to('cuda')
v = v.to('cuda')
```

## Integration with Transformers

### Custom Attention Module

```python
import torch.nn as nn
from flash_attn import flash_attn_func

class FlashSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
    def forward(self, x, causal=False):
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        
        # Apply flash attention
        output = flash_attn_func(q, k, v, dropout_p=self.dropout, causal=causal)
        
        # Reshape and project output
        output = output.reshape(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(output)
        
        return output
```

### Using with Existing Models

```python
# Replace standard attention with flash attention
model = YourTransformerModel()

# Enable FlagGems backend
os.environ["FLASH_ATTENTION_USE_FLAGGEMS"] = "TRUE"

# The model will now use FlashAttention-Plus
output = model(input_ids)
```

## Performance Tips

1. **First Run**: The first run may be slower due to Triton kernel compilation. Subsequent runs will use cached kernels.

2. **Batch Processing**: Process multiple sequences together for better GPU utilization:
   ```python
   # Good: Batch multiple sequences
   batch_output = flash_attn_func(batch_q, batch_k, batch_v)
   
   # Less efficient: Process one at a time
   for i in range(batch_size):
       output_i = flash_attn_func(q[i:i+1], k[i:i+1], v[i:i+1])
   ```

3. **Memory Efficiency**: FlashAttention is designed to be memory-efficient. You can process longer sequences than standard attention.

## Debugging

Enable debug mode to get more information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# This will show detailed information about the attention computation
output = flash_attn_func(q, k, v, causal=True)
```

## Next Steps

- See [Examples](examples.md) for complete working examples
- Check the [API Reference](api.md) for detailed parameter documentation
- Read the [Technical Details](technical.md) to understand how it works