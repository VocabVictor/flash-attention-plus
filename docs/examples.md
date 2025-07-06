# Code Examples - FlashAttention-Plus

## Basic Usage

### Simple Attention Computation

```python
import torch
import os

# Enable FlagGems backend
os.environ["FLASH_ATTENTION_USE_FLAGGEMS"] = "TRUE"

from flash_attn import flash_attn_func

# Create sample tensors
batch_size, seq_len, n_heads, head_dim = 2, 1024, 16, 64
device = torch.device('cuda')
dtype = torch.float16

q = torch.randn(batch_size, seq_len, n_heads, head_dim, device=device, dtype=dtype)
k = torch.randn(batch_size, seq_len, n_heads, head_dim, device=device, dtype=dtype)
v = torch.randn(batch_size, seq_len, n_heads, head_dim, device=device, dtype=dtype)

# Compute attention
output = flash_attn_func(q, k, v, causal=True)
print(f"Output shape: {output.shape}")
```

### Using Packed Formats

```python
# QKV packed format
qkv = torch.randn(batch_size, seq_len, 3, n_heads, head_dim, device=device, dtype=dtype)
from flash_attn import flash_attn_qkvpacked_func
output = flash_attn_qkvpacked_func(qkv, causal=True)

# KV packed format (useful for cross-attention)
q = torch.randn(batch_size, seq_len, n_heads, head_dim, device=device, dtype=dtype)
kv = torch.randn(batch_size, seq_len, 2, n_heads, head_dim, device=device, dtype=dtype)
from flash_attn import flash_attn_kvpacked_func
output = flash_attn_kvpacked_func(q, kv)
```

## Integration with Transformers

### Custom Attention Layer

```python
import torch.nn as nn
from flash_attn.modules.mha import FlashMHA

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.1):
        super().__init__()
        self.attn = FlashMHA(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=dropout,
            causal=True
        )
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Pre-norm architecture
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

# Usage
model = TransformerBlock(dim=768, n_heads=12).cuda()
x = torch.randn(2, 512, 768, device='cuda', dtype=torch.float16)
output = model(x)
```

### Multi-Query Attention (MQA)

```python
# MQA: multiple query heads share the same key/value heads
batch_size, seq_len = 2, 1024
n_heads_q, n_heads_kv = 32, 8  # 32 query heads, 8 key/value heads
head_dim = 128

q = torch.randn(batch_size, seq_len, n_heads_q, head_dim, device=device, dtype=dtype)
k = torch.randn(batch_size, seq_len, n_heads_kv, head_dim, device=device, dtype=dtype)
v = torch.randn(batch_size, seq_len, n_heads_kv, head_dim, device=device, dtype=dtype)

# FlashAttention automatically handles MQA
output = flash_attn_func(q, k, v, causal=True)
print(f"MQA output shape: {output.shape}")  # (2, 1024, 32, 128)
```

## Advanced Features

### Sliding Window Attention

```python
# Local attention with window size
window_size = (256, 0)  # Look back 256 tokens, no look-ahead
output = flash_attn_func(
    q, k, v,
    causal=True,
    window_size=window_size
)
```

### Custom Softmax Scaling

```python
# Override default scaling factor (1/sqrt(head_dim))
custom_scale = 1.0 / (head_dim ** 0.5) * 1.5
output = flash_attn_func(
    q, k, v,
    softmax_scale=custom_scale,
    causal=True
)
```

### Dropout During Training

```python
# Apply dropout to attention weights
model.train()
output = flash_attn_func(
    q, k, v,
    dropout_p=0.1,  # 10% dropout
    causal=True
)

# Disable dropout for inference
model.eval()
with torch.no_grad():
    output = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
```

## Performance Benchmarking

```python
import time
import torch
from flash_attn import flash_attn_func

def benchmark_attention(seq_lengths, n_heads=16, head_dim=64, num_iters=100):
    device = torch.device('cuda')
    dtype = torch.float16
    
    for seq_len in seq_lengths:
        q = torch.randn(1, seq_len, n_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(1, seq_len, n_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(1, seq_len, n_heads, head_dim, device=device, dtype=dtype)
        
        # Warmup
        for _ in range(10):
            _ = flash_attn_func(q, k, v, causal=True)
        
        torch.cuda.synchronize()
        start = time.time()
        
        for _ in range(num_iters):
            _ = flash_attn_func(q, k, v, causal=True)
        
        torch.cuda.synchronize()
        end = time.time()
        
        avg_time = (end - start) / num_iters * 1000  # ms
        print(f"Seq length {seq_len}: {avg_time:.2f} ms")

# Run benchmark
benchmark_attention([512, 1024, 2048, 4096, 8192])
```

## Migration from Standard Attention

### Before (Standard PyTorch)

```python
def standard_attention(q, k, v, mask=None, dropout_p=0.0):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    attn_weights = torch.softmax(scores, dim=-1)
    if dropout_p > 0:
        attn_weights = torch.dropout(attn_weights, dropout_p, train=True)
    
    output = torch.matmul(attn_weights, v)
    return output
```

### After (FlashAttention-Plus)

```python
os.environ["FLASH_ATTENTION_USE_FLAGGEMS"] = "TRUE"
from flash_attn import flash_attn_func

def flash_attention(q, k, v, is_causal=True, dropout_p=0.0):
    # Note: q, k, v should be (batch, seq_len, n_heads, head_dim)
    # If your tensors are (batch, n_heads, seq_len, head_dim), transpose:
    # q = q.transpose(1, 2)
    # k = k.transpose(1, 2)
    # v = v.transpose(1, 2)
    
    output = flash_attn_func(q, k, v, causal=is_causal, dropout_p=dropout_p)
    return output
```

## Error Handling Example

```python
def safe_flash_attention(q, k, v, **kwargs):
    """Wrapper with fallback to standard attention"""
    try:
        # Try FlashAttention first
        return flash_attn_func(q, k, v, **kwargs)
    except Exception as e:
        print(f"FlashAttention failed: {e}, falling back to standard attention")
        
        # Fallback implementation
        batch, seq_len, n_heads, head_dim = q.shape
        q = q.transpose(1, 2)  # (batch, n_heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        
        if kwargs.get('causal', False):
            mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1)
            scores = scores.masked_fill(mask.bool(), float('-inf'))
        
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        return output.transpose(1, 2)  # Back to (batch, seq_len, n_heads, head_dim)
```