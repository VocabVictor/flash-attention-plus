# 代码示例 - FlashAttention-Plus

## 基础用法

### 简单的注意力计算

```python
import torch
import os

# 启用 FlagGems 后端
os.environ["FLASH_ATTENTION_USE_FLAGGEMS"] = "TRUE"

from flash_attn import flash_attn_func

# 创建示例张量
batch_size, seq_len, n_heads, head_dim = 2, 1024, 16, 64
device = torch.device('cuda')
dtype = torch.float16

q = torch.randn(batch_size, seq_len, n_heads, head_dim, device=device, dtype=dtype)
k = torch.randn(batch_size, seq_len, n_heads, head_dim, device=device, dtype=dtype)
v = torch.randn(batch_size, seq_len, n_heads, head_dim, device=device, dtype=dtype)

# 计算注意力
output = flash_attn_func(q, k, v, causal=True)
print(f"输出形状：{output.shape}")
```

### 使用打包格式

```python
# QKV 打包格式
qkv = torch.randn(batch_size, seq_len, 3, n_heads, head_dim, device=device, dtype=dtype)
from flash_attn import flash_attn_qkvpacked_func
output = flash_attn_qkvpacked_func(qkv, causal=True)

# KV 打包格式（用于交叉注意力）
q = torch.randn(batch_size, seq_len, n_heads, head_dim, device=device, dtype=dtype)
kv = torch.randn(batch_size, seq_len, 2, n_heads, head_dim, device=device, dtype=dtype)
from flash_attn import flash_attn_kvpacked_func
output = flash_attn_kvpacked_func(q, kv)
```

## 与 Transformers 集成

### 自定义注意力层

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
        # Pre-norm 架构
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

# 使用
model = TransformerBlock(dim=768, n_heads=12).cuda()
x = torch.randn(2, 512, 768, device='cuda', dtype=torch.float16)
output = model(x)
```

### 多查询注意力（MQA）

```python
# MQA：多个查询头共享相同的键/值头
batch_size, seq_len = 2, 1024
n_heads_q, n_heads_kv = 32, 8  # 32 个查询头，8 个键/值头
head_dim = 128

q = torch.randn(batch_size, seq_len, n_heads_q, head_dim, device=device, dtype=dtype)
k = torch.randn(batch_size, seq_len, n_heads_kv, head_dim, device=device, dtype=dtype)
v = torch.randn(batch_size, seq_len, n_heads_kv, head_dim, device=device, dtype=dtype)

# FlashAttention 自动处理 MQA
output = flash_attn_func(q, k, v, causal=True)
print(f"MQA 输出形状：{output.shape}")  # (2, 1024, 32, 128)
```

## 高级功能

### 滑动窗口注意力

```python
# 具有窗口大小的局部注意力
window_size = (256, 0)  # 回看 256 个标记，不向前看
output = flash_attn_func(
    q, k, v,
    causal=True,
    window_size=window_size
)
```

### 自定义 Softmax 缩放

```python
# 覆盖默认缩放因子 (1/sqrt(head_dim))
custom_scale = 1.0 / (head_dim ** 0.5) * 1.5
output = flash_attn_func(
    q, k, v,
    softmax_scale=custom_scale,
    causal=True
)
```

### 训练期间的 Dropout

```python
# 对注意力权重应用 dropout
model.train()
output = flash_attn_func(
    q, k, v,
    dropout_p=0.1,  # 10% dropout
    causal=True
)

# 推理时禁用 dropout
model.eval()
with torch.no_grad():
    output = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
```

## 性能基准测试

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
        
        # 预热
        for _ in range(10):
            _ = flash_attn_func(q, k, v, causal=True)
        
        torch.cuda.synchronize()
        start = time.time()
        
        for _ in range(num_iters):
            _ = flash_attn_func(q, k, v, causal=True)
        
        torch.cuda.synchronize()
        end = time.time()
        
        avg_time = (end - start) / num_iters * 1000  # 毫秒
        print(f"序列长度 {seq_len}：{avg_time:.2f} 毫秒")

# 运行基准测试
benchmark_attention([512, 1024, 2048, 4096, 8192])
```

## 从标准注意力迁移

### 之前（标准 PyTorch）

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

### 之后（FlashAttention-Plus）

```python
os.environ["FLASH_ATTENTION_USE_FLAGGEMS"] = "TRUE"
from flash_attn import flash_attn_func

def flash_attention(q, k, v, is_causal=True, dropout_p=0.0):
    # 注意：q, k, v 应该是 (batch, seq_len, n_heads, head_dim)
    # 如果您的张量是 (batch, n_heads, seq_len, head_dim)，请转置：
    # q = q.transpose(1, 2)
    # k = k.transpose(1, 2)
    # v = v.transpose(1, 2)
    
    output = flash_attn_func(q, k, v, causal=is_causal, dropout_p=dropout_p)
    return output
```

## 错误处理示例

```python
def safe_flash_attention(q, k, v, **kwargs):
    """带有回退到标准注意力的包装器"""
    try:
        # 首先尝试 FlashAttention
        return flash_attn_func(q, k, v, **kwargs)
    except Exception as e:
        print(f"FlashAttention 失败：{e}，回退到标准注意力")
        
        # 回退实现
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
        return output.transpose(1, 2)  # 返回到 (batch, seq_len, n_heads, head_dim)
```