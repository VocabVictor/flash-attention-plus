# 使用指南

本指南介绍如何在项目中使用 FlashAttention-Plus。

## 基础用法

### 启用 FlagGems 后端

FlashAttention-Plus 使用环境变量来切换后端：

```python
import os

# 启用 FlagGems 后端（默认）
os.environ["FLASH_ATTENTION_USE_FLAGGEMS"] = "TRUE"

# 或禁用以使用原始 CUDA 后端（如果可用）
os.environ["FLASH_ATTENTION_USE_FLAGGEMS"] = "FALSE"
```

### 简单示例

```python
import torch
from flash_attn import flash_attn_func

# 创建输入张量
batch_size = 2
seq_length = 1024
num_heads = 16
head_dim = 64

# 注意：输入必须是 fp16 或 bf16
q = torch.randn(batch_size, seq_length, num_heads, head_dim, 
                device='cuda', dtype=torch.float16)
k = torch.randn(batch_size, seq_length, num_heads, head_dim, 
                device='cuda', dtype=torch.float16)
v = torch.randn(batch_size, seq_length, num_heads, head_dim, 
                device='cuda', dtype=torch.float16)

# 应用闪存注意力
output = flash_attn_func(q, k, v, causal=True)
```

## 高级用法

### 使用 Dropout

```python
# 应用带有 dropout 的注意力
output = flash_attn_func(q, k, v, dropout_p=0.1, causal=True)
```

!!! warning "Dropout 支持"
    Dropout 接口可用，但在当前版本中可能无法完全正常工作。

### 自定义 Softmax 缩放

```python
# 自定义缩放因子
scale = 1.0 / math.sqrt(head_dim)
output = flash_attn_func(q, k, v, softmax_scale=scale, causal=True)
```

### 非因果注意力

```python
# 用于双向注意力（例如 BERT）
output = flash_attn_func(q, k, v, causal=False)
```

## 输入要求

### 数据类型

FlashAttention-Plus 要求输入为半精度格式：

- `torch.float16` (fp16)
- `torch.bfloat16` (bf16)

```python
# 如需要，转换为 fp16
q = q.to(torch.float16)
k = k.to(torch.float16)
v = v.to(torch.float16)
```

### 张量形状

输入张量应具有形状：`[batch_size, seq_length, num_heads, head_dim]`

### 设备

所有张量必须在 CUDA 设备上：

```python
q = q.to('cuda')
k = k.to('cuda')
v = v.to('cuda')
```

## 与 Transformers 集成

### 自定义注意力模块

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
        
        # 计算 Q, K, V
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        
        # 应用闪存注意力
        output = flash_attn_func(q, k, v, dropout_p=self.dropout, causal=causal)
        
        # 重塑并投影输出
        output = output.reshape(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(output)
        
        return output
```

### 与现有模型一起使用

```python
# 用闪存注意力替换标准注意力
model = YourTransformerModel()

# 启用 FlagGems 后端
os.environ["FLASH_ATTENTION_USE_FLAGGEMS"] = "TRUE"

# 模型现在将使用 FlashAttention-Plus
output = model(input_ids)
```

## 性能提示

1. **首次运行**：由于 Triton 内核编译，首次运行可能较慢。后续运行将使用缓存的内核。

2. **批处理**：一起处理多个序列以获得更好的 GPU 利用率：
   ```python
   # 好：批量处理多个序列
   batch_output = flash_attn_func(batch_q, batch_k, batch_v)
   
   # 效率较低：逐个处理
   for i in range(batch_size):
       output_i = flash_attn_func(q[i:i+1], k[i:i+1], v[i:i+1])
   ```

3. **内存效率**：FlashAttention 设计为内存高效。您可以处理比标准注意力更长的序列。

## 调试

启用调试模式以获取更多信息：

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 这将显示有关注意力计算的详细信息
output = flash_attn_func(q, k, v, causal=True)
```

## 下一步

- 查看[示例](examples.md)获取完整的工作示例
- 查看 [API 参考](api.md)了解详细的参数文档
- 阅读[技术细节](technical.md)了解其工作原理