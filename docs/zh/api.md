# API 参考 - FlashAttention-Plus

## 核心函数

### flash_attn_func

与原始 FlashAttention API 兼容的主要注意力函数。

```python
flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor, 
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False
) -> torch.Tensor
```

**参数：**
- `q`：形状为 `(batch_size, seqlen, nheads, headdim)` 的查询张量
- `k`：形状为 `(batch_size, seqlen, nheads_k, headdim)` 的键张量
- `v`：形状为 `(batch_size, seqlen, nheads_k, headdim)` 的值张量
- `dropout_p`：Dropout 概率（0.0 到 1.0）
- `softmax_scale`：QK^T 的缩放因子。默认：`1/sqrt(headdim)`
- `causal`：是否应用因果掩码
- `window_size`：滑动窗口注意力的（左，右）。默认：无窗口
- `softcap`：注意力分数的软上限值
- `alibi_slopes`：位置偏差的 ALiBi 斜率
- `deterministic`：是否使用确定性算法
- `return_attn_probs`：是否返回注意力概率

**返回：**
- 形状为 `(batch_size, seqlen, nheads, headdim)` 的输出张量
- 如果 `return_attn_probs=True`，还返回注意力概率

### flash_attn_qkvpacked_func

用于打包 QKV 格式的注意力函数。

```python
flash_attn_qkvpacked_func(
    qkv: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False
) -> torch.Tensor
```

**参数：**
- `qkv`：形状为 `(batch_size, seqlen, 3, nheads, headdim)` 的打包 QKV 张量
- 其他参数与 `flash_attn_func` 相同

### flash_attn_kvpacked_func

使用打包 KV 格式的注意力函数。

```python
flash_attn_kvpacked_func(
    q: torch.Tensor,
    kv: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False
) -> torch.Tensor
```

**参数：**
- `q`：形状为 `(batch_size, seqlen, nheads, headdim)` 的查询张量
- `kv`：形状为 `(batch_size, seqlen, 2, nheads_k, headdim)` 的打包 KV 张量
- 其他参数与 `flash_attn_func` 相同

## 环境变量

### FLASH_ATTENTION_USE_FLAGGEMS

控制用于注意力计算的后端。

- `"TRUE"`（默认）：使用 FlagGems/Triton 后端
- `"FALSE"`：尝试使用原始 CUDA 后端

示例：
```python
import os
os.environ["FLASH_ATTENTION_USE_FLAGGEMS"] = "TRUE"
```

## 模块类

### FlashAttention

用于闪存注意力的 PyTorch 模块包装器。

```python
class FlashAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        causal: bool = False,
        window_size: Tuple[int, int] = (-1, -1),
        alibi_slopes: Optional[torch.Tensor] = None
    )
```

**方法：**
- `forward(q, k, v)`：计算注意力
- `reset_parameters()`：重置模块参数

### FlashMHA

使用 FlashAttention 的多头注意力模块。

```python
class FlashMHA(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        causal: bool = False,
        layer_idx: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    )
```

## 数据类型和约束

### 支持的数据类型
- `torch.float16` (FP16)
- `torch.bfloat16` (BF16)

### 张量要求
- 必须在内存中连续
- 必须在 CUDA 设备上
- 序列长度必须可被某些块大小整除（通常为 128）

### 形状约束
- `headdim` 必须是以下之一：32、40、64、80、96、128、160、192、224、256
- `nheads_k` 必须均匀地除以 `nheads`（用于 MQA/GQA）

## 错误处理

常见异常：
- `RuntimeError`：无效的张量形状或不支持的配置
- `ImportError`：FlagGems 未正确安装
- `AssertionError`：约束违反

错误处理示例：
```python
try:
    output = flash_attn_func(q, k, v, causal=True)
except RuntimeError as e:
    print(f"FlashAttention 错误：{e}")
    # 回退到标准注意力
    output = standard_attention(q, k, v)
```