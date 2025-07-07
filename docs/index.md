# FlashAttention-Plus

**基于 FlagGems/Triton 后端的硬件无关 FlashAttention 实现**

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

## 概述

FlashAttention-Plus 是原始 [FlashAttention](https://github.com/Dao-AILab/flash-attention) 的直接替代品，它使用 [FlagGems](https://github.com/FlagOpen/FlagGems) 的 Triton 实现替换了 NVIDIA CUDA 内核。这使得 FlashAttention 能够在更广泛的硬件上运行，同时保持 API 兼容性。

**主要特性：**

- 🚀 **硬件无关**：使用 Triton 而非 CUDA 特定代码
- 🔄 **API 兼容**：可直接替换原始 FlashAttention
- ⚡ **高性能**：利用 FlagGems 的优化 Triton 内核
- 🎯 **易于集成**：只需最少的代码更改

## 为什么选择 FlashAttention-Plus？

原始的 FlashAttention 实现提供了出色的性能，但由于其 CUDA 特定的内核，仅限于 NVIDIA GPU。FlashAttention-Plus 通过使用 FlagGems 基于 Triton 的实现来解决这一限制，这可能在各种硬件加速器上运行，同时保持相同的 API。

## 快速示例

```python
import os
import torch

# 启用 FlagGems 后端
os.environ["FLASH_ATTENTION_USE_FLAGGEMS"] = "TRUE"

from flash_attn import flash_attn_func

# 创建张量（必须是 fp16 或 bf16）
batch_size, seq_len, num_heads, head_dim = 2, 1024, 16, 64
q = torch.randn(batch_size, seq_len, num_heads, head_dim, 
                device='cuda', dtype=torch.float16)
k = torch.randn(batch_size, seq_len, num_heads, head_dim, 
                device='cuda', dtype=torch.float16)
v = torch.randn(batch_size, seq_len, num_heads, head_dim, 
                device='cuda', dtype=torch.float16)

# 运行 flash attention
output = flash_attn_func(q, k, v, causal=True)
print(f"输出形状: {output.shape}")
```

## 快速开始

- [安装指南](installation.md) - 设置 FlashAttention-Plus
- [使用指南](usage.md) - 学习如何使用 FlashAttention-Plus
- [迁移指南](migration.md) - 从原始 FlashAttention 迁移
- [API 参考](api.md) - 详细的 API 文档

## 项目状态

本项目正在积极开发中。当前限制包括：

- ❌ 尚未实现反向传播
- ❌ KV 缓存支持待定
- ❌ 不支持可变长度序列
- ⚠️ Dropout 接口存在但可能功能不完整

查看我们的[路线图](#路线图)了解即将推出的功能。

## 路线图

- [ ] 实现反向传播支持
- [ ] 添加 KV 缓存功能
- [ ] 支持可变长度序列
- [ ] 性能优化
- [ ] 全面的基准测试
- [ ] 支持更多硬件后端

## 许可证

本项目与原始 FlashAttention 保持相同的 BSD 3-Clause 许可证。详见 [LICENSE](https://github.com/VocabVictor/flash-attention-plus/blob/main/LICENSE)。

## 致谢

- 原始 FlashAttention 由 [Tri Dao](https://tridao.me/) 及其团队开发
- [FlagGems](https://github.com/FlagOpen/FlagGems) 团队提供 Triton 内核
- [OpenAI Triton](https://github.com/openai/triton) 提供 GPU 编程语言