# FlashAttention-Plus

**基于 FlagGems/Triton 后端的硬件无关 FlashAttention 实现**

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://vocabvictor.github.io/flash-attention-plus/)

[📖 English Documentation](README_EN.md) | [中文文档](README_CN.md)

## 项目概述

FlashAttention-Plus 是原始 [FlashAttention](https://github.com/Dao-AILab/flash-attention) 的直接替代品，使用 [FlagGems](https://github.com/FlagOpen/FlagGems) 的 Triton 实现替代 NVIDIA CUDA 内核。这使得 FlashAttention 能够在更广泛的硬件上运行，同时保持 API 兼容性。

**主要特性：**
- 🚀 **硬件无关**：使用 Triton 而非 CUDA 专用代码
- 🔄 **API 兼容**：原始 FlashAttention 的直接替代品
- ⚡ **高性能**：利用 FlagGems 优化的 Triton 内核
- 🎯 **易于集成**：只需最少的代码更改

## 安装说明

### 环境要求

```bash
# 支持 CUDA 的 PyTorch
pip install torch>=2.0.0

# Triton (FlagGems 所需)
pip install triton>=3.0.0

# 其他依赖
pip install einops
```

### 安装 FlagGems

```bash
cd ~/.code/library/FlagGems  # 或你喜欢的位置
git clone https://github.com/FlagOpen/FlagGems.git
cd FlagGems
pip install -e .
```

### 安装 FlashAttention-Plus

```bash
git clone https://github.com/VocabVictor/flash-attention-plus.git
cd flash-attention-plus
pip install -e .
```

## 使用方法

### 快速开始

当前版本直接使用 FlagGems 后端，无需设置环境变量：

```python
import torch
from flash_attn import flash_attn_func

# 创建张量 (必须是 fp16 或 bf16)
q = torch.randn(2, 1024, 16, 64, device='cuda', dtype=torch.float16)
k = torch.randn(2, 1024, 16, 64, device='cuda', dtype=torch.float16)
v = torch.randn(2, 1024, 16, 64, device='cuda', dtype=torch.float16)

# 运行 flash attention
output = flash_attn_func(q, k, v, causal=True)
```

### 支持的功能

```python
# 标准注意力
output = flash_attn_func(q, k, v, causal=True)

# QKV 打包格式
qkv = torch.randn(2, 1024, 3, 16, 64, device='cuda', dtype=torch.float16)
output = flash_attn_qkvpacked_func(qkv, causal=True)

# 变长序列
output = flash_attn_varlen_func(q_varlen, k_varlen, v_varlen, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
```

## 与原始 FlashAttention 的主要差异

### 已更改的部分
- **后端**：CUDA 内核 → FlagGems/Triton 内核
- **硬件支持**：仅支持 NVIDIA → 硬件无关
- **安装**：无需 CUDA 编译

### 保留的部分
- ✅ API 兼容性
- ✅ 核心功能
- ✅ 模型支持 (BERT, GPT, LLaMA 等)
- ✅ 性能特性

### 当前状态
- ✅ **前向传播**：完全实现
  - 标准注意力 (flash_attn_func)
  - QKV 打包格式 (flash_attn_qkvpacked_func)
  - KV 打包格式 (flash_attn_kvpacked_func)
  - 变长序列 (flash_attn_varlen_func)
- ❌ **反向传播**：未实现 (仅推理)
- ❌ **KV 缓存**：待开发

## 示例

查看更多使用示例请参考文档：
- `README_CN.md` - 详细的使用指南
- `IMPLEMENTATION_PLAN_CN.md` - 实现计划
- `TASK_CHECKLIST_CN.md` - 开发任务清单

## 性能

性能因硬件和配置而异。首次运行可能较慢（由于 Triton 内核编译），但后续运行使用缓存内核。

## 技术细节

更多关于 FlagGems 集成的信息，请参阅：
- `PROJECT_STATUS_CN.md` - 项目状态报告
- `IMPLEMENTATION_PLAN_CN.md` - 详细技术计划

## 许可证

本项目采用与原始 FlashAttention 相同的 BSD 3-Clause 许可证。详见 [LICENSE](LICENSE)。

## 致谢

- 原始 FlashAttention 由 [Tri Dao](https://tridao.me/) 和团队开发
- [FlagGems](https://github.com/FlagOpen/FlagGems) 团队提供 Triton 内核
- [OpenAI Triton](https://github.com/openai/triton) GPU 编程语言

## 发展路线图

- [ ] 实现反向传播支持
- [ ] 添加 KV 缓存功能
- [ ] 性能优化
- [ ] 全面的基准测试
- [ ] 支持更多硬件后端

详细的开发计划请参阅 `IMPLEMENTATION_PLAN_CN.md`。