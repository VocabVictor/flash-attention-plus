# 安装指南

本指南将帮助您安装 FlashAttention-Plus 及其依赖项。

## 前置要求

在安装 FlashAttention-Plus 之前，请确保您拥有：

- Python 3.8 或更高版本
- PyTorch 2.0 或更高版本（带 CUDA 支持）
- CUDA 11.6 或更高版本（用于 GPU 支持）

## 步骤 1：安装核心依赖

```bash
# 安装带 CUDA 支持的 PyTorch
pip install torch>=2.0.0

# 安装 Triton（FlagGems 所需）
pip install triton>=3.0.0

# 其他依赖
pip install einops
```

!!! note "Triton 版本"
    FlagGems 需要 Triton 3.0 或更高版本。如果您预装的 PyTorch 带有较旧的 Triton 版本，可能需要升级：
    ```bash
    pip install --upgrade triton
    ```

## 步骤 2：安装 FlagGems

FlagGems 提供基于 Triton 的 Flash Attention 实现：

```bash
# 克隆 FlagGems 仓库
git clone https://github.com/FlagOpen/FlagGems.git
cd FlagGems

# 以开发模式安装
pip install -e .
```

## 步骤 3：安装 FlashAttention-Plus

```bash
# 克隆仓库
git clone https://github.com/VocabVictor/flash-attention-plus.git
cd flash-attention-plus

# 以开发模式安装
pip install -e .
```

## 验证安装

验证 FlashAttention-Plus 是否正确安装：

```python
import torch
from flash_attn import flash_attn_func

# 检查导入是否成功
print("FlashAttention-Plus 导入成功！")

# 简单测试
q = torch.randn(1, 64, 8, 32, device='cuda', dtype=torch.float16)
k = torch.randn(1, 64, 8, 32, device='cuda', dtype=torch.float16)
v = torch.randn(1, 64, 8, 32, device='cuda', dtype=torch.float16)

output = flash_attn_func(q, k, v)
print(f"测试通过！输出形状：{output.shape}")
```

## 故障排除

### 常见问题

#### 1. NumPy 版本冲突

如果遇到 NumPy 版本问题：
```bash
pip install numpy==1.26.4
```

#### 2. Triton 导入错误

如果看到与 Triton 导入相关的错误：
```bash
# 将 Triton 升级到最新版本
pip install --upgrade triton
```

#### 3. CUDA 不可用

确保 PyTorch 安装了 CUDA 支持：
```python
import torch
print(torch.cuda.is_available())  # 应返回 True
```

### 环境设置

为了获得干净的安装，我们建议使用 conda 环境：

```bash
# 创建新环境
conda create -n flash-attn-plus python=3.10
conda activate flash-attn-plus

# 安装带 CUDA 支持的 PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 继续上述安装步骤
```

## 下一步

- 查看[使用指南](usage.md)了解如何使用 FlashAttention-Plus
- 查看[示例](examples.md)获取实际代码样例
- 如果您是从原始 FlashAttention 迁移，请阅读[迁移指南](migration.md)