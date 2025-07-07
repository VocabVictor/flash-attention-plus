# 迁移指南

本指南帮助您从原始 FlashAttention 迁移到 FlashAttention-Plus。

## 快速迁移

由于 FlashAttention-Plus 保持 API 兼容性，迁移过程非常简单：

### 步骤 1：安装 FlashAttention-Plus

按照[安装指南](installation.md)安装 FlashAttention-Plus 及其依赖项。

### 步骤 2：启用 FlagGems 后端

在导入 flash_attn 之前添加这一行：

```python
import os
os.environ["FLASH_ATTENTION_USE_FLAGGEMS"] = "TRUE"
```

### 步骤 3：无需修改代码

您现有的代码无需修改即可工作：

```python
# 原始代码 - 无需更改！
from flash_attn import flash_attn_func

output = flash_attn_func(q, k, v, causal=True)
```

## 详细迁移

### 训练脚本

```python
# 之前：原始 FlashAttention
import torch
from flash_attn import flash_attn_func

# 之后：FlashAttention-Plus
import os
os.environ["FLASH_ATTENTION_USE_FLAGGEMS"] = "TRUE"  # 添加这一行

import torch
from flash_attn import flash_attn_func  # 相同的导入
```

### 模型定义

如果您有自定义注意力模块：

```python
# 模块定义保持不变
class MyFlashAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        # 无需更改
        
    def forward(self, x):
        # flash_attn_func 工作方式相同
        return flash_attn_func(q, k, v, causal=self.causal)
```

## 功能兼容性

| 功能 | 原始 FlashAttention | FlashAttention-Plus |
|---------|------------------------|--------------------|
| 前向传播 | ✅ | ✅ |
| 反向传播 | ✅ | ❌ (即将推出) |
| 因果掩码 | ✅ | ✅ |
| Dropout | ✅ | ⚠️ (有限支持) |
| 自定义 Softmax 缩放 | ✅ | ✅ |
| FP16/BF16 支持 | ✅ | ✅ |
| 可变长度 | ✅ | ❌ (即将推出) |
| KV 缓存 | ✅ | ❌ (即将推出) |

## 常见迁移场景

### 场景 1：研究项目

对于使用 FlashAttention 的研究项目：

1. 在现有设置旁边安装 FlashAttention-Plus
2. 在脚本开头设置环境变量
3. 像往常一样运行实验

### 场景 2：生产系统

对于生产部署：

1. 使用您的特定工作负载进行彻底测试
2. 监控性能指标
3. 保持切换回的能力：

```python
USE_FLAGGEMS = os.getenv("USE_FLAGGEMS", "true").lower() == "true"

if USE_FLAGGEMS:
    os.environ["FLASH_ATTENTION_USE_FLAGGEMS"] = "TRUE"
else:
    os.environ["FLASH_ATTENTION_USE_FLAGGEMS"] = "FALSE"
```

### 场景 3：多 GPU 训练

FlashAttention-Plus 适用于分布式训练：

```python
# 适用于 DDP、FSDP 等
model = YourModel()
model = DDP(model)

# FlashAttention-Plus 将在所有 GPU 上工作
```

## 故障排除迁移问题

### 问题：性能回退

**解决方案**：由于 Triton 编译，首次运行可能较慢。运行预热：

```python
# 预热运行
for _ in range(3):
    _ = flash_attn_func(q[:1], k[:1], v[:1])

# 实际计算
output = flash_attn_func(q, k, v)
```

### 问题：反向传播不工作

**当前限制**：反向传播尚未实现。对于训练，您可能需要暂时继续使用原始 FlashAttention。

### 问题：导入错误

**解决方案**：确保所有依赖项都已正确安装：

```bash
pip install --upgrade triton
pip install -e /path/to/FlagGems
pip install -e /path/to/flash-attention-plus
```

## 回滚计划

如果您需要回滚到原始 FlashAttention：

```python
# 只需将环境变量设置为 FALSE
os.environ["FLASH_ATTENTION_USE_FLAGGEMS"] = "FALSE"
```

或卸载 FlashAttention-Plus：

```bash
pip uninstall flash-attn-plus
```

## 下一步

- 查看[示例](examples.md)获取工作代码示例
- 阅读[技术细节](technical.md)了解实现细节
- 在 [GitHub](https://github.com/VocabVictor/flash-attention-plus/issues) 上报告问题