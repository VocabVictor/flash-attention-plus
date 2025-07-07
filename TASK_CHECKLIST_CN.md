# Flash Attention Plus 任务清单

## 当前项目状态 ✅

### 已完成的任务
- [x] 环境设置和 FlagGems 集成
- [x] 移除 USE_FLAGGEMS 条件判断
- [x] 实现 `_flaggems_flash_attn_forward()` 标准前向传播
- [x] 实现 `_flash_attn_qkvpacked_forward()` QKV 打包格式
- [x] 实现 `_flash_attn_kvpacked_forward()` KV 打包格式  
- [x] 实现 `_flash_attn_varlen_forward()` 变长序列支持
- [x] 修复递归调用问题
- [x] 创建完整的测试套件
- [x] 编写中文文档和实现计划

## 下一阶段任务：反向传播实现 🚧

### 阶段 2.1：FlagGems 反向传播调研（预估 1-2 周）

#### 任务 2.1.1：分析 FlagGems 源码结构
**优先级：高 🔴**
```bash
# 执行以下命令了解 FlagGems 反向传播能力
cd /home/Master/YangKY/.code/library/FlagGems

# 1. 搜索反向传播相关代码
find . -name "*.py" -exec grep -l "backward\|grad\|autograd" {} \;

# 2. 查看注意力相关的反向传播
find . -name "*.py" -exec grep -l "attention.*backward" {} \;

# 3. 分析 Triton 内核实现
find . -name "*.py" -exec grep -l "@triton" {} \;

# 4. 查看 autograd 集成
grep -r "torch.autograd.Function" src/
```

**输出要求：**
- [ ] 创建 `docs/flaggems_backward_analysis.md` 报告
- [ ] 列出所有相关的反向传播函数
- [ ] 分析现有的 autograd 集成模式
- [ ] 评估实现反向传播的技术可行性

#### 任务 2.1.2：研究 Flash Attention 反向传播算法
**优先级：高 🔴**

**学习资源：**
1. Flash Attention 论文中的反向传播算法
2. 原始 CUDA 实现的反向传播代码
3. PyTorch 原生注意力的反向传播

**需要理解的核心算法：**
```python
# Flash Attention 反向传播的数学原理
"""
给定：
- dOut: 输出梯度 (B, L, H, D)
- Q, K, V: 输入张量
- Softmax_LSE: log-sum-exp 值
- Attention_weights: 注意力权重

计算：
1. dV = Attention_weights.T @ dOut
2. dAttn = dOut @ V.T  
3. dAttn_softmax = softmax_backward(dAttn, Attention_weights)
4. dScores = dAttn_softmax
5. dQ = dScores @ K * scale
6. dK = dScores.T @ Q * scale
"""
```

**输出要求：**
- [ ] 创建 `docs/flash_attention_backward_algorithm.md`
- [ ] 实现算法的数学推导
- [ ] 用 PyTorch 写出参考实现
- [ ] 分析内存和计算复杂度

#### 任务 2.1.3：设计 FlagGems 反向传播架构
**优先级：中 🟡**

**文件结构设计：**
```
flash_attn/
├── backward/
│   ├── __init__.py
│   ├── flaggems_backward.py      # FlagGems 反向传播核心
│   ├── autograd_functions.py     # PyTorch autograd 集成
│   └── reference_impl.py         # PyTorch 参考实现
```

**输出要求：**
- [ ] 设计详细的 API 接口
- [ ] 创建模块文件结构
- [ ] 定义函数签名和参数规范
- [ ] 制定测试策略

### 阶段 2.2：基础反向传播实现（预估 2-3 周）

#### 任务 2.2.1：创建反向传播模块结构
**优先级：高 🔴**

```bash
# 创建目录结构
mkdir -p flash_attn/backward
touch flash_attn/backward/__init__.py
touch flash_attn/backward/flaggems_backward.py
touch flash_attn/backward/autograd_functions.py
touch flash_attn/backward/reference_impl.py
```

**文件内容模板：**

`flash_attn/backward/__init__.py`:
```python
"""Flash Attention 反向传播模块"""

from .flaggems_backward import (
    flaggems_flash_attn_backward,
    flaggems_flash_attn_varlen_backward,
)
from .autograd_functions import (
    FlashAttnFunction,
    FlashAttnVarlenFunction,
)

__all__ = [
    'flaggems_flash_attn_backward',
    'flaggems_flash_attn_varlen_backward', 
    'FlashAttnFunction',
    'FlashAttnVarlenFunction',
]
```

**输出要求：**
- [ ] 创建完整的模块文件结构
- [ ] 实现基础的导入和导出
- [ ] 添加适当的文档字符串

#### 任务 2.2.2：实现 PyTorch 参考版本
**优先级：高 🔴**

`flash_attn/backward/reference_impl.py`:
```python
import torch
import torch.nn.functional as F
from typing import Tuple, Optional

def pytorch_flash_attention_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    PyTorch 原生实现的 Flash Attention 反向传播
    用于验证 FlagGems 实现的正确性
    """
    # TODO: 实现完整的反向传播算法
    pass

# 单元测试
def test_reference_implementation():
    """测试参考实现的正确性"""
    # TODO: 与 PyTorch 原生 scaled_dot_product_attention 对比
    pass
```

**输出要求：**
- [ ] 完整的 PyTorch 参考实现
- [ ] 通过与原生 PyTorch 函数的对比验证
- [ ] 处理各种边界情况（causal, different shapes等）

#### 任务 2.2.3：实现 FlagGems 反向传播核心
**优先级：高 🔴**

`flash_attn/backward/flaggems_backward.py`:
```python
import torch
from typing import Tuple, Optional
import sys
import os

# 添加 FlagGems 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../FlagGems/src'))

try:
    from flag_gems.ops.attention import flash_attention_backward
    FLAGGEMS_BACKWARD_AVAILABLE = True
except ImportError:
    FLAGGEMS_BACKWARD_AVAILABLE = False
    print("警告: FlagGems 反向传播不可用，将使用 PyTorch 参考实现")

def flaggems_flash_attn_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
    window_size_left: int = -1,
    window_size_right: int = -1,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    使用 FlagGems 后端的 Flash Attention 反向传播
    """
    if not FLAGGEMS_BACKWARD_AVAILABLE:
        # 回退到参考实现
        from .reference_impl import pytorch_flash_attention_backward
        return pytorch_flash_attention_backward(
            dout, q, k, v, out, softmax_lse, causal, softmax_scale
        )
    
    # TODO: 使用 FlagGems 实现反向传播
    # 这里需要调用 FlagGems 的反向传播函数
    
    # 暂时抛出未实现错误
    raise NotImplementedError("FlagGems 反向传播实现待完成")
```

**输出要求：**
- [ ] 完整的函数接口实现
- [ ] 错误处理和回退机制
- [ ] 参数验证和类型检查
- [ ] 与 FlagGems API 的正确集成

#### 任务 2.2.4：实现 AutoGrad 集成
**优先级：高 🔴**

`flash_attn/backward/autograd_functions.py`:
```python
import torch
from typing import Optional, Tuple
from .flaggems_backward import flaggems_flash_attn_backward

class FlashAttnFunction(torch.autograd.Function):
    """Flash Attention AutoGrad 函数，支持自动微分"""
    
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor, 
        v: torch.Tensor,
        causal: bool = False,
        softmax_scale: Optional[float] = None,
        window_size_left: int = -1,
        window_size_right: int = -1,
        alibi_slopes: Optional[torch.Tensor] = None,
        return_softmax_lse: bool = False,
    ):
        # 前向传播 - 使用现有的 FlagGems 实现
        from ..flash_attn_flaggems_backend import _flaggems_flash_attn_forward
        
        out, softmax_lse, S_dmask, rng_state = _flaggems_flash_attn_forward(
            q, k, v,
            dropout_p=0.0,  # AutoGrad 版本不支持 dropout
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            alibi_slopes=alibi_slopes,
            return_softmax=False,
        )
        
        # 保存反向传播需要的张量
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.causal = causal
        ctx.softmax_scale = softmax_scale
        ctx.window_size_left = window_size_left
        ctx.window_size_right = window_size_right
        ctx.alibi_slopes = alibi_slopes
        
        if return_softmax_lse:
            return out, softmax_lse
        return out
    
    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        
        # 调用 FlagGems 反向传播
        dq, dk, dv = flaggems_flash_attn_backward(
            dout, q, k, v, out, softmax_lse,
            causal=ctx.causal,
            softmax_scale=ctx.softmax_scale,
            window_size_left=ctx.window_size_left,
            window_size_right=ctx.window_size_right,
            alibi_slopes=ctx.alibi_slopes,
        )
        
        return dq, dk, dv, None, None, None, None, None, None

# 便捷的 API 函数
def flash_attn_func_with_grad(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
    **kwargs
) -> torch.Tensor:
    """支持梯度的 Flash Attention 函数"""
    return FlashAttnFunction.apply(q, k, v, causal, softmax_scale, **kwargs)
```

**输出要求：**
- [ ] 完整的 AutoGrad 函数实现
- [ ] 正确的梯度传播
- [ ] 适当的上下文保存和恢复
- [ ] 便捷的 API 封装

### 阶段 2.3：测试和验证（预估 1 周）

#### 任务 2.3.1：创建反向传播测试套件
**优先级：高 🔴**

```bash
# 创建测试目录
mkdir -p tests/backward
touch tests/backward/__init__.py
touch tests/backward/test_backward_correctness.py
touch tests/backward/test_autograd_integration.py
touch tests/backward/test_performance.py
```

`tests/backward/test_backward_correctness.py`:
```python
import torch
import pytest
from flash_attn.backward.autograd_functions import flash_attn_func_with_grad

class TestBackwardCorrectness:
    """测试反向传播的数值正确性"""
    
    @pytest.mark.parametrize("batch_size,seq_len,num_heads,head_dim", [
        (1, 32, 4, 64),
        (2, 128, 8, 64), 
        (4, 512, 12, 80),
    ])
    def test_gradient_shapes(self, batch_size, seq_len, num_heads, head_dim):
        """测试梯度形状正确性"""
        device = torch.device('cuda')
        
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, 
                       device=device, requires_grad=True)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim,
                       device=device, requires_grad=True)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim,
                       device=device, requires_grad=True)
        
        out = flash_attn_func_with_grad(q, k, v)
        loss = out.sum()
        loss.backward()
        
        assert q.grad is not None
        assert k.grad is not None  
        assert v.grad is not None
        assert q.grad.shape == q.shape
        assert k.grad.shape == k.shape
        assert v.grad.shape == v.shape
    
    def test_gradient_finite_difference(self):
        """使用有限差分验证梯度正确性"""
        device = torch.device('cuda')
        B, L, H, D = 2, 64, 4, 32
        eps = 1e-3
        
        q = torch.randn(B, L, H, D, device=device, dtype=torch.float64)
        k = torch.randn(B, L, H, D, device=device, dtype=torch.float64)
        v = torch.randn(B, L, H, D, device=device, dtype=torch.float64)
        
        def func(q, k, v):
            return flash_attn_func_with_grad(q, k, v).sum()
        
        # 计算数值梯度
        grad_q_numerical = torch.zeros_like(q)
        for i in range(q.numel()):
            q_plus = q.clone().flatten()
            q_minus = q.clone().flatten()
            q_plus[i] += eps
            q_minus[i] -= eps
            
            q_plus = q_plus.reshape(q.shape)
            q_minus = q_minus.reshape(q.shape)
            
            grad_q_numerical.flatten()[i] = (
                func(q_plus, k, v) - func(q_minus, k, v)
            ) / (2 * eps)
        
        # 计算自动微分梯度
        q_ad = q.clone().requires_grad_(True)
        k_ad = k.clone().requires_grad_(True)  
        v_ad = v.clone().requires_grad_(True)
        
        out = func(q_ad, k_ad, v_ad)
        out.backward()
        
        # 比较梯度
        torch.testing.assert_close(
            q_ad.grad, grad_q_numerical, rtol=1e-3, atol=1e-3
        )
    
    def test_causal_gradients(self):
        """测试因果掩码下的梯度正确性"""
        # TODO: 实现因果掩码梯度测试
        pass
        
    def test_vs_pytorch_native(self):
        """与 PyTorch 原生实现对比"""
        device = torch.device('cuda')
        B, L, H, D = 2, 32, 4, 64
        
        # 生成相同的输入
        torch.manual_seed(42)
        q1 = torch.randn(B, L, H, D, device=device, requires_grad=True)
        k1 = torch.randn(B, L, H, D, device=device, requires_grad=True)
        v1 = torch.randn(B, L, H, D, device=device, requires_grad=True)
        
        torch.manual_seed(42)
        q2 = torch.randn(B, L, H, D, device=device, requires_grad=True)
        k2 = torch.randn(B, L, H, D, device=device, requires_grad=True)
        v2 = torch.randn(B, L, H, D, device=device, requires_grad=True)
        
        # Flash Attention 实现
        out1 = flash_attn_func_with_grad(q1, k1, v1)
        loss1 = out1.sum()
        loss1.backward()
        
        # PyTorch 原生实现
        scale = (D ** -0.5)
        scores = torch.matmul(q2, k2.transpose(-2, -1)) * scale
        attn = torch.softmax(scores, dim=-1)
        out2 = torch.matmul(attn, v2)
        loss2 = out2.sum()
        loss2.backward()
        
        # 比较结果（允许一定误差）
        torch.testing.assert_close(out1, out2, rtol=1e-2, atol=1e-3)
        torch.testing.assert_close(q1.grad, q2.grad, rtol=1e-2, atol=1e-3)
```

**输出要求：**
- [ ] 完整的测试套件
- [ ] 形状正确性验证
- [ ] 数值正确性验证（有限差分）
- [ ] 与 PyTorch 原生实现对比
- [ ] 边界条件测试

#### 任务 2.3.2：性能测试和基准
**优先级：中 🟡**

`tests/backward/test_performance.py`:
```python
import torch
import time
import pytest
from flash_attn.backward.autograd_functions import flash_attn_func_with_grad

class TestBackwardPerformance:
    """测试反向传播性能"""
    
    @pytest.mark.parametrize("seq_len", [128, 512, 1024, 2048])
    def test_backward_timing(self, seq_len):
        """测试不同序列长度下的反向传播时间"""
        device = torch.device('cuda')
        B, H, D = 4, 8, 64
        
        q = torch.randn(B, seq_len, H, D, device=device, requires_grad=True)
        k = torch.randn(B, seq_len, H, D, device=device, requires_grad=True)
        v = torch.randn(B, seq_len, H, D, device=device, requires_grad=True)
        
        # 预热
        for _ in range(10):
            out = flash_attn_func_with_grad(q, k, v)
            loss = out.sum()
            loss.backward()
            q.grad = None
            k.grad = None
            v.grad = None
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(100):
            out = flash_attn_func_with_grad(q, k, v)
            loss = out.sum()
            loss.backward()
            q.grad = None
            k.grad = None
            v.grad = None
            
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100 * 1000  # ms
        print(f"序列长度 {seq_len}: 平均反向传播时间 {avg_time:.2f} ms")
        
        # 性能要求（这些数字需要根据实际测试调整）
        if seq_len <= 512:
            assert avg_time < 50  # 小于 50ms
        elif seq_len <= 1024:
            assert avg_time < 200  # 小于 200ms
    
    def test_memory_usage(self):
        """测试内存使用情况"""
        device = torch.device('cuda')
        B, L, H, D = 4, 1024, 8, 64
        
        # 重置内存统计
        torch.cuda.reset_peak_memory_stats()
        
        q = torch.randn(B, L, H, D, device=device, requires_grad=True)
        k = torch.randn(B, L, H, D, device=device, requires_grad=True)
        v = torch.randn(B, L, H, D, device=device, requires_grad=True)
        
        out = flash_attn_func_with_grad(q, k, v)
        loss = out.sum()
        loss.backward()
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        print(f"峰值内存使用: {peak_memory:.2f} MB")
        
        # 内存使用不应该过高
        expected_memory = B * L * H * D * 4 * 6 / 1024 / 1024  # 大致估算
        assert peak_memory < expected_memory * 3  # 允许 3 倍的开销
```

**输出要求：**
- [ ] 性能基准测试
- [ ] 内存使用分析
- [ ] 与期望性能的对比
- [ ] 性能回归检测

## 阶段 3：变长序列反向传播（预估 1-2 周）

### 任务 3.1：实现变长序列反向传播
**优先级：中 🟡**

```python
# flash_attn/backward/flaggems_backward.py 中添加

def flaggems_flash_attn_varlen_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """变长序列的反向传播实现"""
    
    # TODO: 实现变长序列反向传播
    # 关键挑战：
    # 1. 处理不同长度序列的梯度计算
    # 2. 内存高效的实现
    # 3. 与前向传播的一致性
    
    # 暂时使用批次转换的回退方案
    batch_size = len(cu_seqlens_q) - 1
    
    # 转换为批次格式，计算梯度，再转换回变长格式
    # 具体实现细节待完成
    
    raise NotImplementedError("变长序列反向传播待实现")
```

## 阶段 4：KV 缓存实现（预估 2-3 周）

### 任务 4.1：KV 缓存架构设计
**优先级：低 🟢**

```bash
# 创建 KV 缓存模块
mkdir -p flash_attn/kvcache
touch flash_attn/kvcache/__init__.py
touch flash_attn/kvcache/cache_manager.py
touch flash_attn/kvcache/kv_attention.py
```

**待实现的功能：**
- [ ] KV 缓存管理器
- [ ] 缓存更新策略
- [ ] 内存优化
- [ ] 与现有 API 的集成

## 快速开始指南

### 立即可以开始的任务

1. **任务 2.1.1**: FlagGems 源码分析
   ```bash
   cd /home/Master/YangKY/.code/library/FlagGems
   find . -name "*.py" -exec grep -l "backward" {} \; > backward_files.txt
   cat backward_files.txt
   ```

2. **任务 2.1.2**: 学习 Flash Attention 反向传播算法
   - 阅读论文第 3.2 节
   - 分析原始 CUDA 代码中的反向传播实现
   - 用 PyTorch 实现一个简单的参考版本

3. **任务 2.2.1**: 创建模块结构
   ```bash
   cd /home/Master/YangKY/.code/library/flash-attention-plus
   mkdir -p flash_attn/backward tests/backward
   # 创建基础文件
   ```

### 验证当前实现

运行以下命令确保当前的前向传播实现正常工作：

```bash
micromamba activate dev
cd /home/Master/YangKY/.code/library/flash-attention-plus

python -c "
import torch
import flash_attn

# 测试所有前向传播功能
device = torch.device('cuda')

# 1. 标准注意力
q = torch.randn(2, 128, 8, 64, dtype=torch.float16, device=device)
k = torch.randn(2, 128, 8, 64, dtype=torch.float16, device=device)
v = torch.randn(2, 128, 8, 64, dtype=torch.float16, device=device)
out = flash_attn.flash_attn_func(q, k, v, causal=True)
print(f'✅ 标准注意力: {out.shape}')

# 2. 变长序列
q_varlen = torch.randn(256, 8, 64, dtype=torch.float16, device=device)
k_varlen = torch.randn(256, 8, 64, dtype=torch.float16, device=device)
v_varlen = torch.randn(256, 8, 64, dtype=torch.float16, device=device)
cu_seqlens = torch.tensor([0, 64, 128, 192, 256], dtype=torch.int32, device=device)
out_varlen = flash_attn.flash_attn_varlen_func(
    q_varlen, k_varlen, v_varlen, cu_seqlens, cu_seqlens, 64, 64, causal=True
)
print(f'✅ 变长序列: {out_varlen.shape}')

print('🎉 当前实现工作正常，可以开始反向传播开发！')
"
```

### 推荐的实现顺序

1. **Week 1**: 任务 2.1.1 和 2.1.2 - 调研和学习
2. **Week 2**: 任务 2.1.3 和 2.2.1 - 设计和结构搭建
3. **Week 3**: 任务 2.2.2 - PyTorch 参考实现
4. **Week 4**: 任务 2.2.3 - FlagGems 集成尝试
5. **Week 5**: 任务 2.2.4 和 2.3.1 - AutoGrad 集成和测试
6. **Week 6**: 任务 2.3.2 和优化 - 性能测试和调优

每完成一个任务，记得更新这个清单并提交代码！

## 进度跟踪

请在完成每个任务后，在对应的 `[ ]` 中打上 `[x]` 标记进度。

当前整体进度：**第一阶段完成 ✅，第二阶段等待开始 🚧**