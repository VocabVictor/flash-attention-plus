#!/usr/bin/env python3
"""
Flash Attention Plus 开发环境验证脚本
运行此脚本确保开发环境设置正确
"""

import sys
import os
import torch
import traceback

def check_environment():
    """检查开发环境"""
    print("🔍 检查开发环境...")
    
    # 检查 Python 版本
    python_version = sys.version_info
    print(f"Python 版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    if python_version.major != 3 or python_version.minor < 9:
        print("⚠️  警告: 推荐使用 Python 3.9+")
    else:
        print("✅ Python 版本合适")
    
    # 检查 CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA 可用: {torch.cuda.get_device_name()}")
        print(f"   CUDA 版本: {torch.version.cuda}")
        print(f"   可用 GPU 数量: {torch.cuda.device_count()}")
    else:
        print("❌ CUDA 不可用")
        return False
    
    return True

def check_flaggems():
    """检查 FlagGems 是否可用"""
    print("\n🔍 检查 FlagGems...")
    
    try:
        # 添加 FlagGems 路径
        flaggems_path = "/home/Master/YangKY/.code/library/FlagGems/src"
        if flaggems_path not in sys.path:
            sys.path.insert(0, flaggems_path)
        
        import flag_gems
        print("✅ FlagGems 导入成功")
        
        # 检查具体的注意力函数
        try:
            from flag_gems.ops.attention import flash_attention_forward
            print("✅ flash_attention_forward 可用")
        except ImportError as e:
            print(f"⚠️  flash_attention_forward 导入失败: {e}")
        
        try:
            from flag_gems.ops.flash_api import mha_fwd
            print("✅ mha_fwd 可用")
        except ImportError as e:
            print(f"⚠️  mha_fwd 导入失败: {e}")
            
        return True
        
    except ImportError as e:
        print(f"❌ FlagGems 导入失败: {e}")
        print("请确保 FlagGems 已正确安装并且路径设置正确")
        return False

def check_flash_attention_plus():
    """检查 Flash Attention Plus 实现"""
    print("\n🔍 检查 Flash Attention Plus...")
    
    try:
        import flash_attn
        print("✅ flash_attn 模块导入成功")
        
        # 检查主要函数是否可用
        functions_to_check = [
            'flash_attn_func',
            'flash_attn_qkvpacked_func', 
            'flash_attn_kvpacked_func',
            'flash_attn_varlen_func',
            'flash_attn_with_kvcache',
        ]
        
        for func_name in functions_to_check:
            if hasattr(flash_attn, func_name):
                print(f"✅ {func_name} 可用")
            else:
                print(f"❌ {func_name} 不可用")
                
        return True
        
    except ImportError as e:
        print(f"❌ flash_attn 模块导入失败: {e}")
        return False

def test_forward_functions():
    """测试前向传播函数"""
    print("\n🧪 测试前向传播函数...")
    
    try:
        import flash_attn
        device = torch.device('cuda')
        
        # 测试标准注意力
        print("测试 flash_attn_func...")
        q = torch.randn(2, 64, 4, 32, dtype=torch.float16, device=device)
        k = torch.randn(2, 64, 4, 32, dtype=torch.float16, device=device)
        v = torch.randn(2, 64, 4, 32, dtype=torch.float16, device=device)
        
        out = flash_attn.flash_attn_func(q, k, v, causal=True)
        print(f"✅ flash_attn_func 输出形状: {out.shape}")
        
        # 测试 QKV 打包
        print("测试 flash_attn_qkvpacked_func...")
        qkv = torch.randn(2, 64, 3, 4, 32, dtype=torch.float16, device=device)
        out_qkv = flash_attn.flash_attn_qkvpacked_func(qkv, causal=True)
        print(f"✅ flash_attn_qkvpacked_func 输出形状: {out_qkv.shape}")
        
        # 测试变长序列
        print("测试 flash_attn_varlen_func...")
        total_len = 128
        q_varlen = torch.randn(total_len, 4, 32, dtype=torch.float16, device=device)
        k_varlen = torch.randn(total_len, 4, 32, dtype=torch.float16, device=device)
        v_varlen = torch.randn(total_len, 4, 32, dtype=torch.float16, device=device)
        cu_seqlens = torch.tensor([0, 32, 64, 96, 128], dtype=torch.int32, device=device)
        
        out_varlen = flash_attn.flash_attn_varlen_func(
            q_varlen, k_varlen, v_varlen, cu_seqlens, cu_seqlens, 32, 32, causal=True
        )
        print(f"✅ flash_attn_varlen_func 输出形状: {out_varlen.shape}")
        
        # 测试 KV 缓存（应该失败）
        print("测试 flash_attn_with_kvcache（期望失败）...")
        try:
            q_cache = torch.randn(1, 32, 4, 32, dtype=torch.float16, device=device)
            k_cache = torch.randn(1, 32, 4, 32, dtype=torch.float16, device=device)
            v_cache = torch.randn(1, 32, 4, 32, dtype=torch.float16, device=device)
            
            out_cache = flash_attn.flash_attn_with_kvcache(q_cache, k_cache, v_cache)
            print(f"❌ flash_attn_with_kvcache 应该失败但成功了: {out_cache.shape}")
        except NotImplementedError:
            print("✅ flash_attn_with_kvcache 正确抛出 NotImplementedError")
        
        return True
        
    except Exception as e:
        print(f"❌ 前向传播测试失败: {e}")
        traceback.print_exc()
        return False

def test_backward_limitations():
    """测试反向传播限制（应该失败）"""
    print("\n🧪 测试反向传播限制...")
    
    try:
        import flash_attn
        device = torch.device('cuda')
        
        q = torch.randn(1, 32, 4, 32, dtype=torch.float16, device=device, requires_grad=True)
        k = torch.randn(1, 32, 4, 32, dtype=torch.float16, device=device, requires_grad=True)
        v = torch.randn(1, 32, 4, 32, dtype=torch.float16, device=device, requires_grad=True)
        
        out = flash_attn.flash_attn_func(q, k, v)
        loss = out.sum()
        
        try:
            loss.backward()
            print("❌ 反向传播应该失败但成功了")
            return False
        except NotImplementedError:
            print("✅ 反向传播正确抛出 NotImplementedError")
            return True
            
    except Exception as e:
        print(f"❌ 反向传播限制测试失败: {e}")
        return False

def check_project_structure():
    """检查项目文件结构"""
    print("\n🔍 检查项目结构...")
    
    required_files = [
        'flash_attn/__init__.py',
        'flash_attn/flash_attn_interface.py',
        'flash_attn/flash_attn_flaggems_backend.py',
        'README.md',
        'README_CN.md', 
        'README_EN.md',
        'IMPLEMENTATION_PLAN_CN.md',
        'TASK_CHECKLIST_CN.md',
        'PROJECT_STATUS_CN.md',
        'IMPLEMENTATION_COMPARISON_CN.md',
        'IMPLEMENTATION_ANALYSIS_CN.md',
        'docs/index.md',
        'docs/api.md',
        'docs/usage.md',
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} 缺失")
            all_exist = False
    
    return all_exist

def main():
    """主函数"""
    print("🚀 Flash Attention Plus 开发环境验证")
    print("=" * 50)
    
    all_checks_passed = True
    
    # 基础环境检查
    if not check_environment():
        all_checks_passed = False
    
    # FlagGems 检查
    if not check_flaggems():
        all_checks_passed = False
    
    # Flash Attention Plus 检查
    if not check_flash_attention_plus():
        all_checks_passed = False
    
    # 项目结构检查
    if not check_project_structure():
        all_checks_passed = False
    
    # 功能测试
    if not test_forward_functions():
        all_checks_passed = False
        
    if not test_backward_limitations():
        all_checks_passed = False
    
    print("\n" + "=" * 50)
    if all_checks_passed:
        print("🎉 所有检查通过！开发环境设置正确。")
        print("\n📋 下一步:")
        print("1. 阅读 README_CN.md 了解项目概况")
        print("2. 查看 IMPLEMENTATION_PLAN_CN.md 了解详细计划")
        print("3. 按照 TASK_CHECKLIST_CN.md 开始实现")
        print("4. 从 '任务 2.1.1: FlagGems 源码分析' 开始")
    else:
        print("❌ 部分检查失败，请修复上述问题后重试。")
    
    return all_checks_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)