#!/usr/bin/env python3
"""
Test script for FlashAttention with FlagGems backend
"""

import os
import sys
import torch
import time
import numpy as np

# Ensure we use FlagGems backend
os.environ["FLASH_ATTENTION_USE_FLAGGEMS"] = "TRUE"

# Add paths
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../FlagGems/src'))

# Import flash attention
from flash_attn import flash_attn_func, flash_attn_qkvpacked_func

def test_flash_attn_basic():
    """Test basic flash attention functionality"""
    print("\n=== Testing Basic Flash Attention ===")
    
    # Test parameters
    batch_size = 2
    seq_len = 512
    num_heads = 8
    head_dim = 64
    
    # Create input tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    
    print(f"Input shapes: Q={q.shape}, K={k.shape}, V={v.shape}")
    
    # Test forward pass
    try:
        start_time = time.time()
        out = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
        torch.cuda.synchronize()
        elapsed_time = time.time() - start_time
        
        print(f"✓ Forward pass successful!")
        print(f"  Output shape: {out.shape}")
        print(f"  Time: {elapsed_time*1000:.2f} ms")
        print(f"  Output stats: mean={out.mean().item():.4f}, std={out.std().item():.4f}")
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False
    
    return True


def test_flash_attn_qkvpacked():
    """Test QKV packed flash attention"""
    print("\n=== Testing QKV Packed Flash Attention ===")
    
    # Test parameters
    batch_size = 2
    seq_len = 256
    num_heads = 4
    head_dim = 64
    
    # Create input tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    
    qkv = torch.randn(batch_size, seq_len, 3, num_heads, head_dim, device=device, dtype=dtype)
    
    print(f"Input shape: QKV={qkv.shape}")
    
    # Test forward pass
    try:
        start_time = time.time()
        out = flash_attn_qkvpacked_func(qkv, dropout_p=0.0, causal=True)
        torch.cuda.synchronize()
        elapsed_time = time.time() - start_time
        
        print(f"✓ Forward pass successful!")
        print(f"  Output shape: {out.shape}")
        print(f"  Time: {elapsed_time*1000:.2f} ms")
        print(f"  Output stats: mean={out.mean().item():.4f}, std={out.std().item():.4f}")
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False
    
    return True


def test_comparison_with_standard_attention():
    """Compare FlagGems implementation with standard PyTorch attention"""
    print("\n=== Testing Accuracy vs Standard Attention ===")
    
    # Test parameters (smaller for comparison)
    batch_size = 1
    seq_len = 128
    num_heads = 4
    head_dim = 64
    
    # Create input tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16  # FlagGems requires fp16 or bf16
    
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    
    # Standard attention computation
    scale = 1.0 / np.sqrt(head_dim)
    q_t = q.transpose(1, 2)  # [batch, heads, seq, dim]
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    
    scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale
    
    # Apply causal mask
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=dtype), diagonal=1) * -10000
    scores = scores + mask
    
    attn_weights = torch.softmax(scores, dim=-1)
    out_standard = torch.matmul(attn_weights, v_t).transpose(1, 2)
    
    # Flash attention computation
    try:
        out_flash = flash_attn_func(
            q, k, v, 
            dropout_p=0.0, 
            causal=True,
            softmax_scale=scale
        )
        
        # Compare outputs
        max_diff = torch.max(torch.abs(out_standard - out_flash)).item()
        mean_diff = torch.mean(torch.abs(out_standard - out_flash)).item()
        
        print(f"✓ Comparison successful!")
        print(f"  Max absolute difference: {max_diff:.6f}")
        print(f"  Mean absolute difference: {mean_diff:.6f}")
        
        # Check if difference is acceptable (relaxed tolerance for different implementations)
        if max_diff < 0.1:  # Relaxed tolerance
            print(f"  ✓ Accuracy test PASSED")
            return True
        else:
            print(f"  ✗ Accuracy test FAILED (difference too large)")
            return False
            
    except Exception as e:
        print(f"✗ Comparison failed: {e}")
        return False


def run_performance_benchmark():
    """Run performance benchmark"""
    print("\n=== Performance Benchmark ===")
    
    # Test different configurations
    configs = [
        (1, 512, 8, 64),    # Small
        (2, 1024, 12, 64),  # Medium
        (4, 2048, 16, 64),  # Large
    ]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    
    for batch_size, seq_len, num_heads, head_dim in configs:
        print(f"\nConfig: batch={batch_size}, seq_len={seq_len}, heads={num_heads}, dim={head_dim}")
        
        # Create inputs
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        
        # Warmup
        for _ in range(3):
            _ = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
        
        # Benchmark
        torch.cuda.synchronize()
        start_time = time.time()
        
        num_iterations = 10
        for _ in range(num_iterations):
            _ = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
        
        torch.cuda.synchronize()
        elapsed_time = (time.time() - start_time) / num_iterations
        
        # Calculate FLOPS
        flops = 4 * batch_size * seq_len * seq_len * num_heads * head_dim  # Approximate
        tflops = flops / elapsed_time / 1e12
        
        print(f"  Time per iteration: {elapsed_time*1000:.2f} ms")
        print(f"  Approximate TFLOPS: {tflops:.2f}")


def main():
    """Main test function"""
    print("=" * 60)
    print("FlashAttention with FlagGems Backend - Test Suite")
    print("=" * 60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Tests may fail or run slowly.")
    else:
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    
    # Run tests
    all_passed = True
    
    # Basic functionality test
    if not test_flash_attn_basic():
        all_passed = False
    
    # QKV packed test
    if not test_flash_attn_qkvpacked():
        all_passed = False
    
    # Accuracy test
    if not test_comparison_with_standard_attention():
        all_passed = False
    
    # Performance benchmark
    try:
        run_performance_benchmark()
    except Exception as e:
        print(f"Performance benchmark failed: {e}")
        all_passed = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests PASSED!")
    else:
        print("✗ Some tests FAILED!")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)