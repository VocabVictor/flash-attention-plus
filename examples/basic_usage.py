#!/usr/bin/env python3
"""
Basic usage example for Flash-Attention-Plus with FlagGems backend
"""

import os
import torch
from flash_attn import flash_attn_func, flash_attn_qkvpacked_func

# Ensure we use FlagGems backend
os.environ["FLASH_ATTENTION_USE_FLAGGEMS"] = "TRUE"

def example_basic_attention():
    """Basic Flash Attention usage"""
    print("=== Basic Flash Attention Example ===")
    
    # Parameters
    batch_size = 2
    seq_len = 1024
    num_heads = 16
    head_dim = 64
    
    # Create input tensors (must be fp16 or bf16 for FlagGems)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    
    # Run Flash Attention
    output = flash_attn_func(q, k, v, causal=True)
    
    print(f"Input shape: {q.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output mean: {output.mean().item():.4f}")
    print(f"Output std: {output.std().item():.4f}")
    print()


def example_qkv_packed():
    """QKV packed format example"""
    print("=== QKV Packed Format Example ===")
    
    # Parameters
    batch_size = 4
    seq_len = 512
    num_heads = 8
    head_dim = 64
    
    # Create QKV packed tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    
    qkv = torch.randn(batch_size, seq_len, 3, num_heads, head_dim, device=device, dtype=dtype)
    
    # Run Flash Attention with packed QKV
    output = flash_attn_qkvpacked_func(qkv, causal=True)
    
    print(f"QKV packed shape: {qkv.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output mean: {output.mean().item():.4f}")
    print(f"Output std: {output.std().item():.4f}")
    print()


def example_with_mask():
    """Example with custom parameters"""
    print("=== Flash Attention with Custom Parameters ===")
    
    # Parameters
    batch_size = 1
    seq_len = 256
    num_heads = 4
    head_dim = 64
    
    # Create input tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    
    # Run with custom parameters
    output = flash_attn_func(
        q, k, v,
        dropout_p=0.1,           # Dropout probability
        softmax_scale=0.125,     # Custom scaling factor
        causal=True,             # Causal mask
        window_size=(-1, -1),    # No sliding window
    )
    
    print(f"Input shape: {q.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Custom parameters applied successfully!")
    print()


def main():
    """Run all examples"""
    print("Flash-Attention-Plus Examples")
    print("Using FlagGems Triton Backend")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Examples may not work properly.")
        return
    
    example_basic_attention()
    example_qkv_packed()
    example_with_mask()
    
    print("All examples completed successfully!")


if __name__ == "__main__":
    main()