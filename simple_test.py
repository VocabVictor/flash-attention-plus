#!/usr/bin/env python3
"""
Simple test to verify the flash-attention-plus structure
"""

import os
import sys

# Add paths
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 60)
print("Flash-Attention-Plus Structure Test")
print("=" * 60)

# Test 1: Check if the module can be imported
try:
    import flash_attn
    print("✓ Successfully imported flash_attn module")
    print(f"  Version: {flash_attn.__version__}")
except Exception as e:
    print(f"✗ Failed to import flash_attn: {e}")
    sys.exit(1)

# Test 2: Check if the interface module exists
try:
    from flash_attn import flash_attn_interface
    print("✓ Successfully imported flash_attn_interface")
except Exception as e:
    print(f"✗ Failed to import flash_attn_interface: {e}")

# Test 3: Check if the FlagGems backend exists
try:
    from flash_attn import flash_attn_flaggems_backend
    print("✓ Successfully imported flash_attn_flaggems_backend")
except Exception as e:
    print(f"✗ Failed to import flash_attn_flaggems_backend: {e}")

# Test 4: Check environment variable
use_flaggems = os.getenv("FLASH_ATTENTION_USE_FLAGGEMS", "TRUE")
print(f"\nEnvironment: FLASH_ATTENTION_USE_FLAGGEMS = {use_flaggems}")

# Test 5: List available functions
try:
    from flash_attn import (
        flash_attn_func,
        flash_attn_kvpacked_func,
        flash_attn_qkvpacked_func,
        flash_attn_varlen_func,
        flash_attn_varlen_kvpacked_func,
        flash_attn_varlen_qkvpacked_func,
        flash_attn_with_kvcache,
    )
    print("\n✓ All main functions are available:")
    print("  - flash_attn_func")
    print("  - flash_attn_kvpacked_func")
    print("  - flash_attn_qkvpacked_func")
    print("  - flash_attn_varlen_func")
    print("  - flash_attn_varlen_kvpacked_func")
    print("  - flash_attn_varlen_qkvpacked_func")
    print("  - flash_attn_with_kvcache")
except Exception as e:
    print(f"\n✗ Failed to import functions: {e}")

# Test 6: Check file structure
print("\n" + "=" * 60)
print("File Structure Check:")
print("=" * 60)

import os
def check_file(path, description):
    if os.path.exists(path):
        print(f"✓ {description}: {path}")
    else:
        print(f"✗ {description} missing: {path}")

base_dir = os.path.dirname(__file__)
check_file(os.path.join(base_dir, "flash_attn/__init__.py"), "Main __init__.py")
check_file(os.path.join(base_dir, "flash_attn/flash_attn_interface.py"), "Interface module")
check_file(os.path.join(base_dir, "flash_attn/flash_attn_flaggems_backend.py"), "FlagGems backend")
check_file(os.path.join(base_dir, "setup.py"), "Setup script")
check_file(os.path.join(base_dir, "README_FLAGGEMS.md"), "FlagGems README")

print("\n" + "=" * 60)
print("Structure test complete!")
print("=" * 60)