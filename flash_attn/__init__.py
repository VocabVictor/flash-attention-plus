__version__ = "2.8.0.post2+flaggems"

from flash_attn.flash_attn_interface import (
    flash_attn_func,
    flash_attn_kvpacked_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_with_kvcache,
)

# Check if FlagGems backend is being used
import os
if os.getenv("FLASH_ATTENTION_USE_FLAGGEMS", "TRUE") == "TRUE":
    print("FlashAttention: Using FlagGems Triton backend")
