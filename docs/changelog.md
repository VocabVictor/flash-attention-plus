# Changelog - FlashAttention-Plus

All notable changes to FlashAttention-Plus will be documented in this file.

## [Unreleased]

### In Development
- Backward pass implementation using FlagGems/Triton
- KV cache support for efficient inference
- Variable-length sequence support
- Extended hardware platform testing

## [0.1.0] - Initial Release

### Added
- **FlagGems Backend Integration**
  - Replaced CUDA kernels with FlagGems' Triton implementation
  - Added `flash_attn_flaggems_backend.py` adapter module
  - Environment variable `FLASH_ATTENTION_USE_FLAGGEMS` for backend selection

- **Core Functionality**
  - Forward pass for all main attention functions
  - Support for `flash_attn_func`, `flash_attn_qkvpacked_func`, `flash_attn_kvpacked_func`
  - Causal masking support
  - Multi-head and multi-query attention (MHA/MQA/GQA)
  - FP16 and BF16 precision support

- **API Compatibility**
  - Maintained full API compatibility with original FlashAttention
  - Drop-in replacement capability
  - Preserved all function signatures and return types

- **Documentation**
  - Comprehensive README with installation and usage instructions
  - Migration guide from original FlashAttention
  - Technical documentation for FlagGems integration
  - API reference documentation
  - Code examples and best practices

### Changed
- **Build System**
  - Removed CUDA compilation requirements
  - Simplified installation process
  - Updated dependencies to include Triton and FlagGems

- **Backend Architecture**
  - Abstracted backend selection mechanism
  - Added runtime backend switching capability
  - Improved error handling for missing dependencies

### Removed
- All CUDA C++ source files
- CUDA-specific build scripts and configurations
- NVCC compiler dependencies

### Known Issues
- Backward pass not yet implemented
- Dropout interface exists but may not be fully functional
- Some advanced features (block-sparse, etc.) not yet supported
- Performance may vary compared to hand-tuned CUDA kernels

## Version History

### Versioning Scheme
This project follows [Semantic Versioning](https://semver.org/):
- MAJOR version for incompatible API changes
- MINOR version for backwards-compatible functionality additions
- PATCH version for backwards-compatible bug fixes

### Comparison with Original FlashAttention
FlashAttention-Plus maintains API compatibility with FlashAttention v2.x while providing:
- Hardware-agnostic implementation via Triton
- Easier installation without CUDA compilation
- Broader platform support potential

## Future Roadmap

### v0.2.0 (Planned)
- [ ] Complete backward pass implementation
- [ ] Add gradient checkpointing support
- [ ] Performance optimizations for common configurations
- [ ] Extended test coverage

### v0.3.0 (Planned)
- [ ] KV cache implementation for inference
- [ ] Variable-length sequence support
- [ ] Sliding window attention optimizations
- [ ] AMD GPU performance tuning

### v1.0.0 (Planned)
- [ ] Feature parity with original FlashAttention
- [ ] Production-ready stability
- [ ] Comprehensive benchmarks across platforms
- [ ] Advanced features (block-sparse, etc.)

## Contributing

We welcome contributions! Key areas for contribution:
- Backward pass implementation
- Performance optimizations
- Extended hardware testing
- Documentation improvements

Please see our contribution guidelines in the main repository.