# 更新日志 - FlashAttention-Plus

FlashAttention-Plus 的所有重要更改都将记录在此文件中。

## [未发布]

### 开发中
- 使用 FlagGems/Triton 的反向传播实现
- 高效推理的 KV 缓存支持
- 可变长度序列支持
- 扩展硬件平台测试

## [0.1.0] - 初始版本

### 新增
- **FlagGems 后端集成**
  - 用 FlagGems 的 Triton 实现替换了 CUDA 内核
  - 添加了 `flash_attn_flaggems_backend.py` 适配器模块
  - 环境变量 `FLASH_ATTENTION_USE_FLAGGEMS` 用于后端选择

- **核心功能**
  - 所有主要注意力函数的前向传播
  - 支持 `flash_attn_func`、`flash_attn_qkvpacked_func`、`flash_attn_kvpacked_func`
  - 因果掩码支持
  - 多头和多查询注意力（MHA/MQA/GQA）
  - FP16 和 BF16 精度支持

- **API 兼容性**
  - 保持与原始 FlashAttention 的完全 API 兼容性
  - 即插即用替换能力
  - 保留所有函数签名和返回类型

- **文档**
  - 包含安装和使用说明的综合 README
  - 从原始 FlashAttention 的迁移指南
  - FlagGems 集成的技术文档
  - API 参考文档
  - 代码示例和最佳实践

### 更改
- **构建系统**
  - 移除了 CUDA 编译要求
  - 简化了安装过程
  - 更新依赖项以包含 Triton 和 FlagGems

- **后端架构**
  - 抽象的后端选择机制
  - 添加了运行时后端切换功能
  - 改进了缺少依赖项的错误处理

### 移除
- 所有 CUDA C++ 源文件
- CUDA 特定的构建脚本和配置
- NVCC 编译器依赖项

### 已知问题
- 反向传播尚未实现
- Dropout 接口存在但可能无法完全正常工作
- 某些高级功能（块稀疏等）尚未支持
- 与手动调优的 CUDA 内核相比，性能可能有所不同

## 版本历史

### 版本控制方案
该项目遵循[语义版本控制](https://semver.org/)：
- MAJOR 版本用于不兼容的 API 更改
- MINOR 版本用于向后兼容的功能添加
- PATCH 版本用于向后兼容的错误修复

### 与原始 FlashAttention 的比较
FlashAttention-Plus 保持与 FlashAttention v2.x 的 API 兼容性，同时提供：
- 通过 Triton 的硬件无关实现
- 无需 CUDA 编译的更简单安装
- 更广泛的平台支持潜力

## 未来路线图

### v0.2.0（计划中）
- [ ] 完成反向传播实现
- [ ] 添加梯度检查点支持
- [ ] 常见配置的性能优化
- [ ] 扩展测试覆盖范围

### v0.3.0（计划中）
- [ ] 推理的 KV 缓存实现
- [ ] 可变长度序列支持
- [ ] 滑动窗口注意力优化
- [ ] AMD GPU 性能调优

### v1.0.0（计划中）
- [ ] 与原始 FlashAttention 的功能平等
- [ ] 生产就绪的稳定性
- [ ] 跨平台的综合基准测试
- [ ] 高级功能（块稀疏等）

## 贡献

我们欢迎贡献！关键贡献领域：
- 反向传播实现
- 性能优化
- 扩展硬件测试
- 文档改进

请参阅主存储库中的贡献指南。