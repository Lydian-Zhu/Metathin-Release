"""
Metathin Configuration System
==============================

Metathin 配置系统

This package provides a flexible configuration system for the Metathin framework.
Configuration can be loaded from defaults, files, environment variables, or dictionaries.

本包为 Metathin 框架提供灵活的配置系统。
配置可以从默认值、文件、环境变量或字典加载。

Modules | 模块:
    - schema.py: Configuration data structures | 配置数据结构
    - loader.py: Configuration loading utilities | 配置加载工具
"""

from .schema import (
    PipelineConfig,
    MemoryConfig,
    ObservabilityConfig,
    MetathinConfig,
)

from .loader import (
    ConfigLoader,
    load_config,
    save_config,
)


__all__ = [
    # Schema | 模式
    'PipelineConfig',
    'MemoryConfig',
    'ObservabilityConfig',
    'MetathinConfig',
    
    # Loader | 加载器
    'ConfigLoader',
    'load_config',
    'save_config',
]