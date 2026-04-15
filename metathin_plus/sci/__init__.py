# metathin_sci/__init__.py
"""
Metathin+Sci - 科学发现智能体 | Scientific Discovery Agent
============================================================

基于 Metathin 五元组架构的科学发现模块，用于从数据中发现符号规律。

版本: 0.5.0 (重构版)
"""

__version__ = '0.5.0'
__author__ = 'Lydian-Zhu'
__license__ = 'MIT'

# ============================================================
# 核心模块导入 | Core Module Imports
# ============================================================

from .core import (
    FunctionGenerator,
    FeatureExtractor,
    SimilarityMatcher,
    FunctionType,
    FunctionTemplate,
    FunctionSample,
    FeatureDefinition,
    MatchResult,
    DistanceMetric,
)

# ============================================================
# 记忆模块导入 | Memory Module Imports
# ============================================================

from .memory import (
    FunctionMemory,
    FunctionMemoryBank,
)

# ============================================================
# 发现模块导入 | Discovery Module Imports
# ============================================================

from .discovery import (
    AdaptiveExtrapolator,
    SymbolicForm,
    ScientificMetathin,
    DiscoveryReport,
    DiscoveryPhase,
)

# ============================================================
# 导出接口 | Export Interface
# ============================================================

__all__ = [
    # Version | 版本
    '__version__',
    '__author__',
    '__license__',
    
    # Core | 核心模块
    'FunctionGenerator',
    'FeatureExtractor',
    'SimilarityMatcher',
    'FunctionType',
    'FunctionTemplate',
    'FunctionSample',
    'FeatureDefinition',
    'MatchResult',
    'DistanceMetric',
    
    # Memory | 记忆模块
    'FunctionMemory',
    'FunctionMemoryBank',
    
    # Discovery | 发现模块
    'AdaptiveExtrapolator',
    'SymbolicForm',
    'ScientificMetathin',
    'DiscoveryReport',
    'DiscoveryPhase',
]

# ============================================================
# 初始化信息 | Initialization Message
# ============================================================

print(f"✅ Metathin+Sci v{__version__} loaded | 已加载")
print(f"   Scientific discovery agent ready | 科学发现智能体就绪")