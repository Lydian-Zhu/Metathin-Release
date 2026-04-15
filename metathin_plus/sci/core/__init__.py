# metathin_sci/core/__init__.py
"""
Metathin+Sci Core Module | 核心模块
====================================

Provides foundational components for scientific discovery:
- Function generator for synthetic data generation
- Feature extractor for time series characterization
- Similarity matcher for function retrieval

提供科学发现的基础组件：
- 用于合成数据生成的函数生成器
- 用于时间序列特征化的特征提取器
- 用于函数检索的相似度匹配器
"""

from .function_generator import (
    FunctionGenerator,
    FunctionType,
    FunctionTemplate,
    FunctionSample,
)

from .feature_extractor import (
    FeatureExtractor,
    FeatureType,
    FeatureDefinition,
)

from .similarity_matcher import (
    SimilarityMatcher,
    MatchResult,
    DistanceMetric,
)

__all__ = [
    # Function Generator | 函数生成器
    'FunctionGenerator',
    'FunctionType',
    'FunctionTemplate',
    'FunctionSample',
    
    # Feature Extractor | 特征提取器
    'FeatureExtractor',
    'FeatureType',
    'FeatureDefinition',
    
    # Similarity Matcher | 相似度匹配器
    'SimilarityMatcher',
    'MatchResult',
    'DistanceMetric',
]

import logging
logger = logging.getLogger(__name__)
logger.debug("Metathin+Sci core module initialized | 核心模块初始化完成")