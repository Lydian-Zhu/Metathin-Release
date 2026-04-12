"""
PatternSpace Interface (P) - Perception Layer | 模式空间接口 (P) - 感知层
========================================================================

Defines the PatternSpace interface - the agent's "eyes" for perceiving the environment.
Converts raw input into structured feature vectors.

定义模式空间接口 - 代理感知环境的"眼睛"。
将原始输入转换为结构化的特征向量。

Role in Quintuple | 五元组中的角色:
    P (PatternSpace): Perception - Converts raw input to feature vectors
    P (PatternSpace): 感知 - 将原始输入转换为特征向量

Design Philosophy | 设计理念:
    - Generic: Supports any input type T | 泛型：支持任意输入类型 T
    - Structured: Outputs consistent FeatureVector | 结构化：输出一致的特征向量
    - Observable: Provides feature names for debugging | 可观测：提供特征名称用于调试
"""

from abc import ABC, abstractmethod
from typing import List, Generic
import numpy as np
import logging

from .types import T, FeatureVector
from .exceptions import PatternExtractionError


# ============================================================
# PatternSpace Interface | 模式空间接口
# ============================================================

class PatternSpace(ABC, Generic[T]):
    """
    Pattern Space P: Window to perceive the environment.
    
    模式空间 P：感知环境的窗口。
    
    This serves as the agent's "eyes," responsible for converting raw input into
    structured feature vectors. Feature vectors should capture key information
    from the input, providing a foundation for subsequent decision-making.
    
    这是代理的"眼睛"，负责将原始输入转换为结构化的特征向量。
    特征向量应捕获输入的关键信息，为后续决策提供基础。
    
    Implementation Requirements | 实现要求:
        - extract() MUST be implemented | 必须实现 extract() 方法
        - Feature vectors must be 1D float64 numpy arrays | 特征向量必须是 1D float64 numpy 数组
        - Different inputs should return same dimension vectors | 不同输入应返回相同维度的向量
        - Features should be normalized to reasonable range | 特征应归一化到合理范围
    
    Type Parameters | 类型参数:
        T: Raw input type (can be any Python type) | 原始输入类型
    
    Example | 示例:
        >>> class TextPattern(PatternSpace[str]):
        ...     '''Extract text length and word count | 提取文本长度和词数'''
        ...     def extract(self, text: str) -> FeatureVector:
        ...         char_count = len(text)
        ...         word_count = len(text.split())
        ...         return np.array([char_count, word_count], dtype=np.float64)
        ...     
        ...     def get_feature_names(self) -> List[str]:
        ...         return ['char_count', 'word_count']
        >>> 
        >>> pattern = TextPattern()
        >>> features = pattern.extract("hello world")
        >>> print(features)  # [11, 2]
    """
    
    @abstractmethod
    def extract(self, raw_input: T) -> FeatureVector:
        """
        Extract feature vector from raw input.
        
        从原始输入提取特征向量。
        
        This is the core method of the pattern space and must be implemented by subclasses.
        Feature extraction can be as simple as statistical calculations or as complex
        as deep learning models.
        
        这是模式空间的核心方法，必须由子类实现。
        特征提取可以像统计计算一样简单，也可以像深度学习模型一样复杂。
        
        Args | 参数:
            raw_input: Raw input data, type determined by generic parameter T
                      原始输入数据，类型由泛型参数 T 决定
            
        Returns | 返回:
            FeatureVector: Feature vector as numpy float64 array
                           numpy float64 数组形式的特征向量
            
        Raises | 抛出:
            PatternExtractionError: When feature extraction fails
                                   当特征提取失败时
            
        Notes | 注意:
            - Returned array must be 1-dimensional | 返回的数组必须是一维的
            - Different inputs should return same dimension | 不同输入应返回相同维度
            - Feature values should be normalized | 特征值应归一化
            - Avoid NaN or Inf values | 避免 NaN 或 Inf 值
        """
        pass
    
        try:
            # 实际实现在子类中
            pass
        except Exception as e:
            if not isinstance(e, PatternExtractionError):
                raise PatternExtractionError(f"Feature extraction failed: {e}") from e
            raise    
    def get_feature_names(self) -> List[str]:
        """
        Get feature name list (optional implementation).
        
        获取特征名称列表（可选实现）。
        
        Used for debugging, logging, and visualization. Feature names should
        correspond one-to-one with feature vector dimensions.
        
        用于调试、日志记录和可视化。特征名称应与特征向量维度一一对应。
        
        Returns | 返回:
            List[str]: Feature names list, empty list if not implemented
                      特征名称列表，未实现时返回空列表
        """
        return []
    
    def get_feature_dimension(self) -> int:
        """
        Get feature dimension (optional implementation).
        
        获取特征维度（可选实现）。
        
        Default implementation infers dimension by extracting features from a sample.
        If extraction is expensive, subclasses should override this method.
        
        默认实现通过从样本提取特征来推断维度。
        如果提取成本高，子类应覆盖此方法。
        
        Returns | 返回:
            int: Feature vector dimension, 0 if cannot determine
                特征向量维度，无法确定时返回 0
        """
        try:
            # Use a simple test sample | 使用简单测试样本
            sample = self.extract(None)
            return len(sample)
        except (TypeError, AttributeError, PatternExtractionError):
            return 0
    
    def validate_features(self, features: FeatureVector) -> bool:
        """
        Validate feature vector (optional implementation).
        
        验证特征向量（可选实现）。
        
        Checks whether the feature vector meets requirements.
        
        检查特征向量是否满足要求。
        
        Validation checks | 验证检查项:
            - Is numpy array | 是 numpy 数组
            - Has float64 dtype | 类型为 float64
            - Is one-dimensional | 是一维的
            - Contains no NaN values | 不包含 NaN 值
            - Contains no Inf values | 不包含 Inf 值
        
        Args | 参数:
            features: Feature vector to validate | 要验证的特征向量
            
        Returns | 返回:
            bool: True if valid, False otherwise | 有效返回 True，否则返回 False
        """
        return (isinstance(features, np.ndarray) and 
                features.dtype == np.float64 and
                features.ndim == 1 and
                not np.any(np.isnan(features)) and
                not np.any(np.isinf(features)))


# ============================================================
# Export Interface | 导出接口
# ============================================================

__all__ = [
    'PatternSpace',           # Pattern space interface | 模式空间接口
    'PatternExtractionError', # Feature extraction exception | 特征提取异常
]