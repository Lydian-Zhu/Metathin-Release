"""
Scientific Discovery Pattern Space | 科学发现模式空间
=====================================================

Pattern space adapter that converts time series data to function vectors.
将时间序列数据转换为函数向量的模式空间适配器。

This adapter serves as the perception layer (P) in Metathin,
transforming raw data into feature vectors for the agent.
此适配器作为 Metathin 中的感知层 (P)，
将原始数据转换为代理使用的特征向量。
"""

import numpy as np
from typing import Any, List, Optional, Union
import logging

from metathin.core.p_pattern import PatternSpace
from metathin.core.types import FeatureVector

from ..core.function_space import FunctionSpace, VectorSpaceConfig
from ..core.laurent_expander import LaurentExpander


class SciPatternSpace(PatternSpace):
    """
    Scientific Discovery Pattern Space | 科学发现模式空间
    
    Converts time series data to Laurent coefficient vectors.
    将时间序列数据转换为洛朗系数向量。
    
    This pattern space extracts function vectors from sequential data,
    enabling the agent to perceive the underlying mathematical structure.
    此模式空间从序列数据中提取函数向量，
    使代理能够感知底层的数学结构。
    
    Features | 特性:
        - Streaming data support | 流式数据支持
        - Configurable window size | 可配置窗口大小
        - Automatic vector extraction | 自动向量提取
        - Buffer management | 缓冲区管理
    
    Example | 示例:
        >>> pattern = SciPatternSpace(window_size=50)
        >>> 
        >>> # Process streaming data | 处理流式数据
        >>> for value in data_stream:
        ...     features = pattern.extract(value)
        ...     # features is a 41-dim vector | 41维向量
    """
    
    def __init__(self,
                 window_size: int = 50,
                 n_negative: int = 20,
                 n_positive: int = 20,
                 center: float = 0.0,
                 name: str = "SciPattern",
                 use_cache: bool = True):
        """
        Initialize scientific discovery pattern space | 初始化科学发现模式空间
        
        Args:
            window_size: Window size for vector extraction | 向量提取窗口大小
            n_negative: Number of negative power terms | 负幂项数量
            n_positive: Number of positive power terms | 正幂项数量
            center: Expansion center point | 展开中心点
            name: Pattern space name | 模式空间名称
            use_cache: Whether to cache results | 是否缓存结果
        """
        super().__init__()
        
        self._name = name
        self.window_size = window_size
        self.use_cache = use_cache
        
        # Initialize function space | 初始化函数空间
        config = VectorSpaceConfig(
            n_negative=n_negative,
            n_positive=n_positive,
            center=center
        )
        self._space = FunctionSpace(config)
        self._expander = LaurentExpander(n_negative, n_positive, center, use_cache)
        
        # Data buffer | 数据缓冲区
        self._buffer: List[float] = []
        self._x_counter = 0
        
        # Cache for extracted vectors | 提取向量的缓存
        self._vector_cache: dict = {}
        
        self._logger = logging.getLogger(f"metathin_sci.pattern.{name}")
        self._logger.info(f"Initialized: window={window_size}, dim={config.dimension}")
    
    @property
    def buffer_size(self) -> int:
        """Current buffer size | 当前缓冲区大小"""
        return len(self._buffer)
    
    @property
    def is_ready(self) -> bool:
        """Whether enough data has been collected | 是否已收集足够数据"""
        return len(self._buffer) >= self.window_size
    
    def extract(self, raw_input: Any) -> FeatureVector:
        """
        Extract feature vector from input | 从输入提取特征向量
        
        Supports | 支持:
            - Single value: Adds to buffer, returns vector when buffer is full
              单个值：添加到缓冲区，缓冲区满时返回向量
            - List/array: Processes as complete sequence, returns vector directly
              列表/数组：作为完整序列处理，直接返回向量
            - None: Returns zero vector | 返回零向量
        
        Args:
            raw_input: Input data (number, list, or array) | 输入数据
            
        Returns:
            FeatureVector: 41-dim Laurent coefficient vector | 41维洛朗系数向量
        """
        # Handle None input | 处理 None 输入
        if raw_input is None:
            return np.zeros(self._space.config.dimension, dtype=np.float64)
        
        # Handle batch input (list/array) | 处理批量输入
        if isinstance(raw_input, (list, tuple, np.ndarray)):
            return self._extract_batch(raw_input)
        
        # Handle single value | 处理单个值
        try:
            value = float(raw_input)
            return self._extract_single(value)
        except (ValueError, TypeError):
            self._logger.warning(f"Cannot convert to float: {raw_input}")
            return np.zeros(self._space.config.dimension, dtype=np.float64)
    
    def _extract_single(self, value: float) -> FeatureVector:
        """
        Extract from single value (streaming mode) | 从单个值提取（流式模式）
        
        Args:
            value: Numeric value | 数值
            
        Returns:
            FeatureVector: Feature vector (zero if buffer not ready) | 特征向量
        """
        self._buffer.append(value)
        self._x_counter += 1
        
        # Limit buffer size | 限制缓冲区大小
        max_buffer = self.window_size * 2
        if len(self._buffer) > max_buffer:
            self._buffer = self._buffer[-max_buffer:]
        
        # Not enough data | 数据不足
        if len(self._buffer) < self.window_size:
            return np.zeros(self._space.config.dimension, dtype=np.float64)
        
        # Extract vector from recent window | 从最近窗口提取向量
        x_vals = np.arange(self._x_counter - self.window_size, self._x_counter)
        y_vals = np.array(self._buffer[-self.window_size:])
        
        # Check cache | 检查缓存
        cache_key = tuple(y_vals.tolist())
        if self.use_cache and cache_key in self._vector_cache:
            return self._vector_cache[cache_key]
        
        # Extract vector | 提取向量
        vector = self._expander.expand_data(x_vals, y_vals)
        features = vector.coefficients
        
        # Cache result | 缓存结果
        if self.use_cache:
            self._vector_cache[cache_key] = features
            # Limit cache size | 限制缓存大小
            if len(self._vector_cache) > 100:
                oldest = next(iter(self._vector_cache))
                del self._vector_cache[oldest]
        
        return features
    
    def _extract_batch(self, data: Union[list, tuple, np.ndarray]) -> FeatureVector:
        """
        Extract from batch data | 从批量数据提取
        
        Args:
            data: Complete time series | 完整时间序列
            
        Returns:
            FeatureVector: Feature vector | 特征向量
        """
        arr = np.array(data, dtype=np.float64).flatten()
        
        if len(arr) == 0:
            return np.zeros(self._space.config.dimension, dtype=np.float64)
        
        # Use indices as x coordinates | 使用索引作为 X 坐标
        x_vals = np.arange(len(arr))
        y_vals = arr
        
        # Extract vector | 提取向量
        vector = self._expander.expand_data(x_vals, y_vals)
        return vector.coefficients
    
    def extract_from_xy(self, x_vals: np.ndarray, y_vals: np.ndarray) -> FeatureVector:
        """
        Extract vector from paired (x, y) data | 从配对的 (x, y) 数据提取向量
        
        Args:
            x_vals: X coordinates | X 坐标数组
            y_vals: Y values | Y 值数组
            
        Returns:
            FeatureVector: Feature vector | 特征向量
        """
        if len(x_vals) != len(y_vals):
            raise ValueError("x_vals and y_vals must have same length")
        
        if len(x_vals) == 0:
            return np.zeros(self._space.config.dimension, dtype=np.float64)
        
        vector = self._expander.expand_data(x_vals, y_vals)
        return vector.coefficients
    
    def reset(self) -> None:
        """Reset buffer and cache | 重置缓冲区和缓存"""
        self._buffer = []
        self._x_counter = 0
        self._vector_cache.clear()
        self._logger.info("Pattern space reset")
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names | 获取特征名称
        
        Returns:
            List[str]: Names like 'coeff_-20', ..., 'coeff_0', ..., 'coeff_20'
                      名称如 'coeff_-20', ..., 'coeff_0', ..., 'coeff_20'
        """
        dim = self._space.config.dimension
        n_neg = self._space.config.n_negative
        
        names = []
        for i in range(dim):
            power = i - n_neg
            names.append(f"{self._name}_coeff_{power}")
        return names
    
    def get_feature_dimension(self) -> int:
        """Get feature vector dimension | 获取特征向量维度"""
        return self._space.config.dimension
    
    def get_stats(self) -> dict:
        """Get pattern space statistics | 获取模式空间统计"""
        return {
            'name': self._name,
            'window_size': self.window_size,
            'buffer_size': len(self._buffer),
            'is_ready': self.is_ready,
            'feature_dimension': self.get_feature_dimension(),
            'cache_size': len(self._vector_cache),
            'use_cache': self.use_cache
        }
    
    def __repr__(self) -> str:
        return f"SciPatternSpace(window={self.window_size}, buffer={len(self._buffer)}, dim={self.get_feature_dimension()})"