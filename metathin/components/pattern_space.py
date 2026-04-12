"""
Pattern Space Components | 模式空间组件
=========================================

Provides various feature extractors that convert raw input into feature vectors.
The pattern space serves as the agent's "eyes," responsible for perceiving and 
understanding the external world.

提供各种特征提取器，将原始输入转换为特征向量。
模式空间作为代理的"眼睛"，负责感知和理解外部世界。

Component Types | 组件类型:
    - SimplePatternSpace: Function-based feature extraction | 基于函数的特征提取
    - StatisticalPatternSpace: Statistical feature extraction | 统计特征提取
    - NormalizedPatternSpace: Feature normalization | 特征归一化
    - CompositePatternSpace: Multi-source feature combination | 多源特征组合
    - CachedPatternSpace: Cached feature extraction | 缓存特征提取

Design Philosophy | 设计理念:
    - Flexibility: Supports arbitrary input types | 灵活性：支持任意输入类型
    - Composability: Multiple pattern spaces can be combined | 可组合性：多个模式空间可组合
    - Observable: Provides feature names and dimension info | 可观测性：提供特征名称和维度信息
    - Robustness: Handles exceptional inputs and edge cases | 健壮性：处理异常输入和边缘情况
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
from abc import ABC, abstractmethod
import logging
import warnings

# ============================================================
# Fixed Imports for Refactored Core | 重构后核心的修复导入
# ============================================================
from ..core.p_pattern import PatternSpace
from ..core.types import FeatureVector
from ..core.exceptions import PatternExtractionError


# ============================================================
# Helper Functions | 辅助函数
# ============================================================

def safe_float_convert(value: Any) -> float:
    """
    Safely convert any value to float.
    
    安全地将任意值转换为浮点数。
    
    Handles various input types, ensuring a valid float is returned.
    
    处理各种输入类型，确保返回有效的浮点数。
    
    Args | 参数:
        value: Value to convert | 要转换的值
        
    Returns | 返回:
        float: Converted float value, 0.0 if conversion fails
               转换后的浮点值，转换失败时返回 0.0
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def safe_array_convert(data: Any) -> np.ndarray:
    """
    Safely convert any data to numpy array.
    
    安全地将任意数据转换为 numpy 数组。
    
    Handles different data types, ensuring a float64 numpy array is returned.
    
    处理不同的数据类型，确保返回 float64 类型的 numpy 数组。
    
    Args | 参数:
        data: Data to convert | 要转换的数据
        
    Returns | 返回:
        np.ndarray: Converted numpy array | 转换后的 numpy 数组
    """
    if isinstance(data, np.ndarray):
        return data.astype(np.float64)
    elif isinstance(data, (list, tuple)):
        return np.array([safe_float_convert(x) for x in data], dtype=np.float64)
    elif isinstance(data, (int, float)):
        return np.array([float(data)], dtype=np.float64)
    elif isinstance(data, str):
        # Special handling for strings: return string length | 字符串的特殊处理：返回字符串长度
        return np.array([float(len(data))], dtype=np.float64)
    else:
        return np.array([0.0], dtype=np.float64)


# ============================================================
# 1. Simple Pattern Space | 简单模式空间
# ============================================================

class SimplePatternSpace(PatternSpace):
    """
    Simple Pattern Space: Function-based feature extraction.
    
    简单模式空间：基于函数的特征提取。
    
    Uses user-provided functions to extract features from input, offering maximum flexibility.
    Users can define arbitrary feature extraction logic, returning a list of feature values.
    
    使用用户提供的函数从输入中提取特征，提供最大的灵活性。
    用户可以定义任意的特征提取逻辑，返回特征值列表。
    
    Characteristics | 特性:
        - Flexible: Users can define any feature extraction function | 灵活：用户可以定义任意特征提取函数
        - Simple: Only need to implement a single function | 简单：只需实现一个函数
        - Describable: Can specify feature names | 可描述：可以指定特征名称
    
    Example | 示例:
        >>> # Extract text length and word count | 提取文本长度和词数
        >>> pattern = SimplePatternSpace(
        ...     lambda text: [len(text), len(text.split())],
        ...     feature_names=['char_count', 'word_count'],
        ...     name="text_features"
        ... )
        >>> 
        >>> features = pattern.extract("hello world")
        >>> print(features)  # [11, 2]
        >>> print(pattern.get_feature_names())  # ['char_count', 'word_count']
    """
    
    def __init__(self, 
                 extract_func: Callable[[Any], List[float]],
                 feature_names: Optional[List[str]] = None,
                 name: str = "SimplePattern"):
        """
        Initialize simple pattern space.
        
        初始化简单模式空间。
        
        Args | 参数:
            extract_func: Feature extraction function, receives raw input and returns list of feature values
                         特征提取函数，接收原始输入并返回特征值列表
            feature_names: Feature names list, length must match extract_func's return value
                           特征名称列表，长度必须与 extract_func 返回值匹配
            name: Pattern space name for logging and debugging | 模式空间名称，用于日志和调试
            
        Raises | 抛出:
            TypeError: If extract_func is not callable | 如果 extract_func 不可调用
        """
        super().__init__()
        
        if not callable(extract_func):
            raise TypeError(f"extract_func must be callable, got {type(extract_func)}")
        
        self._extract_func = extract_func
        self._name = name
        self._feature_names = feature_names or []
        self._cached_dim = None
        self._logger = logging.getLogger(f"metathin.pattern.{name}")
    
    def extract(self, raw_input: Any) -> FeatureVector:
        """
        Execute feature extraction.
        
        执行特征提取。
        
        Calls the user-provided function and performs necessary type conversion and validation.
        
        调用用户提供的函数并执行必要的类型转换和验证。
        
        Args | 参数:
            raw_input: Raw input data | 原始输入数据
            
        Returns | 返回:
            FeatureVector: Extracted feature vector | 提取的特征向量
            
        Raises | 抛出:
            PatternExtractionError: If feature extraction fails | 如果特征提取失败
        """
        try:
            self._logger.debug(f"Extracting features: input type={type(raw_input).__name__}")
            
            # Call user function | 调用用户函数
            features = self._extract_func(raw_input)
            
            # Ensure it's a list | 确保是列表
            if not isinstance(features, (list, tuple)):
                features = [features]
            
            # Convert to floats | 转换为浮点数
            features = [safe_float_convert(f) for f in features]
            
            # Convert to numpy array | 转换为 numpy 数组
            result = np.array(features, dtype=np.float64)
            
            # Validate | 验证
            if not self.validate_features(result):
                raise ValueError("Invalid feature vector produced")
            
            # Update cached dimension | 更新缓存的维度
            if self._cached_dim is None:
                self._cached_dim = len(result)
                self._logger.debug(f"Feature dimension: {self._cached_dim}")
            
            return result
            
        except Exception as e:
            self._logger.error(f"Feature extraction failed: {e}")
            raise PatternExtractionError(f"Feature extraction failed: {e}") from e
    
    def get_feature_names(self) -> List[str]:
        """Get feature names | 获取特征名称"""
        if self._feature_names:
            return self._feature_names
        
        # Generate default names | 生成默认名称
        # 尝试从缓存维度获取，如果没有则从样本获取
        if self._cached_dim is not None:
            return [f"{self._name}_{i}" for i in range(self._cached_dim)]
        
        # 尝试通过提取样本获取维度
        try:
            sample_features = self.extract("sample")
            dim = len(sample_features)
            self._cached_dim = dim
            return [f"{self._name}_{i}" for i in range(dim)]
        except:
            return []


# ============================================================
# 2. Statistical Pattern Space | 统计模式空间
# ============================================================

class StatisticalPatternSpace(PatternSpace):
    """
    Statistical Pattern Space: Extracts statistical features from numerical sequences.
    
    统计模式空间：从数值序列中提取统计特征。
    
    Automatically computes various statistics from input data, such as mean, standard deviation, max, etc.
    Particularly suitable for time series data or numerical arrays.
    
    自动从输入数据中计算各种统计量，如均值、标准差、最大值等。
    特别适用于时间序列数据或数值数组。
    
    Available statistical features | 可用的统计特征:
        - 'mean': Arithmetic mean | 算术均值
        - 'std': Standard deviation | 标准差
        - 'var': Variance | 方差
        - 'max': Maximum value | 最大值
        - 'min': Minimum value | 最小值
        - 'median': Median value | 中位数
        - 'sum': Sum of all values | 总和
        - 'count': Number of values | 数值个数
        - 'skew': Skewness | 偏度
        - 'kurtosis': Kurtosis | 峰度
        - 'q1': First quartile (25th percentile) | 第一四分位数
        - 'q3': Third quartile (75th percentile) | 第三四分位数
        - 'range': Range (max - min) | 极差
        - 'iqr': Interquartile range (q3 - q1) | 四分位距
    """
    
    # Available statistical features and their computation functions
    # 可用的统计特征及其计算函数
    _FEATURE_FUNCS = {
        'mean': np.mean,
        'std': np.std,
        'var': np.var,
        'max': np.max,
        'min': np.min,
        'median': np.median,
        'sum': np.sum,
        'count': len,
        'skew': lambda x: float(np.mean((x - np.mean(x)) ** 3) / (np.std(x) ** 3 + 1e-8)),
        'kurtosis': lambda x: float(np.mean((x - np.mean(x)) ** 4) / (np.var(x) ** 2 + 1e-8) - 3),
        'q1': lambda x: float(np.percentile(x, 25)),
        'q3': lambda x: float(np.percentile(x, 75)),
        'range': lambda x: float(np.max(x) - np.min(x)),
        'iqr': lambda x: float(np.percentile(x, 75) - np.percentile(x, 25)),
    }
    
    def __init__(self, 
                 features: Optional[List[str]] = None,
                 name: str = "StatisticalPattern",
                 handle_empty: str = 'zeros'):
        """
        Initialize statistical pattern space.
        
        初始化统计模式空间。
        
        Args | 参数:
            features: List of statistical features to extract, None means use all features
                     要提取的统计特征列表，None 表示使用所有特征
            name: Pattern space name | 模式空间名称
            handle_empty: How to handle empty input | 如何处理空输入
                - 'zeros': Return zero vector | 返回零向量
                - 'raise': Raise an exception | 抛出异常
                
        Raises | 抛出:
            ValueError: If an unknown feature name is provided | 如果提供了未知的特征名称
        """
        super().__init__()
        
        # Set feature list | 设置特征列表
        if features is None:
            self._features = list(self._FEATURE_FUNCS.keys())
        else:
            # Validate feature names | 验证特征名称
            for f in features:
                if f not in self._FEATURE_FUNCS:
                    raise ValueError(f"Unknown statistical feature: {f}")
            self._features = features
        
        self._name = name
        self._handle_empty = handle_empty
        self._logger = logging.getLogger(f"metathin.pattern.{name}")
    
    def _prepare_data(self, raw_input: Any) -> np.ndarray:
        """
        Prepare data: convert input to numpy array.
        
        准备数据：将输入转换为 numpy 数组。
        
        Args | 参数:
            raw_input: Raw input data | 原始输入数据
            
        Returns | 返回:
            np.ndarray: Prepared data array | 准备好的数据数组
            
        Raises | 抛出:
            PatternExtractionError: If data is empty and handle_empty='raise'
                                    如果数据为空且 handle_empty='raise'
        """
        # Convert to array | 转换为数组
        if isinstance(raw_input, np.ndarray):
            data = raw_input.flatten()
        elif isinstance(raw_input, (list, tuple)):
            data = np.array([safe_float_convert(x) for x in raw_input], dtype=np.float64)
        elif isinstance(raw_input, (int, float)):
            data = np.array([float(raw_input)], dtype=np.float64)
        else:
            data = np.array([0.0], dtype=np.float64)
        
        # Handle empty data | 处理空数据
        if len(data) == 0:
            if self._handle_empty == 'raise':
                raise PatternExtractionError("Input data is empty")
            else:
                self._logger.warning("Input data is empty, returning zero vector")
                return np.zeros(1, dtype=np.float64)
        
        return data
    
    def extract(self, raw_input: Any) -> FeatureVector:
        """
        Extract statistical features.
        
        提取统计特征。
        
        Args | 参数:
            raw_input: Raw input data | 原始输入数据
            
        Returns | 返回:
            FeatureVector: Extracted statistical features | 提取的统计特征
            
        Raises | 抛出:
            PatternExtractionError: If feature extraction fails | 如果特征提取失败
        """
        try:
            self._logger.debug(f"Extracting statistical features: input type={type(raw_input).__name__}")
            
            # Prepare data | 准备数据
            data = self._prepare_data(raw_input)
            
            # Compute each statistical feature | 计算每个统计特征
            features = []
            for f_name in self._features:
                func = self._FEATURE_FUNCS[f_name]
                try:
                    value = func(data)
                    features.append(float(value))
                except Exception as e:
                    self._logger.debug(f"Computing feature '{f_name}' failed: {e}")
                    features.append(0.0)
            
            result = np.array(features, dtype=np.float64)
            self._logger.debug(f"Statistical features: {dict(zip(self._features, result))}")
            
            return result
            
        except Exception as e:
            self._logger.error(f"Statistical feature extraction failed: {e}")
            raise PatternExtractionError(f"Statistical feature extraction failed: {e}") from e
    
    def get_feature_names(self) -> List[str]:
        """Get feature names | 获取特征名称"""
        return [f"{self._name}.{f}" for f in self._features]


# ============================================================
# 3. Normalized Pattern Space | 归一化模式空间
# ============================================================

class NormalizedPatternSpace(PatternSpace):
    """
    Normalized Pattern Space: Normalizes features from another pattern space.
    
    归一化模式空间：对另一个模式空间的特征进行归一化。
    
    Maps feature values to a consistent range (e.g., [0,1]), improving compatibility across
    features with different scales. Supports multiple normalization methods.
    
    将特征值映射到一致的范围（如 [0,1]），提高不同尺度特征之间的兼容性。
    支持多种归一化方法。
    
    Normalization Methods | 归一化方法:
        - 'fixed': Fixed range normalization | 固定范围归一化
        - 'adaptive': Adaptive range normalization (dynamically updated) | 自适应范围归一化（动态更新）
        - 'standard': Standardization (z-score) | 标准化（z-score）
    
    Example | 示例:
        >>> base = SimplePatternSpace(lambda x: [x, x**2])
        >>> norm = NormalizedPatternSpace(base, method='standard')
        >>> features = norm.extract(5)
    """
    
    def __init__(self, 
                 base_pattern: PatternSpace,
                 ranges: Optional[List[Tuple[float, float]]] = None,
                 method: str = 'fixed'):
        """
        Initialize normalized pattern space.
        
        初始化归一化模式空间。
        
        Args | 参数:
            base_pattern: Base pattern space | 基础模式空间
            ranges: List of normalization ranges, each feature gets (min, max), only for 'fixed' method
                    归一化范围列表，每个特征对应 (min, max)，仅用于 'fixed' 方法
            method: Normalization method | 归一化方法
                - 'fixed': Fixed range normalization | 固定范围归一化
                - 'adaptive': Adaptive range normalization | 自适应范围归一化
                - 'standard': Standardization (z-score) | 标准化
                
        Raises | 抛出:
            TypeError: If base_pattern is not a PatternSpace instance | 如果 base_pattern 不是 PatternSpace 实例
            ValueError: If ranges length doesn't match base feature dimension | 如果 ranges 长度与基础特征维度不匹配
        """
        super().__init__()
        
        if not isinstance(base_pattern, PatternSpace):
            raise TypeError(f"base_pattern must be a PatternSpace instance")
        
        self._base = base_pattern
        self._method = method
        self._logger = logging.getLogger("metathin.pattern.NormalizedPatternSpace")
        
        # Get base feature dimension | 获取基础特征维度
        try:
            base_dim = base_pattern.get_feature_dimension()
            if base_dim == 0:
                # 使用一个简单的测试输入
                try:
                    sample_features = base_pattern.extract(0)
                    base_dim = len(sample_features)
                except:
                    sample_features = base_pattern.extract(0.0)
                    base_dim = len(sample_features)
        except Exception as e:
            self._logger.debug(f"Failed to get base dimension: {e}")
            base_dim = 1
        
        # Initialize ranges | 初始化范围
        if method == 'fixed':
            if ranges is None:
                self._ranges = [(0.0, 1.0)] * base_dim
                self._logger.warning(f"method='fixed' but no ranges provided, using default")
            else:
                if len(ranges) != base_dim:
                    raise ValueError(f"Number of ranges ({len(ranges)}) doesn't match base feature dimension ({base_dim})")
                self._ranges = ranges
            
            self._mins = np.array([r[0] for r in self._ranges])
            self._maxs = np.array([r[1] for r in self._ranges])
        else:
            self._ranges = [(0.0, 1.0)] * base_dim
            self._mins = np.zeros(base_dim)
            self._maxs = np.ones(base_dim)
            self._means = np.zeros(base_dim)
            self._stds = np.ones(base_dim)
            self._count = 0
    
    def extract(self, raw_input: Any) -> FeatureVector:
        """
        Extract and normalize features.
        
        提取并归一化特征。
        
        Args | 参数:
            raw_input: Raw input data | 原始输入数据
            
        Returns | 返回:
            FeatureVector: Normalized feature vector | 归一化后的特征向量
            
        Raises | 抛出:
            PatternExtractionError: If normalization fails | 如果归一化失败
        """
        try:
            # Extract base features | 提取基础特征
            features = self._base.extract(raw_input)
            
            # Ensure 1-dimensional | 确保一维
            if features.ndim > 1:
                features = features.flatten()
            
            # Normalize based on method | 根据方法归一化
            if self._method == 'fixed':
                normalized = self._normalize_fixed(features)
            elif self._method == 'adaptive':
                normalized = self._normalize_adaptive(features)
            elif self._method == 'standard':
                normalized = self._normalize_standard(features)
            else:
                normalized = features
            
            # Ensure within [0,1] range | 确保在 [0,1] 范围内
            normalized = np.clip(normalized, 0.0, 1.0)
            
            return normalized
            
        except Exception as e:
            self._logger.error(f"Normalization failed: {e}")
            raise PatternExtractionError(f"Normalization failed: {e}") from e
    
    def _normalize_fixed(self, features: np.ndarray) -> np.ndarray:
        """Fixed range normalization: linear mapping to [0,1] | 固定范围归一化：线性映射到 [0,1]"""
        normalized = np.zeros_like(features)
        for i in range(min(len(features), len(self._mins))):
            min_val, max_val = self._mins[i], self._maxs[i]
            if max_val > min_val:
                normalized[i] = (features[i] - min_val) / (max_val - min_val)
            else:
                normalized[i] = 0.5
        return normalized
    
    def _normalize_adaptive(self, features: np.ndarray) -> np.ndarray:
        """Adaptive range normalization: dynamically update min/max | 自适应范围归一化：动态更新 min/max"""
        # Update ranges | 更新范围
        self._count += 1
        for i in range(min(len(features), len(self._mins))):
            self._mins[i] = min(self._mins[i], features[i])
            self._maxs[i] = max(self._maxs[i], features[i])
        
        # Normalize | 归一化
        normalized = np.zeros_like(features)
        for i in range(min(len(features), len(self._mins))):
            if self._maxs[i] > self._mins[i]:
                normalized[i] = (features[i] - self._mins[i]) / (self._maxs[i] - self._mins[i])
            else:
                normalized[i] = 0.5
        
        return normalized
    
    def _normalize_standard(self, features: np.ndarray) -> np.ndarray:
        """Standardization (z-score): transform to mean 0, std 1 | 标准化：转换为均值 0，标准差 1"""
        self._count += 1
        
        # Update mean and variance | 更新均值和方差
        for i in range(min(len(features), len(self._means))):
            delta = features[i] - self._means[i]
            self._means[i] = self._means[i] + delta / self._count
            delta2 = features[i] - self._means[i]
            self._stds[i] = np.sqrt((self._stds[i]**2 * (self._count-1) + delta * delta2) / max(1, self._count))
        
        # Standardize | 标准化
        normalized = np.zeros_like(features)
        for i in range(min(len(features), len(self._means))):
            if self._stds[i] > 0:
                normalized[i] = (features[i] - self._means[i]) / self._stds[i]
                # Clip to reasonable range [-3,3] and map to [0,1] | 裁剪到合理范围并映射到 [0,1]
                normalized[i] = np.clip(normalized[i], -3, 3) / 3
                normalized[i] = (normalized[i] + 1) / 2
            else:
                normalized[i] = 0.5
        
        return normalized
    
    def get_feature_names(self) -> List[str]:
        """Get feature names | 获取特征名称"""
        return self._base.get_feature_names()


# ============================================================
# 4. Composite Pattern Space | 组合模式空间
# ============================================================

class CompositePatternSpace(PatternSpace):
    """
    Composite Pattern Space: Concatenates features from multiple pattern spaces.
    
    组合模式空间：连接多个模式空间的特征。
    
    Combines multiple feature extractors into one, suitable for multimodal input.
    For example: simultaneously processing text and numerical features, then concatenating them.
    
    将多个特征提取器组合成一个，适用于多模态输入。
    例如：同时处理文本和数值特征，然后将它们连接起来。
    
    Example | 示例:
        >>> # Create multiple pattern spaces | 创建多个模式空间
        >>> text_pat = SimplePatternSpace(
        ...     lambda x: [len(x)],
        ...     feature_names=['length']
        ... )
        >>> 
        >>> num_pat = StatisticalPatternSpace(
        ...     features=['mean', 'std'],
        ...     name="numbers"
        >>> )
        >>> 
        >>> # Combine them | 组合它们
        >>> composite = CompositePatternSpace([
        ...     ('text', text_pat),
        ...     ('stats', num_pat)
        ... ])
        >>> 
        >>> features = composite.extract("hello")
        >>> print(len(features))  # 3 (1 text feature + 2 statistical features)
    """
    
    def __init__(self, patterns: List[Union[PatternSpace, Tuple[str, PatternSpace]]]):
        """
        Initialize composite pattern space.
        
        初始化组合模式空间。
        
        Args | 参数:
            patterns: List of pattern spaces, can be:
                     模式空间列表，可以是：
                - PatternSpace instance: automatically generates name (pattern_0, pattern_1, etc.)
                  PatternSpace 实例：自动生成名称
                - (name, PatternSpace) tuple: specifies custom name
                  (名称, PatternSpace) 元组：指定自定义名称
        """
        super().__init__()
        
        self._patterns = []
        self._names = []
        self._dims = []
        
        for i, p in enumerate(patterns):
            if isinstance(p, tuple):
                name, pattern = p
            else:
                name = f"pattern_{i}"
                pattern = p
            
            if not isinstance(pattern, PatternSpace):
                raise TypeError(f"Item {i} must be a PatternSpace instance")
            
            self._patterns.append(pattern)
            self._names.append(name)
            self._dims.append(pattern.get_feature_dimension())
        
        self._logger = logging.getLogger("metathin.pattern.CompositePatternSpace")
    
    def extract(self, raw_input: Any) -> FeatureVector:
        """
        Extract composite features.
        
        提取组合特征。
        
        Supports two input modes | 支持两种输入模式:
            - Single input: Passed to all subspaces | 单一输入：传递给所有子空间
            - List input: Distributed to subspaces in order | 列表输入：按顺序分发给子空间
        
        Args | 参数:
            raw_input: Raw input data (single value or list) | 原始输入数据（单个值或列表）
            
        Returns | 返回:
            FeatureVector: Concatenated feature vector | 连接后的特征向量
        """
        try:
            all_features = []
            
            # Handle multipart input | 处理多部分输入
            if isinstance(raw_input, (list, tuple)) and len(raw_input) == len(self._patterns):
                # Distribute by position | 按位置分发
                inputs = raw_input
            else:
                # All subspaces use same input | 所有子空间使用相同输入
                inputs = [raw_input] * len(self._patterns)
            
            for i, (pattern, input_data) in enumerate(zip(self._patterns, inputs)):
                try:
                    features = pattern.extract(input_data)
                    
                    # Ensure 1-dimensional | 确保一维
                    if features.ndim > 1:
                        features = features.flatten()
                    
                    all_features.append(features)
                    
                except Exception as e:
                    self._logger.error(f"Pattern '{self._names[i]}' extraction failed: {e}")
                    # Zero-pad | 零填充
                    all_features.append(np.zeros(self._dims[i] if self._dims[i] > 0 else 1))
            
            # Concatenate all features | 连接所有特征
            result = np.concatenate(all_features)
            
            return result
            
        except Exception as e:
            self._logger.error(f"Composite feature extraction failed: {e}")
            raise PatternExtractionError(f"Composite feature extraction failed: {e}") from e
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names.
        
        获取特征名称。
        
        Returns feature names with prefix in format "subspace_name.feature_name".
        
        返回带有前缀的特征名称，格式为 "子空间名称.特征名称"。
        """
        names = []
        for name, pattern in zip(self._names, self._patterns):
            sub_names = pattern.get_feature_names()
            if sub_names:
                names.extend([f"{name}.{sub}" for sub in sub_names])
            else:
                dim = pattern.get_feature_dimension()
                names.extend([f"{name}_{i}" for i in range(dim)])
        
        return names


# ============================================================
# 5. Cached Pattern Space | 缓存模式空间
# ============================================================

class CachedPatternSpace(PatternSpace):
    """
    Cached Pattern Space: Caches extraction results from another pattern space.
    
    缓存模式空间：缓存另一个模式空间的提取结果。
    
    For computationally expensive feature extraction, caching improves efficiency.
    Supports cache size limits and LRU (Least Recently Used) eviction policy.
    
    对于计算成本高的特征提取，缓存可以提高效率。
    支持缓存大小限制和 LRU（最近最少使用）淘汰策略。
    
    Example | 示例:
        >>> # Create expensive pattern space | 创建昂贵的模式空间
        >>> expensive = ExpensivePatternSpace()
        >>> 
        >>> # Add caching | 添加缓存
        >>> cached = CachedPatternSpace(expensive, cache_size=100)
        >>> 
        >>> # First extraction computes, subsequent identical inputs use cache
        >>> # 第一次提取计算，后续相同输入使用缓存
        >>> f1 = cached.extract("same_input")  # Computes | 计算
        >>> f2 = cached.extract("same_input")  # Cache hit | 缓存命中
    """
    
    def __init__(self, 
                 base_pattern: PatternSpace,
                 cache_size: int = 100):
        """
        Initialize cached pattern space.
        
        初始化缓存模式空间。
        
        Args | 参数:
            base_pattern: Base pattern space to wrap | 要包装的基础模式空间
            cache_size: Cache size limit | 缓存大小限制
        """
        super().__init__()
        
        self._base = base_pattern
        self._cache_size = cache_size
        
        # Cache: {input_hash: (feature_vector, timestamp)} | 缓存：{输入哈希: (特征向量, 时间戳)}
        self._cache: Dict[int, Tuple[FeatureVector, float]] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        self._logger = logging.getLogger("metathin.pattern.CachedPatternSpace")
    
    def _hash_input(self, raw_input: Any) -> int:
        """
        Compute input hash for cache key generation.
        
        计算输入哈希用于生成缓存键。
        
        Args | 参数:
            raw_input: Raw input data | 原始输入数据
            
        Returns | 返回:
            int: Hash value | 哈希值
        """
        try:
            return hash(raw_input)
        except:
            # If unhashable, use string representation | 如果不可哈希，使用字符串表示
            return hash(str(raw_input))
    
    def _clean_cache(self):
        """Clean oldest cache entries (LRU policy) | 清理最旧的缓存条目（LRU 策略）"""
        if len(self._cache) <= self._cache_size:
            return
        
        # Sort by timestamp, delete oldest | 按时间戳排序，删除最旧的
        items = list(self._cache.items())
        items.sort(key=lambda x: x[1][1])
        
        to_remove = len(self._cache) - self._cache_size
        for key, _ in items[:to_remove]:
            del self._cache[key]
        
        self._logger.debug(f"Cleaned {to_remove} cache entries")
    
    def extract(self, raw_input: Any) -> FeatureVector:
        """
        Extract features with caching.
        
        使用缓存提取特征。
        
        Implementation steps | 实现步骤:
            1. Compute input hash | 计算输入哈希
            2. Check cache hit | 检查缓存命中
            3. If hit, return cached result | 如果命中，返回缓存结果
            4. If miss, call base pattern space and cache result | 如果未命中，调用基础模式空间并缓存结果
        
        Args | 参数:
            raw_input: Raw input data | 原始输入数据
            
        Returns | 返回:
            FeatureVector: Extracted feature vector | 提取的特征向量
        """
        input_hash = self._hash_input(raw_input)
        
        # Check cache | 检查缓存
        if input_hash in self._cache:
            features, _ = self._cache[input_hash]
            self._cache_hits += 1
            self._logger.debug(f"Cache hit: {input_hash}")
            return features.copy()  # Return copy to prevent modification | 返回副本以防止修改
        
        self._cache_misses += 1
        self._logger.debug(f"Cache miss: {input_hash}")
        
        # Compute features | 计算特征
        features = self._base.extract(raw_input)
        
        # Store in cache | 存储到缓存
        self._cache[input_hash] = (features.copy(), time.time())
        self._clean_cache()
        
        return features
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        获取缓存统计信息。
        
        Returns | 返回:
            Dict: Cache statistics | 缓存统计信息
        """
        total = self._cache_hits + self._cache_misses
        return {
            'size': len(self._cache),
            'capacity': self._cache_size,
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'hit_rate': self._cache_hits / max(1, total),
        }
    
    def clear_cache(self) -> None:
        """Clear all cached results | 清空所有缓存结果"""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        self._logger.info("Cache cleared")
    
    def get_feature_names(self) -> List[str]:
        """Get feature names | 获取特征名称"""
        return self._base.get_feature_names()
    
    def get_feature_dimension(self) -> int:
        """Get feature dimension | 获取特征维度"""
        return self._base.get_feature_dimension()


# ============================================================
# Export Interface | 导出接口
# ============================================================

__all__ = [
    'SimplePatternSpace',
    'StatisticalPatternSpace',
    'NormalizedPatternSpace',
    'CompositePatternSpace',
    'CachedPatternSpace',
]