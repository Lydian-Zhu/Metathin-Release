"""
Laurent Series Expander | 洛朗级数展开器
=========================================

Provides flexible Laurent series expansion for both symbolic and numerical functions.
为符号函数和数值函数提供灵活的洛朗级数展开。

Features | 特性:
    - Symbolic expansion using SymPy | 使用 SymPy 进行符号展开
    - Numerical expansion via sampling | 通过采样进行数值展开
    - Configurable order and center | 可配置阶数和中心点
    - Vector caching for performance | 向量缓存以提高性能
"""

import numpy as np
from typing import Callable, Optional, Union, Dict, Any
from functools import lru_cache
import logging

from .function_space import FunctionSpace, FunctionVector, VectorSpaceConfig


class LaurentExpander:
    """
    Laurent Series Expander | 洛朗级数展开器
    
    Provides unified interface for expanding functions into Laurent series.
    提供将函数展开为洛朗级数的统一接口。
    
    Example | 示例:
        >>> expander = LaurentExpander(n_negative=20, n_positive=20)
        >>> 
        >>> # Expand symbolic function | 展开符号函数
        >>> from sympy import symbols, sin
        >>> x = symbols('x')
        >>> vector = expander.expand_symbolic(sin(x))
        >>> 
        >>> # Expand numerical function | 展开数值函数
        >>> def f(x): return np.sin(x)
        >>> vector = expander.expand_numerical(f, -5, 5, 100)
    """
    
    def __init__(self, 
                 n_negative: int = 20,
                 n_positive: int = 20,
                 center: float = 0.0,
                 use_cache: bool = True):
        """
        Initialize Laurent expander | 初始化洛朗展开器
        
        Args:
            n_negative: Number of negative power terms | 负幂项数量
            n_positive: Number of positive power terms | 正幂项数量
            center: Expansion center point | 展开中心点
            use_cache: Whether to cache expansion results | 是否缓存展开结果
        """
        self.config = VectorSpaceConfig(n_negative=n_negative, n_positive=n_positive, center=center)
        self.space = FunctionSpace(self.config)
        self.use_cache = use_cache
        self._logger = logging.getLogger("metathin_sci.LaurentExpander")
        
        # Cache for numerical expansions | 数值展开缓存
        self._cache: Dict[str, FunctionVector] = {}
    
    def expand_symbolic(self, expr, x_symbol=None) -> FunctionVector:
        """
        Expand symbolic expression | 展开符号表达式
        
        Args:
            expr: SymPy expression | SymPy 表达式
            x_symbol: Variable symbol (default: 'x') | 变量符号
            
        Returns:
            FunctionVector: Laurent coefficient vector | 洛朗系数向量
        """
        from sympy import symbols
        
        if x_symbol is None:
            x_symbol = symbols('x')
        
        # Check cache | 检查缓存
        cache_key = f"sym_{str(expr)}"
        if self.use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        vector = self.space.from_symbolic(expr, x_symbol)
        
        if self.use_cache:
            self._cache[cache_key] = vector
        
        return vector
    
    def expand_numerical(self, 
                         func: Callable[[float], float],
                         x_min: float = -10.0,
                         x_max: float = 10.0,
                         n_samples: int = 200) -> FunctionVector:
        """
        Expand numerical function via sampling | 通过采样展开数值函数
        
        Args:
            func: Numerical function f(x) | 数值函数
            x_min: Minimum x for sampling | 采样最小值
            x_max: Maximum x for sampling | 采样最大值
            n_samples: Number of sample points | 采样点数量
            
        Returns:
            FunctionVector: Laurent coefficient vector | 洛朗系数向量
        """
        # Check cache | 检查缓存
        cache_key = f"num_{func.__name__}_{x_min}_{x_max}_{n_samples}"
        if self.use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        # Generate samples | 生成采样点
        x_vals = np.linspace(x_min, x_max, n_samples)
        y_vals = np.array([func(x) for x in x_vals])
        
        # Handle singularities by excluding center point | 处理奇点，排除中心点
        center = self.config.center
        mask = np.abs(x_vals - center) > 1e-8
        x_vals_filtered = x_vals[mask]
        y_vals_filtered = y_vals[mask]
        
        if len(x_vals_filtered) < n_samples // 2:
            self._logger.warning(f"Insufficient valid samples for {func.__name__}")
            x_vals_filtered = x_vals
            y_vals_filtered = y_vals
        
        vector = self.space.from_samples(x_vals_filtered, y_vals_filtered)
        
        if self.use_cache:
            self._cache[cache_key] = vector
        
        return vector
    
    def expand_data(self, x_vals: np.ndarray, y_vals: np.ndarray) -> FunctionVector:
        """
        Expand from raw data points | 从原始数据点展开
        
        Args:
            x_vals: X coordinates | X 坐标数组
            y_vals: Y values | Y 值数组
            
        Returns:
            FunctionVector: Laurent coefficient vector | 洛朗系数向量
        """
        # Remove points near center to avoid singularities | 移除中心点附近的点以避免奇点
        center = self.config.center
        mask = np.abs(x_vals - center) > 1e-8
        x_filtered = x_vals[mask]
        y_filtered = y_vals[mask]
        
        if len(x_filtered) < len(x_vals) // 2:
            self._logger.warning("Most points are near center, using all points")
            x_filtered = x_vals
            y_filtered = y_vals
        
        return self.space.from_samples(x_filtered, y_filtered)
    
    def expand_series(self, coefficients: np.ndarray) -> FunctionVector:
        """
        Create vector from coefficient array | 从系数数组创建向量
        
        Args:
            coefficients: Laurent coefficients | 洛朗系数
            
        Returns:
            FunctionVector: Function vector | 函数向量
        """
        return FunctionVector(coefficients, self.config)
    
    def reconstruct(self, vector: FunctionVector) -> Callable[[float], float]:
        """
        Reconstruct function from vector | 从向量重建函数
        
        Args:
            vector: Function vector | 函数向量
            
        Returns:
            Callable: Reconstructed function f(x) | 重建的函数
        """
        return self.space.to_function(vector)
    
    def compare(self, 
                vec1: FunctionVector, 
                vec2: FunctionVector,
                metric: str = 'cosine') -> float:
        """
        Compare two function vectors | 比较两个函数向量
        
        Args:
            vec1: First vector | 第一个向量
            vec2: Second vector | 第二个向量
            metric: Similarity metric ('cosine', 'euclidean', 'manhattan') | 相似度度量
            
        Returns:
            float: Similarity or distance | 相似度或距离
        """
        if metric == 'cosine':
            return vec1.similarity(vec2)
        elif metric == 'euclidean':
            return float(np.linalg.norm(vec1.coefficients - vec2.coefficients))
        elif metric == 'manhattan':
            return float(np.sum(np.abs(vec1.coefficients - vec2.coefficients)))
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def clear_cache(self):
        """Clear expansion cache | 清空展开缓存"""
        self._cache.clear()
        self._logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics | 获取缓存统计"""
        return {
            'cache_size': len(self._cache),
            'use_cache': self.use_cache,
            'dimension': self.config.dimension
        }


# ============================================================
# Convenience Functions | 便捷函数
# ============================================================

def create_default_expander() -> LaurentExpander:
    """Create a default expander with standard settings | 创建默认展开器"""
    return LaurentExpander(n_negative=20, n_positive=20, center=0.0, use_cache=True)


def expand_function(func, **kwargs) -> FunctionVector:
    """
    Quick function expansion | 快速函数展开
    
    Args:
        func: Function to expand (symbolic or callable) | 要展开的函数
        **kwargs: Additional arguments for expander | 额外参数
        
    Returns:
        FunctionVector: Laurent coefficient vector | 洛朗系数向量
    """
    expander = create_default_expander()
    
    # Try symbolic first | 先尝试符号展开
    try:
        from sympy import sympify
        expr = sympify(func)
        return expander.expand_symbolic(expr)
    except:
        # Fall back to numerical | 回退到数值展开
        if callable(func):
            return expander.expand_numerical(func, **kwargs)
        raise ValueError("Cannot expand function: not symbolic or callable")