"""
Scientific Discovery Selector | 科学发现选择器
===============================================

Selector adapter that uses function matching to evaluate behavior fitness.
使用函数匹配来评估行为适应度的选择器适配器。

This selector serves as the evaluation layer (S) in Metathin,
evaluating how well a behavior matches the current data pattern.
此选择器作为 Metathin 中的评估层 (S)，
评估行为与当前数据模式的匹配程度。
"""

import numpy as np
from typing import List, Dict, Any, Optional
import logging

from metathin.core.s_selector import Selector
from metathin.core.b_behavior import MetaBehavior
from metathin.core.types import FeatureVector, FitnessScore, ParameterDict

from ..core.function_space import FunctionSpace, FunctionVector, VectorSpaceConfig
from ..core.function_library import FunctionLibrary, FunctionEntry
from ..core.laurent_expander import LaurentExpander


class SciSelectorAdapter(Selector):
    """
    Scientific Discovery Selector | 科学发现选择器
    
    Evaluates behavior fitness based on function vector similarity.
    基于函数向量相似度评估行为适应度。
    
    This selector matches the current data pattern (represented as a function vector)
    against registered functions and assigns fitness based on similarity.
    此选择器将当前数据模式（表示为函数向量）与注册的函数进行匹配，
    并根据相似度分配适应度。
    
    Features | 特性:
        - Dynamic function matching | 动态函数匹配
        - Similarity-based fitness | 基于相似度的适应度
        - Automatic parameter fitting | 自动参数拟合
        - Configurable thresholds | 可配置阈值
    
    Example | 示例:
        >>> selector = SciSelectorAdapter()
        >>> 
        >>> # Register custom function pattern | 注册自定义函数模式
        >>> selector.register_function_pattern("sine", "sin(x)", fitness=0.9)
        >>> 
        >>> # Compute fitness for a behavior | 计算行为的适应度
        >>> fitness = selector.compute_fitness(behavior, feature_vector)
    """
    
    def __init__(self,
                 n_negative: int = 20,
                 n_positive: int = 20,
                 center: float = 0.0,
                 similarity_threshold: float = 0.7,
                 default_fitness: float = 0.5,
                 use_cache: bool = True):
        """
        Initialize scientific discovery selector | 初始化科学发现选择器
        
        Args:
            n_negative: Number of negative power terms | 负幂项数量
            n_positive: Number of positive power terms | 正幂项数量
            center: Expansion center point | 展开中心点
            similarity_threshold: Minimum similarity for positive fitness | 正适应度的最小相似度
            default_fitness: Default fitness when no match | 无匹配时的默认适应度
            use_cache: Whether to cache results | 是否缓存结果
        """
        super().__init__()
        
        self.similarity_threshold = similarity_threshold
        self.default_fitness = default_fitness
        self.use_cache = use_cache
        
        # Initialize function space | 初始化函数空间
        config = VectorSpaceConfig(n_negative=n_negative, n_positive=n_positive, center=center)
        self._space = FunctionSpace(config)
        self._expander = LaurentExpander(n_negative, n_positive, center, use_cache)
        self._library = FunctionLibrary(self._space)
        
        # Behavior to function mapping | 行为到函数的映射
        self._behavior_function_map: Dict[str, str] = {}
        self._behavior_fitness_map: Dict[str, float] = {}
        
        # Cache | 缓存
        self._vector_cache: Dict[str, FeatureVector] = {}
        
        self._logger = logging.getLogger("metathin_sci.selector.SciSelectorAdapter")
        self._logger.info(f"Initialized: threshold={similarity_threshold}, dim={config.dimension}")
    
    def register_function_pattern(self,
                                   behavior_name: str,
                                   function_expr: str,
                                   fitness: float = None,
                                   parameters: List[str] = None) -> bool:
        """
        Register a function pattern for a behavior | 为行为注册函数模式
        
        When this function pattern matches the data, the behavior gets higher fitness.
        当此函数模式与数据匹配时，行为获得更高的适应度。
        
        Args:
            behavior_name: Name of the behavior | 行为名称
            function_expr: Function expression | 函数表达式
            fitness: Fitness value when matched (default: similarity) | 匹配时的适应度
            parameters: Parameter names | 参数名称
            
        Returns:
            bool: Success status | 成功状态
        """
        try:
            # Register function in library | 在库中注册函数
            self._library.register(
                name=f"pattern_{behavior_name}",
                expr=function_expr,
                parameters=parameters or [],
                tags=['pattern']
            )
            
            self._behavior_function_map[behavior_name] = f"pattern_{behavior_name}"
            if fitness is not None:
                self._behavior_fitness_map[behavior_name] = fitness
            
            self._logger.info(f"Registered pattern for behavior: {behavior_name}")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to register pattern: {e}")
            return False
    
    def compute_fitness(self, behavior: MetaBehavior, features: FeatureVector) -> FitnessScore:
        """
        Compute fitness for a behavior | 计算行为的适应度
        
        The fitness is determined by how well the behavior's registered function
        matches the current data pattern.
        
        适应度由行为注册的函数与当前数据模式的匹配程度决定。
        
        Args:
            behavior: Behavior to evaluate | 要评估的行为
            features: Feature vector (function vector) | 特征向量（函数向量）
            
        Returns:
            FitnessScore: Fitness value in [0, 1] | [0, 1] 范围内的适应度值
        """
        behavior_name = behavior.name
        
        # Convert features to FunctionVector | 将特征转换为函数向量
        target_vector = FunctionVector(features, self._space.config)
        
        # Check if behavior has registered pattern | 检查行为是否有注册的模式
        pattern_name = self._behavior_function_map.get(behavior_name)
        
        if pattern_name is None:
            # No registered pattern, use default fitness | 无注册模式，使用默认适应度
            fitness = self.default_fitness
            self.record_fitness(behavior_name, fitness)
            return fitness
        
        # Get pattern vector | 获取模式向量
        pattern_entry = self._library.get(pattern_name)
        if pattern_entry is None:
            fitness = self.default_fitness
            self.record_fitness(behavior_name, fitness)
            return fitness
        
        # Compute similarity | 计算相似度
        similarity = target_vector.similarity(pattern_entry.vector)
        
        # Convert similarity to fitness | 将相似度转换为适应度
        if similarity >= self.similarity_threshold:
            # Use custom fitness if set, otherwise use similarity | 使用自定义适应度或相似度
            fitness = self._behavior_fitness_map.get(behavior_name, similarity)
        else:
            # Low similarity, scale down | 低相似度，按比例降低
            fitness = similarity / self.similarity_threshold * 0.5
        
        fitness = float(np.clip(fitness, 0.0, 1.0))
        self.record_fitness(behavior_name, fitness)
        
        return fitness
    
    def compute_fitness_from_data(self,
                                   behavior: MetaBehavior,
                                   x_vals: np.ndarray,
                                   y_vals: np.ndarray) -> FitnessScore:
        """
        Compute fitness directly from (x, y) data | 直接从 (x, y) 数据计算适应度
        
        This method extracts the function vector from raw data and then
        computes fitness based on pattern matching.
        
        此方法从原始数据提取函数向量，然后基于模式匹配计算适应度。
        
        Args:
            behavior: Behavior to evaluate | 要评估的行为
            x_vals: X coordinates | X 坐标
            y_vals: Y values | Y 值
            
        Returns:
            FitnessScore: Fitness value | 适应度值
        """
        # Extract function vector | 提取函数向量
        vector = self._expander.expand_data(x_vals, y_vals)
        
        # Compute fitness using the vector | 使用向量计算适应度
        return self.compute_fitness(behavior, vector.coefficients)
    
    def match_function(self, features: FeatureVector, top_k: int = 3) -> List[tuple]:
        """
        Match function pattern against library | 匹配库中的函数模式
        
        Args:
            features: Feature vector | 特征向量
            top_k: Number of top matches to return | 返回的最佳匹配数量
            
        Returns:
            List[tuple]: (function_name, similarity) pairs | (函数名, 相似度) 对
        """
        target_vector = FunctionVector(features, self._space.config)
        return self._library.match(target_vector, threshold=self.similarity_threshold, top_k=top_k)
    
    def register_custom_function(self,
                                 name: str,
                                 expr,
                                 parameters: List[str] = None,
                                 tags: List[str] = None) -> bool:
        """
        Register a custom function in the library | 在库中注册自定义函数
        
        Args:
            name: Function name | 函数名称
            expr: Function expression | 函数表达式
            parameters: Parameter names | 参数名称
            tags: Search tags | 搜索标签
            
        Returns:
            bool: Success status | 成功状态
        """
        return self._library.register(name, expr, parameters=parameters, tags=tags)
    
    def set_behavior_pattern(self, behavior_name: str, function_name: str, fitness: float = None) -> bool:
        """
        Set the function pattern for a behavior | 设置行为的函数模式
        
        Args:
            behavior_name: Name of the behavior | 行为名称
            function_name: Name of registered function | 已注册函数的名称
            fitness: Custom fitness value (optional) | 自定义适应度值（可选）
            
        Returns:
            bool: Success status | 成功状态
        """
        if function_name not in self._library:
            self._logger.warning(f"Function not found: {function_name}")
            return False
        
        self._behavior_function_map[behavior_name] = function_name
        if fitness is not None:
            self._behavior_fitness_map[behavior_name] = fitness
        
        return True
    
    def get_parameters(self) -> ParameterDict:
        """
        Get learnable parameters | 获取可学习参数
        
        Returns:
            ParameterDict: Empty dict (no learnable parameters) | 空字典（无可学习参数）
        """
        return {}
    
    def update_parameters(self, delta: ParameterDict) -> None:
        """
        Update parameters (not supported) | 更新参数（不支持）
        
        Args:
            delta: Parameter adjustments | 参数调整
        """
        self._logger.debug("Parameter update not supported for SciSelectorAdapter")
    
    def get_library_stats(self) -> Dict[str, Any]:
        """Get function library statistics | 获取函数库统计"""
        return self._library.get_statistics()
    
    def list_patterns(self) -> Dict[str, str]:
        """List all registered behavior patterns | 列出所有注册的行为模式"""
        return self._behavior_function_map.copy()
    
    def clear_patterns(self) -> None:
        """Clear all behavior patterns | 清空所有行为模式"""
        self._behavior_function_map.clear()
        self._behavior_fitness_map.clear()
        self._logger.info("Cleared all patterns")
    
    def __repr__(self) -> str:
        return (f"SciSelectorAdapter(threshold={self.similarity_threshold}, "
                f"patterns={len(self._behavior_function_map)}, "
                f"library={len(self._library)})")