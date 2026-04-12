"""
Selector Components | 选择器组件
=================================

Provides various fitness calculators that evaluate the suitability of behaviors in the current state.
The selector serves as the agent's "evaluator," determining how appropriate each behavior is in the given context.

提供各种适应度计算器，评估行为在当前状态下的适用性。
选择器作为代理的"评估器"，确定每个行为在给定上下文中的合适程度。

Component Types | 组件类型:
    - SimpleSelector: Linear weighted sum + sigmoid | 线性加权和 + sigmoid
    - PolynomialSelector: Polynomial regression | 多项式回归
    - RuleBasedSelector: Expert system based on predefined rules | 基于预定义规则的专家系统
    - EnsembleSelector: Combines multiple selectors | 组合多个选择器
    - AdaptiveSelector: Dynamically switches between selectors | 动态切换选择器

Design Philosophy | 设计理念:
    - Learnable: Selector parameters can be adjusted by learning mechanisms
      可学习：选择器参数可由学习机制调整
    - Diversity: Supports linear, polynomial, rule-based, and other evaluation methods
      多样性：支持线性、多项式、基于规则等多种评估方法
    - Composability: Multiple selectors can be combined for improved robustness
      可组合性：多个选择器可组合以提高鲁棒性
    - Stability: Numerically stable computations prevent overflow
      稳定性：数值稳定的计算防止溢出
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
from ..core.s_selector import Selector
from ..core.b_behavior import MetaBehavior
from ..core.types import FeatureVector, FitnessScore, ParameterDict
from ..core.exceptions import FitnessComputationError, ParameterUpdateError


# ============================================================
# Helper Functions | 辅助函数
# ============================================================

def sigmoid(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Safe sigmoid function.
    
    安全的 sigmoid 函数。
    
    Maps any real number to the (0,1) interval, commonly used to convert linear combinations to probabilities.
    
    将任意实数映射到 (0,1) 区间，常用于将线性组合转换为概率。
    
    Args | 参数:
        x: Input array | 输入数组
        temperature: Temperature parameter, higher values produce smoother outputs
                     温度参数，值越高输出越平滑
        
    Returns | 返回:
        np.ndarray: Sigmoid output in (0,1) | (0,1) 范围内的 sigmoid 输出
    """
    x = np.clip(x / temperature, -500, 500)  # Prevent overflow | 防止溢出
    return 1.0 / (1.0 + np.exp(-x))


def normalize_scores(scores: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Normalize scores to [0,1] range.
    
    将分数归一化到 [0,1] 范围。
    
    Linearly maps arbitrary scores to the [0,1] interval.
    
    将任意分数线性映射到 [0,1] 区间。
    
    Args | 参数:
        scores: Input scores | 输入分数
        epsilon: Small value to prevent division by zero | 防止除零的小值
        
    Returns | 返回:
        np.ndarray: Normalized scores | 归一化后的分数
    """
    min_score = np.min(scores)
    max_score = np.max(scores)
    
    if max_score - min_score < epsilon:
        return np.ones_like(scores) * 0.5
    
    return (scores - min_score) / (max_score - min_score + epsilon)


# ============================================================
# 1. Simple Selector | 简单选择器
# ============================================================

class SimpleSelector(Selector):
    """
    Simple Selector: Computes fitness based on weighted sum.
    
    简单选择器：基于加权和计算适应度。
    
    Uses linear weighted sum: α = sigmoid(w·x + b)
    Supports dynamic expansion for new behaviors, suitable for most scenarios.
    
    使用线性加权和：α = sigmoid(w·x + b)
    支持新行为的动态扩展，适用于大多数场景。
    
    Characteristics | 特性:
        - Learnable: Weights and bias can be adjusted by learning mechanisms
          可学习：权重和偏置可由学习机制调整
        - Dynamic expansion: Automatically expands weight matrix when new behaviors are added
          动态扩展：添加新行为时自动扩展权重矩阵
        - Temperature control: Temperature parameter adjusts output smoothness
          温度控制：温度参数调整输出平滑度
    
    Attributes | 属性:
        temperature: Temperature parameter, higher values produce smoother outputs
                    温度参数，值越高输出越平滑
        max_weight: Maximum weight value to prevent gradient explosion
                    最大权重值，防止梯度爆炸
        weights: Weight matrix [n_behaviors, n_features] | 权重矩阵
        bias: Bias vector [n_behaviors] | 偏置向量
    """
    
    def __init__(self, 
                 n_features: Optional[int] = None,
                 n_behaviors: Optional[int] = None,
                 temperature: float = 2.0,
                 max_weight: float = 10.0,
                 init_scale: float = 0.1):
        """
        Initialize simple selector.
        
        初始化简单选择器。
        
        Args | 参数:
            n_features: Number of features, None for dynamic determination
                       特征数量，None 表示动态确定
            n_behaviors: Number of behaviors, None for dynamic determination
                        行为数量，None 表示动态确定
            temperature: Temperature parameter, higher values produce smoother outputs
                        温度参数，值越高输出越平滑
            max_weight: Maximum weight value to prevent gradient explosion
                       最大权重值，防止梯度爆炸
            init_scale: Weight initialization scaling factor | 权重初始化缩放因子
            
        Raises | 抛出:
            ValueError: If parameters are out of valid range | 如果参数超出有效范围
        """
        super().__init__()
        
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        
        if max_weight <= 0:
            raise ValueError(f"max_weight must be > 0, got {max_weight}")
        
        self.temperature = temperature
        self.max_weight = max_weight
        self.init_scale = init_scale
        self._logger = logging.getLogger("metathin.selector.SimpleSelector")
        
        # Behavior index mapping | 行为索引映射
        self._behavior_indices: Dict[str, int] = {}
        
        # Initialize weight matrix | 初始化权重矩阵
        if n_features is not None and n_behaviors is not None:
            if n_features <= 0 or n_behaviors <= 0:
                raise ValueError("Number of features and behaviors must be positive")
            
            self.weights = np.random.randn(n_behaviors, n_features) * init_scale
            self.bias = np.zeros(n_behaviors)
            self._feature_dim = n_features
            self._logger.debug(f"Initialized weight matrix: {n_behaviors} x {n_features}")
        else:
            self.weights = None
            self.bias = None
            self._feature_dim = None
            self._logger.debug("Using dynamic expansion mode")
        
        # Parameter update history | 参数更新历史
        self._update_history: List[Dict[str, float]] = []
    
    def _get_or_create_index(self, behavior_name: str, feature_dim: int) -> int:
        """
        Get or create behavior index.
        
        获取或创建行为索引。
        
        Automatically expands weight matrix when new behaviors appear.
        
        当新行为出现时自动扩展权重矩阵。
        
        Args | 参数:
            behavior_name: Name of the behavior | 行为名称
            feature_dim: Feature dimension | 特征维度
            
        Returns | 返回:
            int: Index of the behavior | 行为索引
        """
        if behavior_name in self._behavior_indices:
            return self._behavior_indices[behavior_name]
        
        idx = len(self._behavior_indices)
        self._behavior_indices[behavior_name] = idx
        
        self._logger.debug(f"Assigning index {idx} to new behavior '{behavior_name}'")
        
        # Dynamically expand weight matrix | 动态扩展权重矩阵
        if self.weights is None:
            # First creation | 首次创建
            self.weights = np.random.randn(idx + 1, feature_dim) * self.init_scale
            self.bias = np.zeros(idx + 1)
            self._feature_dim = feature_dim
            
        elif idx >= len(self.weights):
            # Need expansion | 需要扩展
            old_shape = self.weights.shape
            new_weights = np.vstack([
                self.weights,
                np.random.randn(idx + 1 - len(self.weights), self.weights.shape[1]) * self.init_scale
            ])
            self.weights = new_weights
            
            # Expand bias | 扩展偏置
            new_bias = np.append(self.bias, np.zeros(idx + 1 - len(self.bias)))
            self.bias = new_bias
        
        return idx
    
    def compute_fitness(self, behavior: MetaBehavior, features: FeatureVector) -> FitnessScore:
        """
        Compute fitness: α = sigmoid((w·x + b) / temperature)
        
        计算适应度：α = sigmoid((w·x + b) / temperature)
        
        Args | 参数:
            behavior: Behavior to evaluate | 要评估的行为
            features: Feature vector | 特征向量
            
        Returns | 返回:
            FitnessScore: Fitness value in [0,1] | [0,1] 范围内的适应度值
            
        Raises | 抛出:
            FitnessComputationError: If fitness computation fails | 如果适应度计算失败
        """
        try:
            # Validate features | 验证特征
            if not isinstance(features, np.ndarray):
                features = np.array(features, dtype=np.float64)
            
            if features.ndim > 1:
                features = features.flatten()
            
            # Get or create behavior index | 获取或创建行为索引
            idx = self._get_or_create_index(behavior.name, len(features))
            
            # Check index validity | 检查索引有效性
            if idx >= len(self.weights):
                raise FitnessComputationError(f"Behavior index {idx} exceeds weight matrix size ({len(self.weights)})")
            
            # Compute linear combination | 计算线性组合
            z = np.dot(self.weights[idx], features)
            
            if idx < len(self.bias):
                z += self.bias[idx]
            
            # Apply temperature | 应用温度
            z = z / self.temperature
            
            # Compute sigmoid | 计算 sigmoid
            fitness = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
            fitness = float(np.clip(fitness, 0.0, 1.0))
            
            # Record fitness | 记录适应度
            self.record_fitness(behavior.name, fitness)
            
            return fitness
            
        except Exception as e:
            self._logger.error(f"Fitness computation failed: {e}")
            raise FitnessComputationError(f"Fitness computation failed: {e}") from e
    
    def get_parameters(self) -> ParameterDict:
        """Get current learnable parameters | 获取当前可学习参数"""
        params = {}
        
        if self.weights is not None:
            for i in range(len(self.weights)):
                for j in range(self.weights.shape[1]):
                    params[f'w_{i}_{j}'] = float(self.weights[i, j])
        
        if self.bias is not None:
            for i in range(len(self.bias)):
                params[f'b_{i}'] = float(self.bias[i])
        
        return params
    
    def update_parameters(self, delta: ParameterDict) -> None:
        """
        Update parameters during learning.
        
        在学习过程中更新参数。
        
        Args | 参数:
            delta: Parameter adjustments | 参数调整量
            
        Raises | 抛出:
            ParameterUpdateError: If parameter update fails | 如果参数更新失败
        """
        try:
            update_record = {}
            
            for key, value in delta.items():
                if key.startswith('w_'):
                    parts = key.split('_')
                    if len(parts) == 3:
                        _, i_str, j_str = parts
                        try:
                            i, j = int(i_str), int(j_str)
                            
                            if self.weights is not None and i < len(self.weights) and j < self.weights.shape[1]:
                                old_val = self.weights[i, j]
                                new_val = old_val + value
                                self.weights[i, j] = np.clip(new_val, -self.max_weight, self.max_weight)
                        except ValueError:
                            pass
                
                elif key.startswith('b_'):
                    parts = key.split('_')
                    if len(parts) == 2:
                        _, i_str = parts
                        try:
                            i = int(i_str)
                            if self.bias is not None and i < len(self.bias):
                                self.bias[i] += value
                        except ValueError:
                            pass
        except Exception as e:
            raise ParameterUpdateError(f"Parameter update failed: {e}") from e


# ============================================================
# 2. Polynomial Selector | 多项式选择器
# ============================================================

class PolynomialSelector(Selector):
    """
    Polynomial Selector: Uses polynomial regression for fitness computation.
    
    多项式选择器：使用多项式回归计算适应度。
    
    Supports linear terms, square terms, cubic terms, and interaction terms,
    capturing nonlinear relationships between features.
    
    支持线性项、平方项、立方项和交互项，
    捕捉特征之间的非线性关系。
    
    Characteristics | 特性:
        - Nonlinear: Captures relationships between features through polynomial expansion
          非线性：通过多项式展开捕捉特征之间的关系
        - Interaction terms: Learns interactions between features | 交互项：学习特征之间的交互
        - Regularization: Prevents overfitting | 正则化：防止过拟合
        - Numerically stable: Feature normalization prevents overflow | 数值稳定：特征归一化防止溢出
    
    Attributes | 属性:
        degree: Polynomial degree (1-3) | 多项式次数
        include_interaction: Whether to include interaction terms | 是否包含交互项
        include_bias: Whether to include bias term | 是否包含偏置项
        regularization: L2 regularization coefficient | L2 正则化系数
        feature_means: Feature means for normalization | 特征均值，用于归一化
        feature_stds: Feature standard deviations for normalization | 特征标准差，用于归一化
    """
    
    def __init__(self, 
                 degree: int = 2,
                 n_features: Optional[int] = None,
                 n_behaviors: Optional[int] = None,
                 include_interaction: bool = True,
                 include_bias: bool = True,
                 regularization: float = 0.01,
                 normalize_features: bool = True):
        """
        Initialize polynomial selector.
        
        初始化多项式选择器。
        
        Args | 参数:
            degree: Polynomial degree, supports 1-3 | 多项式次数，支持 1-3
            n_features: Number of original features | 原始特征数量
            n_behaviors: Number of behaviors | 行为数量
            include_interaction: Whether to include interaction terms | 是否包含交互项
            include_bias: Whether to include bias term | 是否包含偏置项
            regularization: L2 regularization coefficient | L2 正则化系数
            normalize_features: Whether to normalize features | 是否归一化特征
            
        Raises | 抛出:
            ValueError: If degree is out of valid range | 如果次数超出有效范围
        """
        super().__init__()
        
        if degree < 1 or degree > 3:
            raise ValueError(f"degree must be between 1 and 3, got {degree}")
        
        self.degree = degree
        self.include_interaction = include_interaction
        self.include_bias = include_bias
        self.regularization = regularization
        self.normalize_features = normalize_features
        self._logger = logging.getLogger("metathin.selector.PolynomialSelector")
        
        # Behavior index mapping | 行为索引映射
        self._behavior_indices: Dict[str, int] = {}
        
        # Feature normalization statistics | 特征归一化统计
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None
        
        # Count polynomial features | 计算多项式特征数量
        self.n_poly = 0
        if n_features:
            self.n_poly = self._count_poly_features(n_features)
            self._feature_dim = n_features
        else:
            self._feature_dim = None
        
        # Initialize weights | 初始化权重
        if n_behaviors and self.n_poly:
            scale = 1.0 / np.sqrt(self.n_poly)
            self.weights = np.random.randn(n_behaviors, self.n_poly) * scale
            self._logger.debug(f"Initialized polynomial weights: {n_behaviors} x {self.n_poly}")
        else:
            self.weights = None
        
        # Parameter update history | 参数更新历史
        self._update_history: List[Dict[str, float]] = []
    
    def _count_poly_features(self, n: int) -> int:
        """
        Count polynomial features.
        
        计算多项式特征数量。
        
        Includes | 包含:
            - Linear terms: n | 线性项
            - Quadratic terms: n (squares) + n*(n-1)/2 (interactions, if enabled)
              二次项：n（平方）+ n*(n-1)/2（交互，如果启用）
            - Cubic terms: n (cubes) + n*(n-1) (square*linear) + n*(n-1)*(n-2)/6 (cubic interactions)
              三次项：n（立方）+ n*(n-1)（平方*线性）+ n*(n-1)*(n-2)/6（三次交互）
        """
        count = n  # Linear terms | 线性项
        
        if self.degree >= 2:
            count += n  # Square terms | 平方项
            if self.include_interaction:
                count += n * (n - 1) // 2  # Quadratic interactions | 二次交互
        
        if self.degree >= 3:
            count += n  # Cube terms | 立方项
            if self.include_interaction:
                count += n * (n - 1)  # Square * linear | 平方 * 线性
                count += n * (n - 1) * (n - 2) // 6  # Cubic interactions | 三次交互
        
        if self.include_bias:
            count += 1  # Bias term | 偏置项
        
        return count
    
    def _normalize_features(self, features: np.ndarray, update_stats: bool = False) -> np.ndarray:
        """
        Normalize features to prevent numerical overflow.
        
        归一化特征以防止数值溢出。
        
        Args | 参数:
            features: Raw features | 原始特征
            update_stats: Whether to update running statistics | 是否更新运行统计
            
        Returns | 返回:
            np.ndarray: Normalized features | 归一化后的特征
        """
        if not self.normalize_features:
            return features
        
        features = features.astype(np.float64)
        
        if update_stats:
            if self.feature_means is None:
                self.feature_means = np.mean(features)
                self.feature_stds = np.std(features) + 1e-8
            else:
                # Online update of mean and std | 在线更新均值和标准差
                n = len(features)
                delta = features - self.feature_means
                self.feature_means = self.feature_means + delta / n
                self.feature_stds = np.sqrt(
                    (self.feature_stds**2 * (n-1) + delta**2) / n
                ) + 1e-8
        
        if self.feature_means is not None and self.feature_stds is not None:
            features = (features - self.feature_means) / self.feature_stds
        
        return features
    
    def _expand_features(self, features: np.ndarray) -> np.ndarray:
        """
        Expand to polynomial features.
        
        扩展为多项式特征。
        
        Supports 1-3 degree polynomial expansion, including interaction terms.
        
        支持 1-3 次多项式展开，包括交互项。
        
        Args | 参数:
            features: Original feature vector | 原始特征向量
            
        Returns | 返回:
            np.ndarray: Expanded polynomial feature vector | 扩展后的多项式特征向量
        """
        n = len(features)
        expanded = []
        
        # 1. Linear terms | 线性项
        expanded.extend(features)
        
        # 2. Quadratic terms | 二次项
        if self.degree >= 2:
            # Square terms | 平方项
            expanded.extend(features ** 2)
            
            # Interaction terms | 交互项
            if self.include_interaction:
                for i in range(n):
                    for j in range(i + 1, n):
                        expanded.append(features[i] * features[j])
        
        # 3. Cubic terms | 三次项
        if self.degree >= 3:
            # Cube terms | 立方项
            expanded.extend(features ** 3)
            
            # Interaction terms | 交互项
            if self.include_interaction:
                # Square * linear | 平方 * 线性
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            expanded.append((features[i] ** 2) * features[j])
                
                # Cubic interactions | 三次交互
                for i in range(n):
                    for j in range(i + 1, n):
                        for k in range(j + 1, n):
                            expanded.append(features[i] * features[j] * features[k])
        
        # 4. Bias term | 偏置项
        if self.include_bias:
            expanded.append(1.0)
        
        return np.array(expanded, dtype=np.float64)
    
    def _get_or_create_index(self, behavior_name: str) -> int:
        """Get or create behavior index | 获取或创建行为索引"""
        if behavior_name in self._behavior_indices:
            return self._behavior_indices[behavior_name]
        
        idx = len(self._behavior_indices)
        self._behavior_indices[behavior_name] = idx
        return idx
    
    def compute_fitness(self, behavior: MetaBehavior, features: FeatureVector) -> FitnessScore:
        """
        Compute polynomial fitness.
        
        计算多项式适应度。
        
        Complete pipeline | 完整流程:
            1. Feature normalization | 特征归一化
            2. Polynomial expansion | 多项式展开
            3. Weighted sum | 加权和
            4. Regularization | 正则化
            5. Sigmoid mapping | Sigmoid 映射
        
        Args | 参数:
            behavior: Behavior to evaluate | 要评估的行为
            features: Feature vector | 特征向量
            
        Returns | 返回:
            FitnessScore: Fitness value in [0,1] | [0,1] 范围内的适应度值
        """
        try:
            # Get or create behavior index | 获取或创建行为索引
            name = behavior.name
            idx = self._get_or_create_index(name)
            
            # Normalize features | 归一化特征
            features_norm = self._normalize_features(features, update_stats=True)
            
            # Expand to polynomial features | 扩展为多项式特征
            poly_features = self._expand_features(features_norm)
            
            # Dynamically expand weights | 动态扩展权重
            if self.weights is None:
                self.n_poly = len(poly_features)
                self.weights = np.random.randn(idx + 1, self.n_poly) * 0.1
                self._feature_dim = len(features)
            elif idx >= len(self.weights):
                new_weights = np.vstack([
                    self.weights,
                    np.random.randn(idx + 1 - len(self.weights), self.weights.shape[1]) * 0.1
                ])
                self.weights = new_weights
            
            # Ensure dimension match | 确保维度匹配
            if len(poly_features) != self.weights.shape[1]:
                if len(poly_features) > self.weights.shape[1]:
                    poly_features = poly_features[:self.weights.shape[1]]
                else:
                    poly_features = np.pad(poly_features, 
                                         (0, self.weights.shape[1] - len(poly_features)),
                                         mode='constant', constant_values=0)
            
            # Compute weighted sum | 计算加权和
            z = np.dot(self.weights[idx], poly_features)
            
            # Apply L2 regularization | 应用 L2 正则化
            z -= self.regularization * np.sum(self.weights[idx] ** 2)
            
            # Apply sigmoid with temperature control | 应用带温度控制的 sigmoid
            fitness = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
            fitness = float(np.clip(fitness, 0.0, 1.0))
            
            # Record fitness | 记录适应度
            self.record_fitness(behavior.name, fitness)
            
            return fitness
            
        except Exception as e:
            self._logger.error(f"Polynomial fitness computation failed: {e}")
            raise FitnessComputationError(f"Polynomial fitness computation failed: {e}") from e
    
    def get_parameters(self) -> ParameterDict:
        """Get parameters | 获取参数"""
        params = {}
        if self.weights is not None:
            for i in range(len(self.weights)):
                for j in range(self.weights.shape[1]):
                    params[f'poly_w_{i}_{j}'] = float(self.weights[i, j])
        return params
    
    def update_parameters(self, delta: ParameterDict) -> None:
        """
        Update parameters.
        
        更新参数。
        
        Args | 参数:
            delta: Parameter adjustments | 参数调整量
            
        Raises | 抛出:
            ParameterUpdateError: If parameter update fails | 如果参数更新失败
        """
        try:
            for key, value in delta.items():
                if key.startswith('poly_w_'):
                    parts = key.split('_')
                    if len(parts) == 4:
                        _, _, i_str, j_str = parts
                        try:
                            i, j = int(i_str), int(j_str)
                            if self.weights is not None and i < len(self.weights) and j < self.weights.shape[1]:
                                self.weights[i, j] += value
                        except ValueError:
                            pass
        except Exception as e:
            raise ParameterUpdateError(f"Polynomial parameter update failed: {e}") from e
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance.
        
        获取特征重要性。
        
        Analyzes the contribution of each original feature to decisions.
        
        分析每个原始特征对决策的贡献。
        
        Returns | 返回:
            Dict[str, float]: Mapping from feature names to importance values
                              特征名称到重要性值的映射
        """
        if self.weights is None or self.weights.shape[1] == 0:
            return {}
        
        # Calculate average weight contribution for each original feature
        # 计算每个原始特征的平均权重贡献
        n_orig_features = self._feature_dim if self._feature_dim else 0
        if n_orig_features == 0:
            return {}
        
        importance = np.zeros(n_orig_features)
        
        for i in range(self.weights.shape[0]):
            for j in range(self.weights.shape[1]):
                # Rough estimate: distribute polynomial weights to original features
                # 粗略估计：将多项式权重分配给原始特征
                if j < n_orig_features:
                    importance[j] += abs(self.weights[i, j])
        
        importance = importance / np.sum(importance)
        
        return {f"feature_{i}": float(imp) for i, imp in enumerate(importance)}
    
    def __repr__(self) -> str:
        return f"PolynomialSelector(degree={self.degree}, features={self.n_poly}, reg={self.regularization})"


# ============================================================
# 3. Rule-Based Selector | 基于规则的选择器
# ============================================================

class RuleBasedSelector(Selector):
    """
    Rule-Based Selector: Uses predefined rules to compute fitness.
    
    基于规则的选择器：使用预定义规则计算适应度。
    
    Suitable for expert systems or deterministic scenarios where rules are clear and non-learnable.
    
    适用于规则清晰且不可学习的专家系统或确定性场景。
    
    Characteristics | 特性:
        - Deterministic: Clear rules, predictable results | 确定性：规则清晰，结果可预测
        - Expert knowledge: Encodes domain expert experience | 专家知识：编码领域专家经验
        - Non-learnable: Parameters cannot be adjusted | 不可学习：参数无法调整
    """
    
    def __init__(self, 
                 rules: Dict[str, Callable[[FeatureVector], float]],
                 default_fitness: float = 0.5):
        """
        Initialize rule-based selector.
        
        初始化基于规则的选择器。
        
        Args | 参数:
            rules: Mapping from behavior names to rule functions | 行为名称到规则函数的映射
            default_fitness: Default fitness value when no rule exists | 无规则时的默认适应度值
            
        Raises | 抛出:
            TypeError: If any rule is not callable | 如果任何规则不可调用
        """
        super().__init__()
        
        self.rules = rules
        self.default_fitness = np.clip(default_fitness, 0.0, 1.0)
        self._logger = logging.getLogger("metathin.selector.RuleBasedSelector")
        
        # Validate rules | 验证规则
        for name, rule in rules.items():
            if not callable(rule):
                raise TypeError(f"Rule '{name}' must be callable")
    
    def compute_fitness(self, behavior: MetaBehavior, features: FeatureVector) -> FitnessScore:
        """
        Compute fitness using rules.
        
        使用规则计算适应度。
        
        Args | 参数:
            behavior: Behavior to evaluate | 要评估的行为
            features: Feature vector | 特征向量
            
        Returns | 返回:
            FitnessScore: Fitness value in [0,1] | [0,1] 范围内的适应度值
        """
        try:
            name = behavior.name
            
            if name in self.rules:
                fitness = self.rules[name](features)
            else:
                self._logger.debug(f"Behavior '{name}' has no rule, using default fitness {self.default_fitness}")
                fitness = self.default_fitness
            
            fitness = float(np.clip(fitness, 0.0, 1.0))
            self.record_fitness(behavior.name, fitness)
            
            return fitness
            
        except Exception as e:
            self._logger.error(f"Rule-based fitness computation failed: {e}")
            raise FitnessComputationError(f"Rule-based fitness computation failed: {e}") from e
    
    def add_rule(self, behavior_name: str, rule: Callable[[FeatureVector], float]) -> None:
        """Add a rule | 添加规则"""
        if not callable(rule):
            raise TypeError("Rule must be callable")
        self.rules[behavior_name] = rule
    
    def remove_rule(self, behavior_name: str) -> bool:
        """Remove a rule | 移除规则"""
        if behavior_name in self.rules:
            del self.rules[behavior_name]
            return True
        return False


# ============================================================
# 4. Ensemble Selector | 集成选择器
# ============================================================

class EnsembleSelector(Selector):
    """
    Ensemble Selector: Combines results from multiple selectors.
    
    集成选择器：组合多个选择器的结果。
    
    Aggregates evaluations from multiple selectors through weighted averaging or voting,
    improving evaluation stability and accuracy.
    
    通过加权平均或投票聚合多个选择器的评估结果，
    提高评估稳定性和准确性。
    
    Characteristics | 特性:
        - Robustness: Individual selector failures don't affect overall result
          鲁棒性：单个选择器失败不影响整体结果
        - Diversity: Leverages strengths of different selectors | 多样性：利用不同选择器的优势
        - Flexibility: Supports multiple aggregation methods | 灵活性：支持多种聚合方法
    """
    
    def __init__(self,
                 selectors: List[Selector],
                 weights: Optional[List[float]] = None,
                 aggregation: str = 'weighted_average'):
        """
        Initialize ensemble selector.
        
        初始化集成选择器。
        
        Args | 参数:
            selectors: List of component selectors | 组件选择器列表
            weights: Weight for each selector, None for equal weights | 每个选择器的权重，None 表示相等权重
            aggregation: Aggregation method | 聚合方法:
                - 'weighted_average': Weighted average | 加权平均
                - 'max': Take maximum value | 取最大值
                - 'min': Take minimum value | 取最小值
                - 'median': Take median value | 取中位数
                - 'product': Product (logical AND) | 乘积（逻辑与）
                
        Raises | 抛出:
            ValueError: If selectors list is empty or weights mismatch | 如果选择器列表为空或权重不匹配
        """
        super().__init__()
        
        if not selectors:
            raise ValueError("Selectors list cannot be empty")
        
        self.selectors = selectors
        self.aggregation = aggregation
        
        # Set weights | 设置权重
        if weights is None:
            self.weights = [1.0 / len(selectors)] * len(selectors)
        else:
            if len(weights) != len(selectors):
                raise ValueError(f"Number of weights ({len(weights)}) must match number of selectors ({len(selectors)})")
            total = sum(weights)
            self.weights = [w / total for w in weights]
        
        self._logger = logging.getLogger("metathin.selector.EnsembleSelector")
    
    def compute_fitness(self, behavior: MetaBehavior, features: FeatureVector) -> FitnessScore:
        """
        Compute ensemble fitness.
        
        计算集成适应度。
        
        Args | 参数:
            behavior: Behavior to evaluate | 要评估的行为
            features: Feature vector | 特征向量
            
        Returns | 返回:
            FitnessScore: Fitness value in [0,1] | [0,1] 范围内的适应度值
        """
        try:
            # Collect evaluations from all selectors | 收集所有选择器的评估结果
            scores = []
            valid_weights = []
            
            for selector, weight in zip(self.selectors, self.weights):
                try:
                    score = selector.compute_fitness(behavior, features)
                    
                    if 0.0 <= score <= 1.0:
                        scores.append(score)
                        valid_weights.append(weight)
                except Exception as e:
                    self._logger.debug(f"Selector {type(selector).__name__} evaluation failed: {e}")
                    continue
            
            if not scores:
                return 0.5
            
            # Compute based on aggregation method | 根据聚合方法计算
            if self.aggregation == 'weighted_average':
                valid_weights = np.array(valid_weights)
                valid_weights = valid_weights / valid_weights.sum()
                fitness = np.average(scores, weights=valid_weights)
            elif self.aggregation == 'max':
                fitness = max(scores)
            elif self.aggregation == 'min':
                fitness = min(scores)
            elif self.aggregation == 'median':
                fitness = np.median(scores)
            elif self.aggregation == 'product':
                fitness = np.prod(scores)
            else:
                fitness = np.mean(scores)
            
            fitness = float(np.clip(fitness, 0.0, 1.0))
            self.record_fitness(behavior.name, fitness)
            
            return fitness
            
        except Exception as e:
            self._logger.error(f"Ensemble fitness computation failed: {e}")
            raise FitnessComputationError(f"Ensemble fitness computation failed: {e}") from e
    
    def add_selector(self, selector: Selector, weight: Optional[float] = None) -> None:
        """Add a selector to the ensemble | 向集成中添加一个选择器"""
        self.selectors.append(selector)
        
        if weight is None:
            new_weight = 1.0 / len(self.selectors)
            self.weights = [new_weight] * len(self.selectors)
        else:
            self.weights.append(weight)
            total = sum(self.weights)
            self.weights = [w / total for w in self.weights]


# ============================================================
# 5. Adaptive Selector | 自适应选择器
# ============================================================

class AdaptiveSelector(Selector):
    """
    Adaptive Selector: Dynamically adjusts strategy based on performance.
    
    自适应选择器：根据性能动态调整策略。
    
    Dynamically switches between multiple sub-selectors, choosing the one with best recent performance.
    Uses ε-greedy strategy to balance exploration and exploitation.
    
    在多个子选择器之间动态切换，选择最近性能最好的一个。
    使用 ε-greedy 策略平衡探索和利用。
    
    Characteristics | 特性:
        - Dynamic switching: Selects best-performing selector based on recent history
          动态切换：根据最近历史选择性能最佳的选择器
        - Exploration-exploitation: ε-greedy strategy balances exploration and exploitation
          探索-利用：ε-greedy 策略平衡探索和利用
        - Performance tracking: Records historical performance of each selector
          性能跟踪：记录每个选择器的历史性能
    """
    
    def __init__(self,
                 selectors: List[Selector],
                 performance_window: int = 100,
                 exploration_rate: float = 0.1):
        """
        Initialize adaptive selector.
        
        初始化自适应选择器。
        
        Args | 参数:
            selectors: List of sub-selectors | 子选择器列表
            performance_window: Performance evaluation window size | 性能评估窗口大小
            exploration_rate: Exploration rate for ε-greedy strategy | ε-greedy 策略的探索率
        """
        super().__init__()
        
        self.selectors = selectors
        self.performance_window = performance_window
        self.exploration_rate = exploration_rate
        
        # Currently active selector index | 当前激活的选择器索引
        self.active_idx = 0
        
        # Performance records | 性能记录
        self.selector_performance: List[List[float]] = [[] for _ in selectors]
        
        self._logger = logging.getLogger("metathin.selector.AdaptiveSelector")
    
    def compute_fitness(self, behavior: MetaBehavior, features: FeatureVector) -> FitnessScore:
        """
        Compute fitness using the currently active selector.
        
        使用当前激活的选择器计算适应度。
        
        Args | 参数:
            behavior: Behavior to evaluate | 要评估的行为
            features: Feature vector | 特征向量
            
        Returns | 返回:
            FitnessScore: Fitness value in [0,1] | [0,1] 范围内的适应度值
        """
        # ε-greedy exploration | ε-greedy 探索
        if np.random.random() < self.exploration_rate:
            # Exploration mode: random selection | 探索模式：随机选择
            idx = np.random.randint(len(self.selectors))
            self._logger.debug(f"Exploration mode: using selector {idx}")
        else:
            # Exploitation mode: choose best performing | 利用模式：选择性能最好的
            idx = self._get_best_selector()
        
        self.active_idx = idx
        selector = self.selectors[idx]
        
        try:
            fitness = selector.compute_fitness(behavior, features)
            return fitness
        except Exception as e:
            self._logger.error(f"Selector {idx} failed: {e}")
            return self.selectors[0].compute_fitness(behavior, features)
    
    def _get_best_selector(self) -> int:
        """Get index of the best performing selector | 获取性能最佳的选择器索引"""
        avg_performance = []
        
        for i, perf in enumerate(self.selector_performance):
            if perf:
                avg_perf = np.mean(perf[-self.performance_window:])
                avg_performance.append((avg_perf, i))
            else:
                avg_performance.append((0, i))
        
        return max(avg_performance, key=lambda x: x[0])[1]
    
    def record_performance(self, selector_idx: int, performance: float):
        """Record selector performance | 记录选择器性能"""
        if 0 <= selector_idx < len(self.selector_performance):
            self.selector_performance[selector_idx].append(performance)
            # Limit history length | 限制历史长度
            if len(self.selector_performance[selector_idx]) > 1000:
                self.selector_performance[selector_idx] = self.selector_performance[selector_idx][-1000:]


# ============================================================
# Export Interface | 导出接口
# ============================================================

__all__ = [
    'SimpleSelector',
    'PolynomialSelector',
    'RuleBasedSelector',
    'EnsembleSelector',
    'AdaptiveSelector',
]