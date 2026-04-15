# metathin_plus/chaos/selector.py - 完整修复版本

"""
Chaos Selector (S) | 混沌选择器 (S)
====================================

Evaluates fitness of each prediction behavior based on current features.
基于当前特征评估每个预测行为的适应度。

核心改进：
    - 使用特征向量计算适应度，而非简单历史平均
    - 支持在线学习和权重更新
    - 不同特征输入产生不同适应度分数
"""

import numpy as np
import logging
from typing import Dict, List, Optional

from metathin.core.s_selector import Selector
from metathin.core.b_behavior import MetaBehavior
from metathin.core.types import FeatureVector, FitnessScore, ParameterDict


class ChaosSelector(Selector):
    """
    Chaos Selector - S component for chaos prediction.
    混沌选择器 - 混沌预测的评估层组件。
    
    核心公式:
        fitness_i = sigmoid(w_i · features + b_i)
    
    这样选择器可以根据当前系统状态的特征，
    动态判断哪个行为更合适。
    """
    
    def __init__(
        self,
        n_features: int = 7,
        n_behaviors: Optional[int] = None,
        learning_rate: float = 0.01,
        temperature: float = 1.0,
        use_history: bool = True,
        history_weight: float = 0.3
    ):
        """
        Initialize chaos selector.
        初始化混沌选择器。
        
        Args:
            n_features: Feature vector dimension | 特征向量维度
            n_behaviors: Number of behaviors (optional) | 行为数量
            learning_rate: Learning rate for weight updates | 学习率
            temperature: Sigmoid temperature | Sigmoid 温度参数
            use_history: Whether to use historical error | 是否使用历史误差
            history_weight: Weight for historical error (0-1) | 历史误差权重
        """
        super().__init__()
        
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.use_history = use_history
        self.history_weight = history_weight
        
        # Behavior index mapping | 行为索引映射
        self._behavior_indices: Dict[str, int] = {}
        
        # Weight matrix [n_behaviors, n_features] | 权重矩阵
        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[np.ndarray] = None
        
        # Initialize if number of behaviors is known | 如果知道行为数量则初始化
        if n_behaviors is not None:
            self.weights = np.random.randn(n_behaviors, n_features) * 0.1
            self.bias = np.zeros(n_behaviors)
        
        # Historical prediction errors | 历史预测误差
        self._prediction_errors: Dict[str, List[float]] = {}
        
        self._logger = logging.getLogger("metathin_plus.chaos.selector.ChaosSelector")
    
    def _get_or_create_index(self, behavior_name: str) -> int:
        """Get or create behavior index | 获取或创建行为索引"""
        if behavior_name in self._behavior_indices:
            return self._behavior_indices[behavior_name]
        
        idx = len(self._behavior_indices)
        self._behavior_indices[behavior_name] = idx
        
        # Dynamically expand weight matrix | 动态扩展权重矩阵
        if self.weights is None:
            self.weights = np.random.randn(idx + 1, self.n_features) * 0.1
            self.bias = np.zeros(idx + 1)
        elif idx >= len(self.weights):
            new_weights = np.vstack([
                self.weights,
                np.random.randn(idx + 1 - len(self.weights), self.n_features) * 0.1
            ])
            self.weights = new_weights
            new_bias = np.append(self.bias, np.zeros(idx + 1 - len(self.bias)))
            self.bias = new_bias
        
        return idx
    
    def _ensure_feature_dimension(self, features: FeatureVector) -> FeatureVector:
        """Ensure feature dimension matches n_features | 确保特征维度匹配"""
        if len(features) != self.n_features:
            if len(features) > self.n_features:
                return features[:self.n_features]
            else:
                return np.pad(features, (0, self.n_features - len(features)))
        return features
    
    def compute_fitness(self, behavior: MetaBehavior, features: FeatureVector) -> FitnessScore:
        """
        Compute fitness based on current features.
        基于当前特征计算适应度。
        
        核心公式: fitness = sigmoid((w·x + b) / temperature)
        """
        idx = self._get_or_create_index(behavior.name)
        
        # Ensure feature dimension | 确保特征维度
        features = self._ensure_feature_dimension(features)
        
        # Compute linear combination | 计算线性组合
        z = np.dot(self.weights[idx], features) + self.bias[idx]
        
        # Sigmoid activation | Sigmoid 激活
        fitness = 1.0 / (1.0 + np.exp(-z / self.temperature))
        
        # Optionally blend with historical error | 可选：结合历史误差
        if self.use_history and behavior.name in self._prediction_errors:
            errors = self._prediction_errors[behavior.name]
            if errors:
                recent_errors = errors[-20:]  # Last 20 errors
                avg_error = np.mean(recent_errors)
                error_fitness = 1.0 / (1.0 + avg_error)
                # Blend: feature-based + history-based | 混合：基于特征 + 基于历史
                fitness = (1 - self.history_weight) * fitness + self.history_weight * error_fitness
        
        fitness = float(np.clip(fitness, 0.0, 1.0))
        
        self._logger.debug(f"{behavior.name}: z={z:.3f}, fitness={fitness:.3f}")
        
        return fitness
    
    def record_prediction_error(self, behavior_name: str, error: float):
        """
        Record prediction error for learning.
        记录预测误差用于学习。
        """
        if behavior_name not in self._prediction_errors:
            self._prediction_errors[behavior_name] = []
        self._prediction_errors[behavior_name].append(error)
        
        # Limit history length | 限制历史长度
        if len(self._prediction_errors[behavior_name]) > 100:
            self._prediction_errors[behavior_name] = self._prediction_errors[behavior_name][-100:]
    
    def record_prediction(self, behavior_name: str, prediction, actual_value: float):
        """
        Record prediction result (compatibility with old interface).
        记录预测结果（兼容旧接口）。
        """
        # Extract prediction value | 提取预测值
        if hasattr(prediction, 'value'):
            pred_value = prediction.value
        else:
            pred_value = prediction
        
        error = abs(pred_value - actual_value)
        self.record_prediction_error(behavior_name, error)
    
    def update_parameters(self, delta: ParameterDict) -> None:
        """
        Update weights based on learning feedback.
        基于学习反馈更新权重。
        """
        for key, value in delta.items():
            if key.startswith('w_'):
                parts = key.split('_')
                if len(parts) == 3:
                    try:
                        i, j = int(parts[1]), int(parts[2])
                        if self.weights is not None and i < len(self.weights) and j < self.weights.shape[1]:
                            self.weights[i, j] += value
                    except (ValueError, IndexError):
                        pass
            elif key.startswith('b_'):
                try:
                    i = int(key.split('_')[1])
                    if self.bias is not None and i < len(self.bias):
                        self.bias[i] += value
                except (ValueError, IndexError):
                    pass
    
    def get_parameters(self) -> ParameterDict:
        """Get current parameters | 获取当前参数"""
        params = {}
        if self.weights is not None:
            for i in range(len(self.weights)):
                for j in range(self.weights.shape[1]):
                    params[f'w_{i}_{j}'] = float(self.weights[i, j])
        if self.bias is not None:
            for i in range(len(self.bias)):
                params[f'b_{i}'] = float(self.bias[i])
        return params
    
    def get_behavior_error(self, behavior_name: str) -> Optional[float]:
        """Get recent average error for a behavior | 获取行为的近期平均误差"""
        errors = self._prediction_errors.get(behavior_name, [])
        if not errors:
            return None
        return float(np.mean(errors[-10:]))
    
    def reset(self) -> None:
        """Reset selector state | 重置选择器状态"""
        self._prediction_errors.clear()
        self.reset_history()
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance (average absolute weight).
        获取特征重要性（平均绝对权重）。
        """
        if self.weights is None:
            return {}
        
        importance = np.mean(np.abs(self.weights), axis=0)
        return {f"feature_{i}": float(imp) for i, imp in enumerate(importance)}