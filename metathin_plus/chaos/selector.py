"""
Chaos Selector (S) | 混沌选择器 (S)
====================================

Evaluates fitness of each prediction behavior based on current features.
基于当前特征评估每个预测行为的适应度。
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
        super().__init__()
        
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.use_history = use_history
        self.history_weight = history_weight
        
        self._behavior_indices: Dict[str, int] = {}
        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[np.ndarray] = None
        
        if n_behaviors is not None:
            self.weights = np.random.randn(n_behaviors, n_features) * 0.1
            self.bias = np.zeros(n_behaviors)
        
        self._prediction_errors: Dict[str, List[float]] = {}
        self._logger = logging.getLogger("metathin_plus.chaos.selector.ChaosSelector")
    
    def _get_or_create_index(self, behavior_name: str) -> int:
        """获取或创建行为索引"""
        if behavior_name in self._behavior_indices:
            return self._behavior_indices[behavior_name]
        
        idx = len(self._behavior_indices)
        self._behavior_indices[behavior_name] = idx
        
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
    
    def _ensure_features(self, features: FeatureVector) -> FeatureVector:
        """确保特征维度匹配"""
        if len(features) != self.n_features:
            if len(features) > self.n_features:
                return features[:self.n_features]
            else:
                return np.pad(features, (0, self.n_features - len(features)))
        return features
    
    def compute_fitness(self, behavior: MetaBehavior, features: FeatureVector) -> FitnessScore:
        """基于特征计算适应度"""
        idx = self._get_or_create_index(behavior.name)
        features = self._ensure_features(features)
        
        z = np.dot(self.weights[idx], features) + self.bias[idx]
        fitness = 1.0 / (1.0 + np.exp(-z / self.temperature))
        
        if self.use_history and behavior.name in self._prediction_errors:
            errors = self._prediction_errors[behavior.name]
            if errors:
                avg_error = np.mean(errors[-20:])
                error_fitness = 1.0 / (1.0 + avg_error)
                fitness = (1 - self.history_weight) * fitness + self.history_weight * error_fitness
        
        return float(np.clip(fitness, 0.0, 1.0))
    
    def record_prediction_error(self, behavior_name: str, error: float):
        """记录预测误差"""
        if behavior_name not in self._prediction_errors:
            self._prediction_errors[behavior_name] = []
        self._prediction_errors[behavior_name].append(error)
        
        if len(self._prediction_errors[behavior_name]) > 100:
            self._prediction_errors[behavior_name] = self._prediction_errors[behavior_name][-100:]
    
    def record_prediction(self, behavior_name: str, prediction, actual_value: float):
        """记录预测结果（兼容接口）"""
        if hasattr(prediction, 'value'):
            pred_value = prediction.value
        else:
            pred_value = prediction
        error = abs(pred_value - actual_value)
        self.record_prediction_error(behavior_name, error)
    
    def update_parameters(self, delta: ParameterDict) -> None:
        """更新权重参数"""
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
        """获取当前参数"""
        params = {}
        if self.weights is not None:
            for i in range(len(self.weights)):
                for j in range(self.weights.shape[1]):
                    params[f'w_{i}_{j}'] = float(self.weights[i, j])
        if self.bias is not None:
            for i in range(len(self.bias)):
                params[f'b_{i}'] = float(self.bias[i])
        return params
    
    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性"""
        if self.weights is None:
            return {}
        importance = np.mean(np.abs(self.weights), axis=0)
        return {f"feature_{i}": float(imp) for i, imp in enumerate(importance)}
    
    def reset(self) -> None:
        """重置状态"""
        self._prediction_errors.clear()
        self.reset_history()