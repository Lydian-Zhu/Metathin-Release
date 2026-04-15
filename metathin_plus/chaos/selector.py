# metathin_plus/chaos/selector.py
"""
Chaos Selector (S) | 混沌选择器 (S)
====================================

Evaluates fitness of each prediction behavior.
评估每个预测行为的适应度。
"""

import logging
from typing import Dict, List, Optional
import numpy as np

from metathin.core.s_selector import Selector
from metathin.core.b_behavior import MetaBehavior
from metathin.core.types import FeatureVector, FitnessScore, ParameterDict

from .base import PredictionResult


class ChaosSelector(Selector):
    """
    Chaos Selector - S component for chaos prediction.
    混沌选择器 - 混沌预测的评估层组件。
    
    Evaluates behavior fitness based on recent prediction accuracy.
    基于近期预测准确率评估行为适应度。
    
    Fitness = 1 / (1 + recent_error) * confidence_factor
    """
    
    def __init__(
        self,
        error_window: int = 10,
        use_confidence: bool = True,
        default_fitness: float = 0.5
    ):
        """
        Initialize chaos selector.
        初始化混沌选择器。
        
        Args:
            error_window: Window for error calculation | 误差计算窗口
            use_confidence: Include behavior confidence in fitness | 是否包含行为置信度
            default_fitness: Default fitness for new behaviors | 新行为的默认适应度
        """
        super().__init__()
        self.error_window = error_window
        self.use_confidence = use_confidence
        self.default_fitness = default_fitness
        
        # Store behavior errors | 存储行为误差
        self._behavior_errors: Dict[str, List[float]] = {}
        self._behavior_confidences: Dict[str, List[float]] = {}
        self._logger = logging.getLogger("metathin_plus.chaos.selector.ChaosSelector")
    
    def record_prediction(
        self,
        behavior_name: str,
        prediction: PredictionResult,
        actual_value: float
    ) -> None:
        """
        Record prediction result for fitness calculation.
        记录预测结果用于适应度计算。
        
        Args:
            behavior_name: Name of the behavior | 行为名称
            prediction: Prediction result | 预测结果
            actual_value: Actual observed value | 实际观测值
        """
        error = abs(prediction.value - actual_value)
        
        if behavior_name not in self._behavior_errors:
            self._behavior_errors[behavior_name] = []
            self._behavior_confidences[behavior_name] = []
        
        self._behavior_errors[behavior_name].append(error)
        self._behavior_confidences[behavior_name].append(prediction.confidence)
        
        # Limit history | 限制历史
        if len(self._behavior_errors[behavior_name]) > 1000:
            self._behavior_errors[behavior_name] = self._behavior_errors[behavior_name][-1000:]
            self._behavior_confidences[behavior_name] = self._behavior_confidences[behavior_name][-1000:]
    
    def compute_fitness(
        self,
        behavior: MetaBehavior,
        features: FeatureVector
    ) -> FitnessScore:
        """
        Compute fitness for a behavior.
        计算行为的适应度。
        
        Fitness = (1 - normalized_error) * confidence_factor
        
        Args:
            behavior: Behavior to evaluate | 要评估的行为
            features: Feature vector | 特征向量
            
        Returns:
            FitnessScore: Fitness in [0, 1] | [0, 1] 范围内的适应度
        """
        behavior_name = behavior.name
        
        # Get recent errors | 获取近期误差
        errors = self._behavior_errors.get(behavior_name, [])
        confidences = self._behavior_confidences.get(behavior_name, [])
        
        if not errors:
            fitness = self.default_fitness
        else:
            # Calculate recent error | 计算近期误差
            recent_errors = errors[-self.error_window:] if len(errors) > self.error_window else errors
            avg_error = np.mean(recent_errors)
            
            # Convert error to fitness (error ~ 0 -> fitness ~ 1) | 误差转适应度
            error_fitness = 1.0 / (1.0 + avg_error)
            
            # Calculate average confidence | 计算平均置信度
            if self.use_confidence and confidences:
                recent_conf = confidences[-self.error_window:] if len(confidences) > self.error_window else confidences
                avg_confidence = np.mean(recent_conf)
            else:
                avg_confidence = 0.5
            
            # Combine | 组合
            fitness = 0.7 * error_fitness + 0.3 * avg_confidence
        
        fitness = float(np.clip(fitness, 0.0, 1.0))
        self.record_fitness(behavior_name, fitness)
        
        return fitness
    
    def get_behavior_error(self, behavior_name: str) -> Optional[float]:
        """Get recent error for a behavior | 获取行为的近期误差"""
        errors = self._behavior_errors.get(behavior_name, [])
        if not errors:
            return None
        window = min(self.error_window, len(errors))
        return float(np.mean(errors[-window:]))
    
    def reset(self) -> None:
        """Reset all recorded data | 重置所有记录数据"""
        self._behavior_errors.clear()
        self._behavior_confidences.clear()
        self.reset_history()
    
    def get_parameters(self) -> ParameterDict:
        """Get learnable parameters | 获取可学习参数"""
        return {}  # No learnable parameters | 无可学习参数