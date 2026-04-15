# metathin_plus/chaos/decision.py
"""
Chaos Decision Strategies (D) | 混沌决策策略 (D)
=================================================

Selects the best prediction behavior based on fitness scores.
基于适应度分数选择最佳预测行为。
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
import random

from metathin.core.d_decision import DecisionStrategy
from metathin.core.b_behavior import MetaBehavior
from metathin.core.types import FeatureVector, FitnessScore

from .base import PredictionResult


class MinErrorStrategy(DecisionStrategy):
    """
    Select behavior with minimum recent error.
    选择近期误差最小的行为。
    
    Pure exploitation strategy - always picks the best performing behavior.
    纯利用策略 - 总是选择表现最好的行为。
    """
    
    def __init__(self, error_window: int = 10):
        """
        Initialize min error strategy.
        初始化最小误差策略。
        
        Args:
            error_window: Window for error calculation | 误差计算窗口
        """
        self.error_window = error_window
        self._behavior_errors: Dict[str, List[float]] = {}
        self._logger = logging.getLogger("metathin_plus.chaos.decision.MinErrorStrategy")
    
    def record_error(self, behavior_name: str, error: float) -> None:
        """Record prediction error | 记录预测误差"""
        if behavior_name not in self._behavior_errors:
            self._behavior_errors[behavior_name] = []
        self._behavior_errors[behavior_name].append(error)
        
        if len(self._behavior_errors[behavior_name]) > 1000:
            self._behavior_errors[behavior_name] = self._behavior_errors[behavior_name][-1000:]
    
    def select(
        self,
        behaviors: List[MetaBehavior],
        fitness_scores: List[FitnessScore],
        features: FeatureVector
    ) -> MetaBehavior:
        """
        Select behavior with minimum error (highest fitness).
        选择误差最小（适应度最高）的行为。
        
        Args:
            behaviors: List of behaviors | 行为列表
            fitness_scores: Fitness scores | 适应度分数
            features: Feature vector | 特征向量
            
        Returns:
            MetaBehavior: Selected behavior | 选中的行为
        """
        if not behaviors:
            raise NoBehaviorError("No behaviors available")
        
        # Find index with highest fitness | 找到适应度最高的索引
        best_idx = int(np.argmax(fitness_scores))
        
        return behaviors[best_idx]
    
    def get_confidence(self, fitness_scores: List[FitnessScore]) -> float:
        """Get decision confidence | 获取决策置信度"""
        if len(fitness_scores) < 2:
            return 1.0
        
        sorted_scores = sorted(fitness_scores, reverse=True)
        diff = sorted_scores[0] - sorted_scores[1]
        return float(np.clip(diff, 0.0, 1.0))


class WeightedVoteStrategy(DecisionStrategy):
    """
    Weighted vote strategy - combines predictions from all behaviors.
    加权投票策略 - 综合所有行为的预测。
    
    Weighted average based on fitness scores.
    基于适应度分数的加权平均。
    """
    
    def __init__(self, temperature: float = 1.0):
        """
        Initialize weighted vote strategy.
        初始化加权投票策略。
        
        Args:
            temperature: Temperature for softmax weighting | Softmax 权重温度
        """
        self.temperature = temperature
        self._last_predictions: Dict[str, PredictionResult] = {}
        self._logger = logging.getLogger("metathin_plus.chaos.decision.WeightedVoteStrategy")
    
    def record_prediction(self, behavior_name: str, prediction: PredictionResult) -> None:
        """Record prediction for voting | 记录预测用于投票"""
        self._last_predictions[behavior_name] = prediction
    
    def select(
        self,
        behaviors: List[MetaBehavior],
        fitness_scores: List[FitnessScore],
        features: FeatureVector
    ) -> MetaBehavior:
        """
        Return a dummy behavior (actual prediction is aggregated separately).
        返回一个虚拟行为（实际预测单独聚合）。
        
        The actual weighted prediction is handled by the agent.
        实际的加权预测由智能体处理。
        
        Returns:
            MetaBehavior: The first behavior (as placeholder) | 第一个行为（作为占位符）
        """
        if not behaviors:
            raise NoBehaviorError("No behaviors available")
        
        # Return first behavior as placeholder | 返回第一个行为作为占位符
        # The weighted prediction will be computed separately
        # 加权预测将单独计算
        return behaviors[0]
    
    def get_weighted_prediction(self, predictions: List[tuple]) -> float:
        """
        Compute weighted average prediction.
        计算加权平均预测。
        
        Args:
            predictions: List of (behavior_name, prediction_value, fitness)
                        (行为名, 预测值, 适应度) 列表
            
        Returns:
            float: Weighted prediction | 加权预测值
        """
        if not predictions:
            return 0.0
        
        # Compute softmax weights | 计算 Softmax 权重
        fitnesses = [p[2] for p in predictions]
        exp_f = np.exp(np.array(fitnesses) / self.temperature)
        weights = exp_f / (np.sum(exp_f) + 1e-8)
        
        # Weighted average | 加权平均
        weighted_pred = sum(p[1] * w for p, w in zip(predictions, weights))
        
        return float(weighted_pred)
    
    def get_confidence(self, fitness_scores: List[FitnessScore]) -> float:
        """Get decision confidence | 获取决策置信度"""
        if not fitness_scores:
            return 0.5
        
        # Confidence based on variance of predictions | 基于预测方差的置信度
        # Higher confidence when predictions agree | 预测一致时置信度高
        # Simplified: use fitness spread | 简化：使用适应度分布
        std = np.std(fitness_scores)
        return float(1.0 / (1.0 + std))


class AdaptiveStrategy(DecisionStrategy):
    """
    Adaptive strategy - switches between min-error and weighted voting.
    自适应策略 - 在最小误差和加权投票之间切换。
    
    Uses epsilon-greedy for exploration.
    使用 ε-greedy 进行探索。
    """
    
    def __init__(
        self,
        epsilon: float = 0.1,
        decay: float = 0.99,
        min_epsilon: float = 0.01,
        weighted_threshold: float = 0.7
    ):
        """
        Initialize adaptive strategy.
        初始化自适应策略。
        
        Args:
            epsilon: Exploration rate | 探索率
            decay: Epsilon decay rate | Epsilon 衰减率
            min_epsilon: Minimum exploration rate | 最小探索率
            weighted_threshold: Use weighted voting when confidence > threshold
                               置信度超过阈值时使用加权投票
        """
        self.epsilon = epsilon
        self.decay = decay
        self.min_epsilon = min_epsilon
        self.weighted_threshold = weighted_threshold
        self._step = 0
        self._logger = logging.getLogger("metathin_plus.chaos.decision.AdaptiveStrategy")
    
    def select(
        self,
        behaviors: List[MetaBehavior],
        fitness_scores: List[FitnessScore],
        features: FeatureVector
    ) -> MetaBehavior:
        """
        Select behavior adaptively.
        自适应选择行为。
        
        Args:
            behaviors: List of behaviors | 行为列表
            fitness_scores: Fitness scores | 适应度分数
            features: Feature vector | 特征向量
            
        Returns:
            MetaBehavior: Selected behavior | 选中的行为
        """
        if not behaviors:
            raise NoBehaviorError("No behaviors available")
        
        self._step += 1
        
        # Exploration | 探索
        if random.random() < self.epsilon:
            idx = random.randrange(len(behaviors))
            selected = behaviors[idx]
            mode = "explore"
        else:
            # Exploitation: choose best fitness | 利用：选择最佳适应度
            idx = int(np.argmax(fitness_scores))
            selected = behaviors[idx]
            mode = "exploit"
        
        # Decay epsilon | 衰减 epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)
        
        self._logger.debug(f"Selected {selected.name} ({mode}), epsilon={self.epsilon:.3f}")
        
        return selected
    
    def get_confidence(self, fitness_scores: List[FitnessScore]) -> float:
        """Get decision confidence | 获取决策置信度"""
        return 1.0 - self.epsilon
    
    def should_use_weighted(self, fitness_scores: List[FitnessScore]) -> bool:
        """Check if weighted voting should be used | 检查是否应使用加权投票"""
        if not fitness_scores:
            return False
        
        max_fitness = max(fitness_scores)
        return max_fitness < self.weighted_threshold


# Import for NoBehaviorError | 导入 NoBehaviorError
from metathin.core.exceptions import NoBehaviorError