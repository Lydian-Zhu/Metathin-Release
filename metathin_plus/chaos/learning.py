# metathin_plus/chaos/learning.py
"""
Chaos Learning Mechanisms (Ψ) | 混沌学习机制 (Ψ)
=================================================

Updates selector parameters based on prediction errors.
基于预测误差更新选择器参数。
"""

import logging
from typing import Any, Dict, List, Optional
import numpy as np

from metathin.core.psi_learning import LearningMechanism
from metathin.core.types import ParameterDict


class ErrorLearning(LearningMechanism):
    """
    Error-based learning - adjusts behavior weights based on prediction error.
    基于误差的学习 - 根据预测误差调整行为权重。
    
    When a behavior makes a large error, its weight decreases.
    当行为产生较大误差时，其权重降低。
    """
    
    def __init__(self, learning_rate: float = 0.1, error_threshold: float = 0.5):
        """
        Initialize error learning.
        初始化误差学习。
        
        Args:
            learning_rate: Learning rate for weight updates | 权重更新的学习率
            error_threshold: Error threshold for negative learning | 负学习的误差阈值
        """
        self.learning_rate = learning_rate
        self.error_threshold = error_threshold
        self._weight_history: Dict[str, List[float]] = {}
        self._logger = logging.getLogger("metathin_plus.chaos.learning.ErrorLearning")
    
    def compute_adjustment(
        self,
        expected: Any,
        actual: Any,
        context: Dict[str, Any]
    ) -> Optional[ParameterDict]:
        """
        Compute parameter adjustments based on error.
        基于误差计算参数调整。
        
        Args:
            expected: Expected value (prediction) | 期望值（预测值）
            actual: Actual value | 实际值
            context: Context containing behavior_name, etc.
                     包含行为名称等的上下文
            
        Returns:
            Optional[ParameterDict]: Adjustments for parameters | 参数调整量
        """
        # Convert to float | 转换为浮点数
        try:
            pred = float(expected)
            actual_val = float(actual)
        except (TypeError, ValueError):
            return None
        
        error = abs(pred - actual_val)
        behavior_name = context.get('behavior_name', 'unknown')
        
        # Record error | 记录误差
        if behavior_name not in self._weight_history:
            self._weight_history[behavior_name] = []
        self._weight_history[behavior_name].append(error)
        
        # Calculate adjustment | 计算调整量
        if error > self.error_threshold:
            # Negative adjustment | 负调整
            adjustment = -self.learning_rate * error
        else:
            # Positive adjustment (small) | 正调整（小）
            adjustment = self.learning_rate * (1.0 - error / self.error_threshold)
        
        adjustment = float(np.clip(adjustment, -0.5, 0.5))
        
        return {f"weight_{behavior_name}": adjustment}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics | 获取学习统计"""
        avg_errors = {}
        for name, errors in self._weight_history.items():
            if errors:
                avg_errors[name] = float(np.mean(errors[-100:]))
        
        return {
            'name': self.__class__.__name__,
            'type': 'error_based',
            'learning_rate': self.learning_rate,
            'error_threshold': self.error_threshold,
            'avg_errors': avg_errors
        }


class RewardLearning(LearningMechanism):
    """
    Reward-based learning - reinforces behaviors that make accurate predictions.
    基于奖励的学习 - 强化预测准确的行为。
    
    Reward = 1 - normalized_error
    奖励 = 1 - 归一化误差
    """
    
    def __init__(self, learning_rate: float = 0.1, baseline: float = 0.5):
        """
        Initialize reward learning.
        初始化奖励学习。
        
        Args:
            learning_rate: Learning rate | 学习率
            baseline: Baseline reward for comparison | 用于比较的基线奖励
        """
        self.learning_rate = learning_rate
        self.baseline = baseline
        self._rewards: Dict[str, List[float]] = {}
        self._cumulative_reward: Dict[str, float] = {}
        self._logger = logging.getLogger("metathin_plus.chaos.learning.RewardLearning")
    
    def compute_adjustment(
        self,
        expected: Any,
        actual: Any,
        context: Dict[str, Any]
    ) -> Optional[ParameterDict]:
        """
        Compute parameter adjustments based on reward.
        基于奖励计算参数调整。
        
        Args:
            expected: Expected value (prediction) | 期望值（预测值）
            actual: Actual value | 实际值
            context: Context containing behavior_name | 包含行为名称的上下文
            
        Returns:
            Optional[ParameterDict]: Adjustments for parameters | 参数调整量
        """
        try:
            pred = float(expected)
            actual_val = float(actual)
        except (TypeError, ValueError):
            return None
        
        # Calculate reward | 计算奖励
        error = abs(pred - actual_val)
        reward = 1.0 / (1.0 + error)
        
        behavior_name = context.get('behavior_name', 'unknown')
        
        # Record reward | 记录奖励
        if behavior_name not in self._rewards:
            self._rewards[behavior_name] = []
            self._cumulative_reward[behavior_name] = self.baseline
        self._rewards[behavior_name].append(reward)
        
        # Calculate advantage | 计算优势
        advantage = reward - self._cumulative_reward[behavior_name]
        
        # Update cumulative reward | 更新累积奖励
        n = len(self._rewards[behavior_name])
        self._cumulative_reward[behavior_name] = (
            self._cumulative_reward[behavior_name] * (n - 1) + reward
        ) / n
        
        # Calculate adjustment | 计算调整量
        adjustment = self.learning_rate * advantage
        adjustment = float(np.clip(adjustment, -0.5, 0.5))
        
        return {f"weight_{behavior_name}": adjustment}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics | 获取学习统计"""
        avg_rewards = {}
        for name, rewards in self._rewards.items():
            if rewards:
                avg_rewards[name] = float(np.mean(rewards[-100:]))
        
        return {
            'name': self.__class__.__name__,
            'type': 'reward_based',
            'learning_rate': self.learning_rate,
            'baseline': self.baseline,
            'avg_rewards': avg_rewards
        }