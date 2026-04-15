"""
Chaos Decision Strategies (D) | 混沌决策策略 (D)
=================================================
"""

import logging
import numpy as np
import random
from typing import List, Dict, Any, Optional

from metathin.core.d_decision import DecisionStrategy
from metathin.core.b_behavior import MetaBehavior
from metathin.core.types import FeatureVector, FitnessScore
from metathin.core.exceptions import NoBehaviorError


class MinErrorStrategy(DecisionStrategy):
    """选择适应度最高的行为"""
    
    def __init__(self):
        self._logger = logging.getLogger("metathin_plus.chaos.decision.MinErrorStrategy")
    
    def select(self, behaviors: List[MetaBehavior], fitness_scores: List[float],
               features: FeatureVector) -> MetaBehavior:
        if not behaviors:
            raise NoBehaviorError("No behaviors available")
        best_idx = int(np.argmax(fitness_scores))
        self._logger.debug(f"Selected {behaviors[best_idx].name} (fitness={fitness_scores[best_idx]:.4f})")
        return behaviors[best_idx]
    
    def get_confidence(self, fitness_scores: List[float]) -> float:
        if len(fitness_scores) < 2:
            return 1.0
        sorted_scores = sorted(fitness_scores, reverse=True)
        return float(np.clip(sorted_scores[0] - sorted_scores[1], 0.0, 1.0))


class WeightedVoteStrategy(DecisionStrategy):
    """加权投票策略"""
    
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
        self._logger = logging.getLogger("metathin_plus.chaos.decision.WeightedVoteStrategy")
    
    def select(self, behaviors: List[MetaBehavior], fitness_scores: List[float],
               features: FeatureVector) -> MetaBehavior:
        if not behaviors:
            raise NoBehaviorError("No behaviors available")
        # 返回第一个行为作为占位符，实际预测单独处理
        return behaviors[0]
    
    def get_weighted_prediction(self, predictions: List[tuple]) -> float:
        if not predictions:
            return 0.0
        fitnesses = [p[2] for p in predictions]
        exp_f = np.exp(np.array(fitnesses) / self.temperature)
        weights = exp_f / (np.sum(exp_f) + 1e-8)
        return float(sum(p[1] * w for p, w in zip(predictions, weights)))
    
    def get_confidence(self, fitness_scores: List[float]) -> float:
        if not fitness_scores:
            return 0.5
        return float(1.0 / (1.0 + np.std(fitness_scores)))


class AdaptiveStrategy(DecisionStrategy):
    """ε-greedy 自适应策略"""
    
    def __init__(self, epsilon: float = 0.1, decay: float = 0.99, min_epsilon: float = 0.01):
        self.epsilon = epsilon
        self.decay = decay
        self.min_epsilon = min_epsilon
        self._step = 0
        self._logger = logging.getLogger("metathin_plus.chaos.decision.AdaptiveStrategy")
    
    def select(self, behaviors: List[MetaBehavior], fitness_scores: List[float],
               features: FeatureVector) -> MetaBehavior:
        if not behaviors:
            raise NoBehaviorError("No behaviors available")
        
        self._step += 1
        current_epsilon = max(self.min_epsilon, self.epsilon * (self.decay ** self._step))
        
        if random.random() < current_epsilon:
            idx = random.randrange(len(behaviors))
            mode = "explore"
        else:
            idx = int(np.argmax(fitness_scores))
            mode = "exploit"
        
        self._logger.debug(f"Selected {behaviors[idx].name} ({mode}), ε={current_epsilon:.3f}")
        return behaviors[idx]
    
    def get_confidence(self, fitness_scores: List[float]) -> float:
        return 1.0 - self.epsilon