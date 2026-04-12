"""
DecisionStrategy Interface (D) - Decision Layer | 决策策略接口 (D) - 决策层
===========================================================================

Defines the DecisionStrategy interface - selects optimal behavior based on fitness scores.
B* = σ(α), where σ is a decision strategy based on all α(Bi) values.

定义决策策略接口 - 基于适应度分数选择最优行为。
B* = σ(α)，其中 σ 是基于所有 α(Bi) 的决策策略。

Role in Quintuple | 五元组中的角色:
    D (DecisionStrategy): Decision - Selects behavior based on fitness
    D (DecisionStrategy): 决策 - 基于适应度选择行为

Design Philosophy | 设计理念:
    - Diverse: Supports deterministic, stochastic, and hybrid strategies
              支持确定性、随机性和混合策略
    - Tunable: Exploration degree can be controlled via parameters
              可通过参数控制探索程度
    - Observable: Provides confidence and probability information
                 提供置信度和概率信息
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any

from .types import FeatureVector, FitnessScore
from .b_behavior import MetaBehavior
from .exceptions import DecisionError, NoBehaviorError


# ============================================================
# DecisionStrategy Interface | 决策策略接口
# ============================================================

class DecisionStrategy(ABC):
    """
    Decision Space D: Selects optimal behavior based on fitness.
    
    决策空间 D：基于适应度选择最优行为。
    
    Different strategies embody different exploration-exploitation trade-offs.
    
    不同的策略体现了不同的探索-利用权衡。
    
    Implementation Requirements | 实现要求:
        - select() MUST be implemented | 必须实现 select() 方法
        - get_confidence() is optional | get_confidence() 是可选的
    
    Strategy Categories | 策略分类:
        - Deterministic: Always picks highest fitness (greedy)
          确定性：总是选择最高适应度（贪婪）
        - Stochastic: Probabilistic selection (exploration)
          随机性：概率选择（探索）
        - Hybrid: Combines multiple strategies
          混合：组合多种策略
        - Fair: Round-robin, ensures equal opportunity
          公平：轮询，确保均等机会
    
    Example | 示例:
        >>> class EpsilonGreedyStrategy(DecisionStrategy):
        ...     '''ε-greedy strategy: explores with probability ε'''
        ...     
        ...     def __init__(self, epsilon: float = 0.1):
        ...         self.epsilon = epsilon
        ...         self.step = 0
        ...     
        ...     def select(self, behaviors, fitness_scores, features):
        ...         self.step += 1
        ...         import random
        ...         if random.random() < self.epsilon:
        ...             return random.choice(behaviors)  # Explore
        ...         idx = max(range(len(fitness_scores)), 
        ...                    key=lambda i: fitness_scores[i])
        ...         return behaviors[idx]  # Exploit
        ...     
        ...     def get_confidence(self, fitness_scores):
        ...         return 1.0 - self.epsilon
    """
    
    @abstractmethod
    def select(self, 
               behaviors: List[MetaBehavior], 
               fitness_scores: List[FitnessScore],
               features: FeatureVector) -> MetaBehavior:
        """
        Select the optimal behavior from the candidate list.
        
        从候选列表中选择最优行为。
        
        Args | 参数:
            behaviors: List of candidate behaviors | 候选行为列表
            fitness_scores: Corresponding fitness scores | 对应的适应度分数
            features: Current feature vector | 当前特征向量
            
        Returns | 返回:
            MetaBehavior: Selected behavior | 选中的行为
            
        Raises | 抛出:
            NoBehaviorError: When no behaviors are available | 无可用行为时
            DecisionError: When other errors occur during decision | 决策过程中发生其他错误时
            
        Notes | 注意:
            - behaviors and fitness_scores must have same length
              behaviors 和 fitness_scores 长度必须相同
            - Fitness scores should already be in [0,1] range
              适应度分数应该在 [0,1] 范围内
        """
        pass
    
    def get_confidence(self, fitness_scores: List[FitnessScore]) -> float:
        """
        Calculate decision confidence (optional implementation).
        
        计算决策置信度（可选实现）。
        
        Used to assess decision reliability.
        Higher confidence indicates greater certainty in the decision.
        
        用于评估决策可靠性。
        置信度越高表示决策越确定。
        
        Default implementation: uses difference between highest and second-highest.
        
        默认实现：使用最高分与第二高分的差值。
        
        Args | 参数:
            fitness_scores: Fitness scores of all candidate behaviors
                           所有候选行为的适应度分数
            
        Returns | 返回:
            float: Confidence in the range [0,1], 1 indicates very certain
                  [0,1] 范围内的置信度，1 表示非常确定
        """
        if len(fitness_scores) < 2:
            return 1.0
        sorted_scores = sorted(fitness_scores, reverse=True)
        confidence = sorted_scores[0] - sorted_scores[1]
        return max(0.0, min(1.0, confidence))
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get strategy information (optional implementation).
        
        获取策略信息（可选实现）。
        
        Returns | 返回:
            Dict: Contains strategy name, type, parameters, etc.
                 包含策略名称、类型、参数等
        """
        return {
            'name': self.__class__.__name__,
            'type': 'decision_strategy'
        }


# ============================================================
# Export Interface | 导出接口
# ============================================================

__all__ = [
    'DecisionStrategy',   # Decision strategy interface | 决策策略接口
    'DecisionError',      # Decision exception | 决策异常
    'NoBehaviorError',    # No available behavior exception | 无可用行为异常
]