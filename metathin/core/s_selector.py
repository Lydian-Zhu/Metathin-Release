"""
Selector Interface (S) - Evaluation Layer | 选择器接口 (S) - 评估层
====================================================================

Defines the Selector interface - evaluates behavior suitability in current state.
Computes fitness score α(Bi) = fi(q) for each behavior.

定义选择器接口 - 评估行为在当前状态下的适用性。
为每个行为计算适应度分数 α(Bi) = fi(q)。

Role in Quintuple | 五元组中的角色:
    S (Selector): Evaluation - Computes fitness for each behavior
    S (Selector): 评估 - 为每个行为计算适应度

Design Philosophy | 设计理念:
    - Learnable: Parameters can be adjusted by learning mechanism | 可学习：参数可由学习机制调整
    - Observable: Records fitness history for analysis | 可观测：记录适应度历史用于分析
    - Extensible: Supports various evaluation strategies | 可扩展：支持多种评估策略
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import logging

from .types import FeatureVector, FitnessScore, ParameterDict
from .b_behavior import MetaBehavior
from .exceptions import FitnessComputationError, ParameterUpdateError


# ============================================================
# Selector Interface | 选择器接口
# ============================================================

class Selector(ABC):
    """
    Selection Mapping S: Evaluates behavior suitability.
    
    选择映射 S：评估行为适用性。
    
    Computes fitness score α(Bi) = fi(q) for each behavior.
    The selector serves as the agent's "evaluator," can maintain internal parameters,
    and can be adjusted through the learning mechanism.
    
    为每个行为计算适应度分数 α(Bi) = fi(q)。
    选择器作为代理的"评估器"，可以维护内部参数，
    并可通过学习机制进行调整。
    
    Implementation Requirements | 实现要求:
        - compute_fitness() MUST be implemented | 必须实现 compute_fitness() 方法
        - Fitness scores MUST be in [0,1] range | 适应度分数必须在 [0,1] 范围内
        - get_parameters() and update_parameters() are optional for learnable selectors
          get_parameters() 和 update_parameters() 对于可学习选择器是可选的
    
    Example | 示例:
        >>> class LinearSelector(Selector):
        ...     '''Linear selector: α = sigmoid(w·x + b) | 线性选择器'''
        ...     
        ...     def __init__(self, n_features: int, n_behaviors: int):
        ...         super().__init__()
        ...         self.weights = np.random.randn(n_behaviors, n_features) * 0.1
        ...         self.bias = np.zeros(n_behaviors)
        ...         self._behavior_indices = {}
        ...     
        ...     def compute_fitness(self, behavior: MetaBehavior, 
        ...                        features: FeatureVector) -> float:
        ...         if behavior.name not in self._behavior_indices:
        ...             self._behavior_indices[behavior.name] = len(self._behavior_indices)
        ...         idx = self._behavior_indices[behavior.name]
        ...         z = np.dot(self.weights[idx], features) + self.bias[idx]
        ...         return 1.0 / (1.0 + np.exp(-z))
        ...     
        ...     def get_parameters(self) -> ParameterDict:
        ...         params = {}
        ...         for i in range(len(self.weights)):
        ...             for j in range(self.weights.shape[1]):
        ...                 params[f'w_{i}_{j}'] = float(self.weights[i, j])
        ...             params[f'b_{i}'] = float(self.bias[i])
        ...         return params
    """
    
    def __init__(self):
        """Initialize selector base class."""
        self._logger = logging.getLogger(f"metathin.selector.{self.__class__.__name__}")
        self._fitness_history: Dict[str, List[float]] = {}
        """Fitness history per behavior | 每个行为的适应度历史"""
    
    @abstractmethod
    def compute_fitness(self, 
                       behavior: MetaBehavior, 
                       features: FeatureVector) -> FitnessScore:
        """
        Compute fitness score for a behavior given the current features.
        
        计算给定当前特征下行为的适应度分数。
        
        This is the core method of the selector and must be implemented.
        
        这是选择器的核心方法，必须实现。
        
        Args | 参数:
            behavior: Behavior to evaluate | 要评估的行为
            features: Current feature vector | 当前特征向量
            
        Returns | 返回:
            FitnessScore: Fitness score in the range [0,1]
                          [0,1] 范围内的适应度分数
            
        Raises | 抛出:
            FitnessComputationError: When computation fails | 计算失败时
        """
        pass
    
    def get_parameters(self) -> ParameterDict:
        """
        Get current learnable parameters (optional implementation).
        
        获取当前可学习参数（可选实现）。
        
        These parameters will be adjusted by the learning mechanism.
        Parameter format must be compatible with update_parameters().
        
        这些参数将由学习机制调整。
        参数格式必须与 update_parameters() 兼容。
        
        Returns | 返回:
            ParameterDict: Parameter dictionary, empty if not learnable
                          参数字典，不可学习时返回空字典
        """
        return {}
    
    def update_parameters(self, delta: ParameterDict) -> None:
        """
        Update parameters based on adjustments from learning mechanism (optional).
        
        根据学习机制的调整更新参数（可选）。
        
        Args | 参数:
            delta: Parameter adjustment dictionary | 参数调整字典
            
        Raises | 抛出:
            ParameterUpdateError: When update fails | 更新失败时
        """
        pass
    
    def record_fitness(self, behavior_name: str, fitness: float) -> None:
        """
        Record fitness history for analysis and debugging.
        
        记录适应度历史，用于分析和调试。
        
        Args | 参数:
            behavior_name: Name of the behavior | 行为名称
            fitness: Fitness value to record | 要记录的适应度值
        """
        if behavior_name not in self._fitness_history:
            self._fitness_history[behavior_name] = []
        self._fitness_history[behavior_name].append(fitness)
        
        # Limit history size to prevent memory bloat | 限制历史大小防止内存膨胀
        if len(self._fitness_history[behavior_name]) > 1000:
            self._fitness_history[behavior_name] = self._fitness_history[behavior_name][-1000:]
    
    def get_fitness_history(self, behavior_name: Optional[str] = None) -> Dict[str, List[float]]:
        """
        Get fitness history.
        
        获取适应度历史。
        
        Args | 参数:
            behavior_name: Specific behavior name, returns all if None
                          特定行为名称，为 None 时返回所有
            
        Returns | 返回:
            Dict: Mapping from behavior name to fitness history
                 行为名称到适应度历史的映射
        """
        if behavior_name:
            return {behavior_name: self._fitness_history.get(behavior_name, [])}
        return self._fitness_history.copy()
    
    def reset_history(self) -> None:
        """Reset fitness history."""
        self._fitness_history.clear()


# ============================================================
# Export Interface | 导出接口
# ============================================================

__all__ = [
    'Selector',                 # Selector interface | 选择器接口
    'FitnessComputationError',  # Fitness computation exception | 适应度计算异常
    'ParameterUpdateError',     # Parameter update exception | 参数更新异常
]