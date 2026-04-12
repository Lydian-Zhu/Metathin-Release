"""
LearningMechanism Interface (Ψ) - Learning Layer | 学习机制接口 (Ψ) - 学习层
=============================================================================

Defines the LearningMechanism interface - adjusts selector parameters based on feedback.
Ψ: δ ⟼ γ̂, where δ = E - ε is the residual (difference between expected and actual).

定义学习机制接口 - 基于反馈调整选择器参数。
Ψ: δ ⟼ γ̂，其中 δ = E - ε 是残差（期望与实际之差）。

Role in Quintuple | 五元组中的角色:
    Ψ (LearningMechanism): Learning - Adjusts parameters based on feedback
    Ψ (LearningMechanism): 学习 - 基于反馈调整参数

Design Philosophy | 设计理念:
    - Diverse: Supports supervised, reinforcement, and unsupervised learning
              支持监督学习、强化学习和无监督学习
    - Composable: Multiple learning mechanisms can be combined
                 可组合：多个学习机制可组合使用
    - Observable: Provides learning statistics for monitoring
                 可观测：提供学习统计用于监控
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from .types import ParameterDict
from .exceptions import LearningError


# ============================================================
# LearningMechanism Interface | 学习机制接口
# ============================================================

class LearningMechanism(ABC):
    """
    Pattern Regulation Mapping Ψ: Learning capability.
    
    模式调节映射 Ψ：学习能力。
    
    Adjusts selector parameters based on the discrepancy between execution results
    and expectations. Ψ: δ ⟼ γ̂, where δ = E - ε is the residual.
    
    基于执行结果与期望的差异调整选择器参数。
    Ψ: δ ⟼ γ̂，其中 δ = E - ε 是残差。
    
    Implementation Requirements | 实现要求:
        - compute_adjustment() MUST be implemented | 必须实现 compute_adjustment() 方法
        - should_learn() and get_stats() are optional | should_learn() 和 get_stats() 是可选的
    
    Learning Paradigms | 学习范式:
        - Supervised learning: Uses difference between expected and actual
          监督学习：使用期望与实际之间的差异
        - Reinforcement learning: Uses reward signals
          强化学习：使用奖励信号
        - Unsupervised learning: Discovers patterns in data
          无监督学习：发现数据中的模式
        - Meta-learning: Learns how to learn
          元学习：学习如何学习
    
    Example | 示例:
        >>> class GradientDescentLearning(LearningMechanism):
        ...     '''Gradient descent learning: updates using error gradients'''
        ...     
        ...     def __init__(self, learning_rate: float = 0.01):
        ...         self.lr = learning_rate
        ...     
        ...     def compute_adjustment(self, expected, actual, context):
        ...         # Get current parameters
        ...         params = context.get('parameters', {})
        ...         features = context.get('features', np.array([1.0]))
        ...         
        ...         # Calculate error
        ...         if isinstance(expected, (int, float)):
        ...             error = expected - actual
        ...         else:
        ...             error = 1.0 if expected == actual else -1.0
        ...         
        ...         # Calculate gradients
        ...         adjustments = {}
        ...         for key, value in params.items():
        ...             if key.startswith('w_'):
        ...                 feat_idx = int(key.split('_')[1])
        ...                 grad = -2 * error * features[feat_idx % len(features)]
        ...                 adjustments[key] = self.lr * grad
        ...         
        ...         return adjustments
    """
    
    @abstractmethod
    def compute_adjustment(self, 
                          expected: Any, 
                          actual: Any,
                          context: Dict[str, Any]) -> Optional[ParameterDict]:
        """
        Compute parameter adjustment amount.
        
        计算参数调整量。
        
        This is the core method of the learning mechanism. Based on the discrepancy
        between expected and actual results, it calculates how much the selector
        parameters should be adjusted.
        
        这是学习机制的核心方法。基于期望与实际结果的差异，
        计算选择器参数应调整多少。
        
        Args | 参数:
            expected: Expected result | 期望结果
            actual: Actual result | 实际结果
            context: Context information including:
                     上下文信息，包括：
                - features: Current feature vector | 当前特征向量
                - behavior_name: Name of executed behavior | 执行的行为名称
                - parameters: Current selector parameters | 当前选择器参数
                - learning_rate: Learning rate (optional) | 学习率（可选）
                - timestamp: Timestamp | 时间戳
                - reward: Reward value (reinforcement learning) | 奖励值（强化学习）
                - metadata: Other metadata | 其他元数据
                
        Returns | 返回:
            Optional[ParameterDict]: Parameter adjustment dictionary, None if no adjustment needed
                                    参数调整字典，不需要调整时返回 None
                                    
        Raises | 抛出:
            LearningError: When learning process encounters an error | 学习过程出错时
        """
        pass
    
    def should_learn(self, expected: Any, actual: Any) -> bool:
        """
        Determine whether learning should occur (optional implementation).
        
        判断是否应该进行学习（可选实现）。
        
        Can be used for conditional learning, such as only learning when error is large.
        
        可用于条件学习，例如仅在误差较大时学习。
        
        Args | 参数:
            expected: Expected result | 期望结果
            actual: Actual result | 实际结果
            
        Returns | 返回:
            bool: Whether learning should occur, default returns True
                 是否应该进行学习，默认返回 True
        """
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get learning mechanism statistics (optional implementation).
        
        获取学习机制统计信息（可选实现）。
        
        Returns | 返回:
            Dict: Statistics including learning count, average error, etc.
                 包含学习次数、平均误差等的统计字典
        """
        return {
            'name': self.__class__.__name__,
            'type': 'learning_mechanism'
        }


# ============================================================
# Export Interface | 导出接口
# ============================================================

__all__ = [
    'LearningMechanism',  # Learning mechanism interface | 学习机制接口
    'LearningError',      # Learning exception | 学习异常
]