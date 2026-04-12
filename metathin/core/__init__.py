"""
Metathin Core Package - Quintuple Interface Definitions
=========================================================

Metathin 核心包 - 五元组接口定义

This package contains the fundamental interfaces for the quintuple (P, B, S, D, Ψ)
and supporting type definitions. These interfaces are the foundation of the entire
framework - all custom components must inherit from these interfaces.

本包包含五元组 (P, B, S, D, Ψ) 的基础接口和支持类型定义。
这些接口是整个框架的基础——所有自定义组件都必须继承这些接口。

Module Structure | 模块结构:
    - types.py: Type aliases (FeatureVector, FitnessScore, ParameterDict)
               类型别名（特征向量、适应度分数、参数字典）
    - exceptions.py: Unified exception hierarchy
                     统一异常体系
    - p_pattern.py: PatternSpace interface (Perception layer | 感知层)
    - b_behavior.py: MetaBehavior interface (Action layer | 行动层)
    - s_selector.py: Selector interface (Evaluation layer | 评估层)
    - d_decision.py: DecisionStrategy interface (Decision layer | 决策层)
    - psi_learning.py: LearningMechanism interface (Learning layer | 学习层)
    - memory_backend.py: MemoryBackend interface (Storage backend | 存储后端)

Design Philosophy | 设计理念:
    - Fixed interfaces, free implementation | 固定接口，自由实现
    - Type safety throughout | 全程类型安全
    - Clear separation of concerns | 清晰的关注点分离
"""

# ============================================================
# Type Definitions | 类型定义
# ============================================================

from .types import (
    T,                  # Generic input type | 泛型输入类型
    R,                  # Generic result type | 泛型结果类型
    FeatureVector,      # Feature vector type | 特征向量类型
    FitnessScore,       # Fitness score type | 适应度分数类型
    ParameterDict,      # Parameter dictionary type | 参数字典类型
)

# ============================================================
# Exception Definitions | 异常定义
# ============================================================

from .exceptions import (
    # Base exception | 基础异常
    MetathinError,
    
    # Perception layer (P) | 感知层
    PatternExtractionError,
    
    # Action layer (B) | 行动层
    BehaviorExecutionError,
    
    # Evaluation layer (S) | 评估层
    FitnessComputationError,
    
    # Decision layer (D) | 决策层
    DecisionError,
    NoBehaviorError,
    
    # Learning layer (Ψ) | 学习层
    LearningError,
    ParameterUpdateError,
)

# ============================================================
# Quintuple Interfaces | 五元组接口
# ============================================================

# P - Perception | 感知
from .p_pattern import PatternSpace

# B - Action | 行动
from .b_behavior import MetaBehavior

# S - Evaluation | 评估
from .s_selector import Selector

# D - Decision | 决策
from .d_decision import DecisionStrategy

# Ψ - Learning | 学习
from .psi_learning import LearningMechanism

# ============================================================
# Auxiliary Interfaces | 辅助接口
# ============================================================

# Memory storage backend | 记忆存储后端
from .memory_backend import MemoryBackend


# ============================================================
# Export Interface | 导出接口
# ============================================================

__all__ = [
    # Types | 类型
    'T',
    'R',
    'FeatureVector',
    'FitnessScore',
    'ParameterDict',
    
    # Exceptions | 异常
    'MetathinError',
    'PatternExtractionError',
    'BehaviorExecutionError',
    'FitnessComputationError',
    'DecisionError',
    'NoBehaviorError',
    'LearningError',
    'ParameterUpdateError',
    
    # Quintuple Interfaces (P, B, S, D, Ψ) | 五元组接口
    'PatternSpace',      # P - Perception | 感知
    'MetaBehavior',      # B - Action | 行动
    'Selector',          # S - Evaluation | 评估
    'DecisionStrategy',  # D - Decision | 决策
    'LearningMechanism', # Ψ - Learning | 学习
    
    # Auxiliary Interfaces | 辅助接口
    'MemoryBackend',     # Storage backend | 存储后端
]


# ============================================================
# Package Documentation | 包文档
# ============================================================

"""
Usage Example | 使用示例:

>>> from metathin.core import (
...     PatternSpace, MetaBehavior, Selector,
...     DecisionStrategy, LearningMechanism,
...     FeatureVector, FitnessScore
... )
>>> 
>>> # Implement custom pattern space | 实现自定义模式空间
>>> class MyPattern(PatternSpace[str]):
...     def extract(self, raw_input: str) -> FeatureVector:
...         return np.array([len(raw_input)], dtype=np.float64)
>>> 
>>> # Implement custom behavior | 实现自定义行为
>>> class MyBehavior(MetaBehavior[str]):
...     @property
...     def name(self) -> str:
...         return "my_behavior"
...     
...     def execute(self, features: FeatureVector, **kwargs) -> str:
...         return f"Feature length: {features[0]}"
>>> 
>>> # Implement custom selector | 实现自定义选择器
>>> class MySelector(Selector):
...     def compute_fitness(self, behavior: MetaBehavior, 
...                        features: FeatureVector) -> FitnessScore:
...         return 0.5  # Always 0.5 | 总是 0.5
"""