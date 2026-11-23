"""
Metathin - Meta-Cognitive Agent System Construction Framework
===============================================================

Metathin 是一个基于五元组结构 (P, B, S, D, Ψ) 构建认知代理的框架。

Metathin is a framework for building cognitive agents based on the quintuple
structure (P, B, S, D, Ψ).

This release ships the interfaces and the orchestration core only.
Applications provide their own P, B, S, D and Ψ implementations.

本版本只提供接口与编排核心，具体实现由使用方自行提供。

Core Components | 核心组件:
    - PatternSpace (P): 模式空间，将输入转换为特征向量 (感知层)
    - MetaBehavior (B): 元行为，可执行的技能单元 (行动层)
    - Selector (S): 选择器，评估行为适用性 (评估层)
    - DecisionStrategy (D): 决策策略，选择最优行为 (决策层)
    - LearningMechanism (Ψ): 学习机制，根据反馈调整参数 (学习层)

Version | 版本: 0.5.0 (interfaces-only)
Author | 作者: Lydian-Zhu
License | 许可证: MIT
"""

__version__ = '0.5.0'
__author__ = 'Lydian-Zhu'
__license__ = 'MIT'

# Agent classes | 代理类
from metathin.agent import Metathin, MetathinBuilder

# Core interfaces (P, B, S, D, Ψ) | 核心接口
from metathin.core import (
    # Types | 类型
    T, R,
    FeatureVector,
    FitnessScore,
    ParameterDict,

    # Exceptions | 异常
    MetathinError,
    PatternExtractionError,
    BehaviorExecutionError,
    FitnessComputationError,
    DecisionError,
    NoBehaviorError,
    LearningError,
    ParameterUpdateError,

    # Quintuple Interfaces | 五元组接口
    PatternSpace,
    MetaBehavior,
    Selector,
    DecisionStrategy,
    LearningMechanism,

    # Memory Backend | 记忆后端
    MemoryBackend,
)

# Configuration | 配置
from metathin.config import (
    MetathinConfig,
    PipelineConfig,
    MemoryConfig,
    ObservabilityConfig,
    load_config,
    save_config,
)

# Services | 服务
from metathin.services import (
    MemoryManager,
    HistoryTracker,
    ThoughtRecord,
    MetricsCollector,
    ThoughtMetrics,
    AggregatedMetrics,
)


__all__ = [
    # Version | 版本
    '__version__',
    '__author__',
    '__license__',

    # Agent | 代理
    'Metathin',
    'MetathinBuilder',

    # Types | 类型
    'T', 'R',
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

    # Core Interfaces | 核心接口
    'PatternSpace',
    'MetaBehavior',
    'Selector',
    'DecisionStrategy',
    'LearningMechanism',

    # Memory | 记忆
    'MemoryBackend',
    'MemoryManager',

    # History | 历史
    'HistoryTracker',
    'ThoughtRecord',

    # Metrics | 指标
    'MetricsCollector',
    'ThoughtMetrics',
    'AggregatedMetrics',

    # Configuration | 配置
    'MetathinConfig',
    'PipelineConfig',
    'MemoryConfig',
    'ObservabilityConfig',
    'load_config',
    'save_config',
]
