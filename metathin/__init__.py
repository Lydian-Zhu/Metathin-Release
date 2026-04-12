"""
Metathin - Meta-Cognitive Agent System Construction Framework
===============================================================

Metathin 是一个基于五元组结构 (P, B, S, D, Ψ) 构建认知代理的框架。

Metathin is a framework for building cognitive agents based on the quintuple
structure (P, B, S, D, Ψ).

Core Components | 核心组件:
    - PatternSpace (P): 模式空间，将输入转换为特征向量 (感知层)
      Pattern space, converts input to feature vectors (perception layer)
    - MetaBehavior (B): 元行为，可执行的技能单元 (行动层)
      Meta-behavior, executable skill units (action layer)
    - Selector (S): 选择器，评估行为适用性 (评估层)
      Selector, evaluates behavior suitability (evaluation layer)
    - DecisionStrategy (D): 决策策略，选择最优行为 (决策层)
      Decision strategy, selects optimal behavior (decision layer)
    - LearningMechanism (Ψ): 学习机制，根据反馈调整参数 (学习层)
      Learning mechanism, adjusts based on feedback (learning layer)

Design Philosophy | 设计理念:
    - Fixed interfaces, free implementation | 固定接口，自由实现
    - Modular and composable | 模块化与可组合
    - Type safe and observable | 类型安全与可观测

Version | 版本: 0.4.0 (重构版 | Refactored)
Author | 作者: Lydian-Zhu
License | 许可证: MIT
"""

# ============================================================
# Version Information | 版本信息
# ============================================================

__version__ = '0.4.0'
__author__ = 'Lydian-Zhu'
__license__ = 'MIT'

# ============================================================
# Core Exports (Backward Compatible) | 核心导出（向后兼容）
# ============================================================

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


# ============================================================
# Lazy Imports for Components (Performance) | 组件延迟导入（性能优化）
# ============================================================

# These are imported lazily to avoid circular imports and improve startup time
# 延迟导入以避免循环依赖并提高启动时间

_COMPONENTS_MODULES = {
    # Pattern Space Components | 模式空间组件
    'SimplePatternSpace': 'metathin.components.pattern_space',
    'StatisticalPatternSpace': 'metathin.components.pattern_space',
    'NormalizedPatternSpace': 'metathin.components.pattern_space',
    'CompositePatternSpace': 'metathin.components.pattern_space',
    'CachedPatternSpace': 'metathin.components.pattern_space',
    
    # Behavior Components | 行为组件
    'FunctionBehavior': 'metathin.components.behavior_library',
    'LambdaBehavior': 'metathin.components.behavior_library',
    'CompositeBehavior': 'metathin.components.behavior_library',
    'RetryBehavior': 'metathin.components.behavior_library',
    'TimeoutBehavior': 'metathin.components.behavior_library',
    'ConditionalBehavior': 'metathin.components.behavior_library',
    'CachedBehavior': 'metathin.components.behavior_library',
    
    # Selector Components | 选择器组件
    'SimpleSelector': 'metathin.components.selector',
    'PolynomialSelector': 'metathin.components.selector',
    'RuleBasedSelector': 'metathin.components.selector',
    'EnsembleSelector': 'metathin.components.selector',
    'AdaptiveSelector': 'metathin.components.selector',
    
    # Decision Strategy Components | 决策策略组件
    'MaxFitnessStrategy': 'metathin.components.decision',
    'ProbabilisticStrategy': 'metathin.components.decision',
    'EpsilonGreedyStrategy': 'metathin.components.decision',
    'RoundRobinStrategy': 'metathin.components.decision',
    'RandomStrategy': 'metathin.components.decision',
    'BoltzmannStrategy': 'metathin.components.decision',
    'HybridStrategy': 'metathin.components.decision',
    
    # Learning Mechanism Components | 学习机制组件
    'GradientLearning': 'metathin.components.learning',
    'RewardLearning': 'metathin.components.learning',
    'MemoryLearning': 'metathin.components.learning',
    'HebbianLearning': 'metathin.components.learning',
    'EnsembleLearning': 'metathin.components.learning',
    'Experience': 'metathin.components.learning',
}


def __getattr__(name):
    """
    Lazy import for components.
    
    组件的延迟导入。
    
    This allows importing components without loading all of them at once,
    improving startup time and reducing memory usage.
    
    这允许在不一次性加载所有组件的情况下导入组件，
    提高启动时间并减少内存使用。
    """
    if name in _COMPONENTS_MODULES:
        import importlib
        module = importlib.import_module(_COMPONENTS_MODULES[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    """
    Custom dir() for better IDE support.
    
    自定义 dir() 以获得更好的 IDE 支持。
    """
    return sorted(list(__all__) + list(_COMPONENTS_MODULES.keys()))


# ============================================================
# Module Docstring | 模块文档字符串
# ============================================================

__doc__ = """
Metathin - Meta-Cognitive Agent System Construction Framework
==============================================================

Metathin is a framework for building cognitive agents based on the quintuple
structure (P, B, S, D, Ψ).

Quick Start | 快速开始:
    >>> from metathin import Metathin, MetathinBuilder
    >>> from metathin import SimplePatternSpace, FunctionBehavior
    >>> 
    >>> # Create agent using builder | 使用构建器创建代理
    >>> agent = (MetathinBuilder()
    ...     .with_pattern_space(SimplePatternSpace(lambda x: [len(x)]))
    ...     .with_behavior(FunctionBehavior("greet", lambda f,**k: "Hello!"))
    ...     .build())
    >>> 
    >>> # Use agent | 使用代理
    >>> result = agent.think("world")
    >>> print(result)  # "Hello!"

For more examples, see the documentation | 更多示例请查看文档:
    https://github.com/Lydian-Zhu/Metathin-Release

Architecture | 架构:
    metathin/
    ├── core/          # Core interfaces (P, B, S, D, Ψ) | 核心接口
    ├── engine/        # Thinking pipeline | 思考流水线
    ├── services/      # Optional services (memory, history, metrics) | 可选服务
    ├── config/        # Configuration system | 配置系统
    ├── agent/         # Agent facade and builder | 代理门面和构建器
    └── components/    # Built-in implementations | 内置实现
"""


# ============================================================
# Export Interface | 导出接口
# ============================================================

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
    
    # Components (lazy) | 组件（延迟）
    'SimplePatternSpace',
    'StatisticalPatternSpace',
    'NormalizedPatternSpace',
    'CompositePatternSpace',
    'CachedPatternSpace',
    'FunctionBehavior',
    'LambdaBehavior',
    'CompositeBehavior',
    'RetryBehavior',
    'TimeoutBehavior',
    'ConditionalBehavior',
    'CachedBehavior',
    'SimpleSelector',
    'PolynomialSelector',
    'RuleBasedSelector',
    'EnsembleSelector',
    'AdaptiveSelector',
    'MaxFitnessStrategy',
    'ProbabilisticStrategy',
    'EpsilonGreedyStrategy',
    'RoundRobinStrategy',
    'RandomStrategy',
    'BoltzmannStrategy',
    'HybridStrategy',
    'GradientLearning',
    'RewardLearning',
    'MemoryLearning',
    'HebbianLearning',
    'EnsembleLearning',
    'Experience',
]