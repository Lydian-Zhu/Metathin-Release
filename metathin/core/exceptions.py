"""
Core Exception Definitions | 核心异常定义
=========================================

Defines the unified exception hierarchy for the Metathin framework.
All custom exceptions inherit from MetathinError for consistent error handling.

定义 Metathin 框架的统一异常体系。
所有自定义异常都继承自 MetathinError，实现一致的错误处理。

Exception Hierarchy | 异常层级:
    MetathinError (base)
    ├── PatternExtractionError    (P - Perception layer)
    ├── BehaviorExecutionError    (B - Action layer)
    ├── FitnessComputationError   (S - Evaluation layer)
    ├── DecisionError             (D - Decision layer)
    │   └── NoBehaviorError
    └── LearningError             (Ψ - Learning layer)
        └── ParameterUpdateError
"""

# ============================================================
# Base Exception | 基础异常
# ============================================================

class MetathinError(Exception):
    """
    Metathin base exception class.
    
    Metathin 基础异常类。
    
    All custom exceptions should inherit from this class. This allows users
    to catch all framework-specific exceptions with a single except clause
    while still having access to specific exception types when needed.
    
    所有自定义异常都应继承此类。这允许用户使用单个 except 子句捕获所有
    框架特定异常，同时在需要时仍能访问特定异常类型。
    
    Example | 示例:
        >>> try:
        ...     agent.think(input)
        ... except MetathinError as e:
        ...     print(f"Framework error: {e}")
    """
    pass


# ============================================================
# Perception Layer Exceptions (P) | 感知层异常
# ============================================================

class PatternExtractionError(MetathinError):
    """
    Pattern extraction error.
    
    模式提取错误。
    
    Raised when the pattern space cannot extract features from input.
    
    当模式空间无法从输入提取特征时抛出。
    
    Possible causes | 可能原因:
        - Invalid input format | 输入格式无效
        - Input contains invalid values (NaN, Inf) | 输入包含无效值
        - Feature extraction algorithm failed | 特征提取算法失败
        - Empty or malformed data | 数据为空或格式错误
    
    Example | 示例:
        >>> pattern = MyPatternSpace()
        >>> try:
        ...     features = pattern.extract(invalid_data)
        ... except PatternExtractionError as e:
        ...     print(f"Feature extraction failed: {e}")
    """
    pass


# ============================================================
# Action Layer Exceptions (B) | 行动层异常
# ============================================================

class BehaviorExecutionError(MetathinError):
    """
    Behavior execution error.
    
    行为执行错误。
    
    Raised when an exception occurs during behavior execution.
    
    当行为执行过程中发生异常时抛出。
    
    Possible causes | 可能原因:
        - Logic error in behavior implementation | 行为实现中的逻辑错误
        - Resources unavailable | 资源不可用
        - Timeout | 执行超时
        - External dependency failure | 外部依赖失败
    
    Example | 示例:
        >>> behavior = MyBehavior()
        >>> try:
        ...     result = behavior.execute(features)
        ... except BehaviorExecutionError as e:
        ...     print(f"Behavior failed: {e}")
    """
    pass


# ============================================================
# Evaluation Layer Exceptions (S) | 评估层异常
# ============================================================

class FitnessComputationError(MetathinError):
    """
    Fitness computation error.
    
    适应度计算错误。
    
    Raised when the selector cannot compute fitness scores.
    
    当选择器无法计算适应度分数时抛出。
    
    Possible causes | 可能原因:
        - Feature vector dimension mismatch | 特征向量维度不匹配
        - Numerical errors during computation | 计算过程中的数值错误
        - Behavior index out of bounds | 行为索引越界
        - Invalid parameter values | 参数值无效
    
    Example | 示例:
        >>> selector = MySelector()
        >>> try:
        ...     fitness = selector.compute_fitness(behavior, features)
        ... except FitnessComputationError as e:
        ...     print(f"Fitness computation failed: {e}")
    """
    pass


# ============================================================
# Decision Layer Exceptions (D) | 决策层异常
# ============================================================

class DecisionError(MetathinError):
    """
    Decision error.
    
    决策错误。
    
    Raised when the decision strategy cannot make a selection.
    
    当决策策略无法做出选择时抛出。
    
    Possible causes | 可能原因:
        - Empty behavior list | 行为列表为空
        - Invalid fitness scores (all NaN/Inf) | 无效的适应度分数
        - Strategy internal error | 策略内部错误
    
    Example | 示例:
        >>> strategy = MyStrategy()
        >>> try:
        ...     selected = strategy.select(behaviors, scores, features)
        ... except DecisionError as e:
        ...     print(f"Decision failed: {e}")
    """
    pass


class NoBehaviorError(DecisionError):
    """
    No available behavior error.
    
    无可用行为错误。
    
    Raised when no behaviors are suitable for the current context.
    
    当没有适合当前上下文的行为时抛出。
    
    This typically occurs when | 通常发生在以下情况:
        - No behaviors are registered | 没有注册任何行为
        - All behaviors' can_execute() returns False | 所有行为都返回 False
        - All behaviors' fitness scores are below threshold | 所有适应度低于阈值
    
    Example | 示例:
        >>> agent = Metathin(pattern_space=MyPattern())
        >>> # No behaviors registered | 没有注册行为
        >>> try:
        ...     result = agent.think(input)
        ... except NoBehaviorError as e:
        ...     print(f"No behaviors available: {e}")
    """
    pass


# ============================================================
# Learning Layer Exceptions (Ψ) | 学习层异常
# ============================================================

class LearningError(MetathinError):
    """
    Learning error.
    
    学习错误。
    
    Raised when the learning mechanism fails to update parameters.
    
    当学习机制无法更新参数时抛出。
    
    Possible causes | 可能原因:
        - Incorrect parameter format | 参数格式错误
        - Improper learning rate settings | 学习率设置不当
        - Numerical instability (gradient explosion/vanishing) | 数值不稳定
    
    Example | 示例:
        >>> learner = MyLearner()
        >>> try:
        ...     adjustment = learner.compute_adjustment(expected, actual, context)
        ... except LearningError as e:
        ...     print(f"Learning failed: {e}")
    """
    pass


class ParameterUpdateError(LearningError):
    """
    Parameter update error.
    
    参数更新错误。
    
    Raised when updating selector parameters fails.
    
    当更新选择器参数失败时抛出。
    
    This is a specialization of LearningError for the parameter update phase.
    
    这是 LearningError 在参数更新阶段的特化。
    
    Possible causes | 可能原因:
        - Parameter key doesn't exist | 参数键不存在
        - Parameter value type mismatch | 参数值类型不匹配
        - Parameter value out of valid range | 参数值超出有效范围
    
    Example | 示例:
        >>> selector = MySelector()
        >>> try:
        ...     selector.update_parameters(invalid_delta)
        ... except ParameterUpdateError as e:
        ...     print(f"Parameter update failed: {e}")
    """
    pass


# ============================================================
# Export Interface | 导出接口
# ============================================================

__all__ = [
    # Base exception | 基础异常
    'MetathinError',
    
    # Perception layer (P) | 感知层
    'PatternExtractionError',
    
    # Action layer (B) | 行动层
    'BehaviorExecutionError',
    
    # Evaluation layer (S) | 评估层
    'FitnessComputationError',
    
    # Decision layer (D) | 决策层
    'DecisionError',
    'NoBehaviorError',
    
    # Learning layer (Ψ) | 学习层
    'LearningError',
    'ParameterUpdateError',
]