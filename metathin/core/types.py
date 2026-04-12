"""
Core Type Definitions | 核心类型定义
=====================================

Defines the fundamental type aliases used throughout the Metathin framework.
These types provide clear semantics and enhance code readability.

定义整个 Metathin 框架使用的基础类型别名。
这些类型提供了清晰的语义，增强了代码可读性。

Type Hierarchy | 类型层级:
    - T: Generic input type (any Python type) | 泛型输入类型
    - R: Generic result type (any Python type) | 泛型结果类型
    - FeatureVector: numpy.ndarray - must be 1D float64 | 特征向量
    - FitnessScore: float in [0,1] - behavior suitability | 适应度分数
    - ParameterDict: Dict[str, float] - learnable parameters | 参数字典
"""

from typing import Dict, TypeVar
import numpy as np

# ============================================================
# Generic Type Variables | 泛型类型变量
# ============================================================

T = TypeVar('T')
"""
Generic input type - can be any Python type.

泛型输入类型 - 可以是任意 Python 类型。

Examples | 示例:
    - str: Text input | 文本输入
    - np.ndarray: Image/sensor data | 图像/传感器数据
    - Dict: Structured data | 结构化数据
    - Custom class: Domain-specific objects | 领域特定对象
"""

R = TypeVar('R')
"""
Generic result type - can be any Python type.

泛型结果类型 - 可以是任意 Python 类型。

Examples | 示例:
    - str: Text response | 文本响应
    - float: Numerical prediction | 数值预测
    - Dict: Structured output | 结构化输出
    - Custom class: Domain-specific results | 领域特定结果
"""


# ============================================================
# Core Type Aliases | 核心类型别名
# ============================================================

FeatureVector = np.ndarray
"""
Feature vector type: q = (q1, q2, ..., qs)

特征向量类型：q = (q1, q2, ..., qs)

This is the output format of the pattern space and serves as the fundamental
input to all subsequent components.

这是模式空间的输出格式，是所有后续组件的基础输入。

Requirements | 要求:
    - Must be numpy.ndarray | 必须是 numpy 数组
    - dtype must be float64 | 类型必须是 float64
    - Must be 1-dimensional | 必须是一维
    - No NaN or Inf values allowed | 不允许包含 NaN 或 Inf

Example | 示例:
    >>> import numpy as np
    >>> features: FeatureVector = np.array([0.5, 0.2, 0.8], dtype=np.float64)
"""

FitnessScore = float
"""
Fitness score: α(Bi) ∈ [0, 1]

适应度分数：α(Bi) ∈ [0, 1]

Represents how suitable a behavior is in the current context.

表示在当前上下文中行为的合适程度。

Interpretation | 解释:
    - 0: Completely unsuitable (should never be selected) | 完全不合适
    - 0.5: Moderately suitable | 中等合适
    - 1: Perfectly suitable (should be selected) | 完全合适

Example | 示例:
    >>> fitness: FitnessScore = 0.85  # 85% suitable | 85% 合适
"""

ParameterDict = Dict[str, float]
"""
Parameter dictionary: Used for selector parameter adjustment

参数字典：用于选择器参数调整

This serves as the interface between the learning mechanism and the selector.

这是学习机制与选择器之间的接口。

Structure | 结构:
    - Key: Parameter name (e.g., 'w_0_1' for weight) | 参数名
    - Value: Parameter value (must be float) | 参数值

Example | 示例:
    >>> params: ParameterDict = {
    ...     'w_0_0': 0.5,   # Weight for behavior 0, feature 0
    ...     'w_0_1': -0.2,  # Weight for behavior 0, feature 1
    ...     'b_0': 0.1      # Bias for behavior 0
    ... }
"""


# ============================================================
# Export Interface | 导出接口
# ============================================================

__all__ = [
    'T',                # Generic input type | 泛型输入类型
    'R',                # Generic result type | 泛型结果类型
    'FeatureVector',    # Feature vector type | 特征向量类型
    'FitnessScore',     # Fitness score type | 适应度分数类型
    'ParameterDict',    # Parameter dictionary type | 参数字典类型
]