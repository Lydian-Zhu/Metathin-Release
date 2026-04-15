"""
Metathin+Sci Adapters | Metathin+Sci 适配器
=============================================

Provides adapters to integrate scientific discovery with Metathin core.
提供将科学发现与 Metathin 核心集成的适配器。

Components | 组件:
    - SciPatternSpace: Pattern space adapter for time series | 时间序列模式空间适配器
    - SciDiscoveryBehavior: Behavior adapter for function discovery | 函数发现行为适配器
    - SciSelectorAdapter: Selector adapter for function matching | 函数匹配选择器适配器
"""

from .pattern_space import SciPatternSpace
from .behavior import SciDiscoveryBehavior
from .selector import SciSelectorAdapter

__all__ = [
    'SciPatternSpace',
    'SciDiscoveryBehavior', 
    'SciSelectorAdapter'
]