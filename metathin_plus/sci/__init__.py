"""
Metathin+Sci - Scientific Discovery Module | 科学发现模块
===========================================================

Provides tools for automated scientific discovery using Laurent series vector space.
提供基于洛朗级数向量空间的自动化科学发现工具。

Module Structure | 模块结构:
    - core: Core mathematical components (independent of Metathin) | 核心数学组件
    - discovery: Scientific discovery agent (standalone) | 科学发现代理
    - adapters: Metathin adapters (optional) | Metathin 适配器

Features | 特性:
    - Function vector space via Laurent series | 基于洛朗级数的函数向量空间
    - Built-in function library (math, physics, chemistry) | 内置函数库
    - User function registration with persistence | 用户函数注册与持久化
    - Adaptive function matching and prediction | 自适应函数匹配与预测
    - Error-based automatic restart | 基于误差的自动重启

Version | 版本: 0.4.0
Author | 作者: Lydian-Zhu
License | 许可证: MIT
"""

__version__ = '0.4.0'
__author__ = 'Lydian-Zhu'
__license__ = 'MIT'

# ============================================================
# Core Module Exports | 核心模块导出
# ============================================================

from typing import List, Tuple

import numpy as np

from .core.function_space import FunctionSpace, FunctionVector, VectorSpaceConfig
from .core.function_library import FunctionLibrary, FunctionEntry
from .core.laurent_expander import LaurentExpander, create_default_expander, expand_function

# ============================================================
# Discovery Module Exports | 发现模块导出
# ============================================================

from .discovery.agent import ScientificDiscoveryAgent, DiscoveryPhase, DiscoveryResult

# ============================================================
# Adapter Exports (Optional - requires Metathin) | 适配器导出（可选）
# ============================================================

try:
    from .adapters.pattern_space import SciPatternSpace
    from .adapters.behavior import SciDiscoveryBehavior
    _METATHIN_AVAILABLE = True
except ImportError:
    _METATHIN_AVAILABLE = False
    SciPatternSpace = None
    SciDiscoveryBehavior = None


def is_metathin_available() -> bool:
    """Check if Metathin core is available for integration | 检查 Metathin 核心是否可用于集成"""
    return _METATHIN_AVAILABLE


# ============================================================
# Convenience Functions | 便捷函数
# ============================================================

def create_agent(window_size: int = 50,
                 error_threshold: float = 0.1,
                 enable_special: bool = False) -> ScientificDiscoveryAgent:
    """
    Create a scientific discovery agent with default settings | 创建默认设置的科学发现代理
    
    Args:
        window_size: Fitting window size | 拟合窗口大小
        error_threshold: Error threshold for restart | 重启误差阈值
        enable_special: Whether to enable special functions | 是否启用特殊函数
        
    Returns:
        ScientificDiscoveryAgent: Configured agent | 配置好的代理
    """
    return ScientificDiscoveryAgent(
        window_size=window_size,
        error_threshold=error_threshold,
        enable_special_functions=enable_special
    )


def quick_match(data: List[float], library: FunctionLibrary = None) -> List[Tuple[str, float]]:
    """
    Quick function matching from data | 从数据快速匹配函数
    
    Args:
        data: Time series data | 时间序列数据
        library: Function library (uses default if None) | 函数库
        
    Returns:
        List[Tuple[str, float]]: Matching results | 匹配结果
    """
    from .core.laurent_expander import create_default_expander
    
    expander = create_default_expander()
    x_vals = np.arange(len(data))
    vector = expander.expand_data(x_vals, np.array(data))
    
    if library is None:
        from .core.function_space import FunctionSpace
        space = FunctionSpace()
        library = FunctionLibrary(space)
    
    return library.match(vector, threshold=0.8, top_k=5)


# ============================================================
# Module Information | 模块信息
# ============================================================

__all__ = [
    # Core | 核心
    'FunctionSpace', 'FunctionVector', 'VectorSpaceConfig',
    'FunctionLibrary', 'FunctionEntry',
    'LaurentExpander', 'create_default_expander', 'expand_function',
    
    # Discovery | 发现
    'ScientificDiscoveryAgent', 'DiscoveryPhase', 'DiscoveryResult',
    
    # Adapters | 适配器
    'SciPatternSpace', 'SciDiscoveryBehavior',
    'is_metathin_available',
    
    # Utilities | 工具
    'create_agent', 'quick_match',
]


# ============================================================
# Module Initialization | 模块初始化
# ============================================================

print(f"✅ Metathin+Sci v{__version__} loaded successfully")
print(f"   Features: Laurent series vector space, function library, adaptive discovery")
if _METATHIN_AVAILABLE:
    print(f"   Metathin integration: available")
else:
    print(f"   Metathin integration: not available (install metathin first)")


# ============================================================
# Usage Examples | 使用示例
# ============================================================

"""
独立使用 | Standalone Usage:
    
    >>> from metathin_plus.sci import ScientificDiscoveryAgent
    >>> 
    >>> agent = ScientificDiscoveryAgent(window_size=50, error_threshold=0.1)
    >>> 
    >>> # Online learning | 在线学习
    >>> for x, y in data_stream:
    ...     result = agent.add_point(x, y)
    ...     if result.prediction:
    ...         print(f"Predicted: {result.prediction:.4f}, Matched: {result.matched_function}")
    ...     
    ...     # Validate with next point | 用下一个点验证
    ...     agent.validate(next_x, next_y)


与 Metathin 集成 | Integration with Metathin:
    
    >>> from metathin import Metathin, MetathinBuilder
    >>> from metathin_plus.sci import SciPatternSpace, SciDiscoveryBehavior
    >>> 
    >>> agent = (MetathinBuilder()
    ...     .with_pattern_space(SciPatternSpace())
    ...     .with_behavior(SciDiscoveryBehavior())
    ...     .build())
    >>> 
    >>> result = agent.think(x, y=y)  # y is actual value


注册自定义函数 | Register custom function:
    
    >>> agent.register_function("my_func", "a*exp(-b*x)*sin(c*x)", 
    ...                         parameters=['a', 'b', 'c'],
    ...                         description="Damped sine wave")
"""