"""
Metathin Agent Layer - Main Entry Point
========================================

Metathin 代理层 - 主入口点

This package contains the main agent classes for the Metathin framework.
Users primarily interact with the Metathin class and MetathinBuilder.

本包包含 Metathin 框架的主要代理类。
用户主要与 Metathin 类和 MetathinBuilder 交互。

Modules | 模块:
    - metathin.py: Main Metathin agent facade | 主 Metathin 代理门面
    - builder.py: Fluent builder for constructing agents | 用于构建代理的流畅构建器

Example | 示例:
    >>> from metathin.agent import Metathin, MetathinBuilder
    >>> 
    >>> # Implement P, S and D yourself | 自行实现 P、S、D
    >>> # Method 1: Direct construction | 方法1：直接构造
    >>> agent = Metathin(
    ...     pattern_space=MyPattern(),
    ...     selector=MySelector(),
    ...     decision_strategy=MyStrategy(),
    ... )
    >>> agent.register_behavior(my_behavior)
    >>> 
    >>> # Method 2: Builder pattern | 方法2：构建器模式
    >>> agent = (MetathinBuilder()
    ...     .with_name("MyAgent")
    ...     .with_pattern_space(MyPattern())
    ...     .with_behavior(my_behavior)
    ...     .with_selector(MySelector())
    ...     .with_decision_strategy(MyStrategy())
    ...     .build()
    ... )
    >>> 
    >>> result = agent.think("world")
"""

from .metathin import Metathin
from .builder import MetathinBuilder


__all__ = [
    'Metathin',         # Main agent facade | 主代理门面
    'MetathinBuilder',  # Fluent builder | 流畅构建器
]