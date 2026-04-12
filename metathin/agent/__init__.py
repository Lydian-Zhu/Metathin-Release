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
    >>> from metathin.core import SimplePatternSpace
    >>> from metathin.components import FunctionBehavior
    >>> 
    >>> # Method 1: Direct construction | 方法1：直接构造
    >>> agent = Metathin(pattern_space=SimplePatternSpace(lambda x: [len(x)]))
    >>> agent.register_behavior(FunctionBehavior("greet", lambda f,**k: "Hello"))
    >>> 
    >>> # Method 2: Builder pattern | 方法2：构建器模式
    >>> agent = (MetathinBuilder()
    ...     .with_name("MyAgent")
    ...     .with_pattern_space(SimplePatternSpace(lambda x: [len(x)]))
    ...     .with_behavior(FunctionBehavior("greet", lambda f,**k: "Hello"))
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