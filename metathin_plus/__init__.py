"""
Metathin+ - Domain Extensions
==============================

Optional domain modules built on top of the Metathin core framework.

可选领域扩展模块，构建在 Metathin 核心框架之上。

Modules | 模块:
    - metathin_plus.sci: Scientific discovery | 科学发现
      Requires the optional ``sci`` dependencies: ``pip install "metathin[sci]"``

Importing this package is intentionally lightweight; it does not pull in any
optional dependency. Import the domain module explicitly, for example::

    from metathin_plus.sci import ScientificMetathin

导入本包不会加载任何可选依赖，请显式导入具体的领域模块。
"""

__version__ = "0.4.0"

__all__ = ["__version__"]
