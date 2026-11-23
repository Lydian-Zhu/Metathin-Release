"""
Metathin Installation Configuration
===================================

Metathin 安装配置。

The distribution contains the interface-first core only: P/B/S/D/Ψ contracts,
the thinking pipeline, configuration, and optional memory/history/metrics
services.  It does not ship concrete component implementations or domain
modules.

Author | 作者: Lydian-Zhu
Contact | 联系方式: 1799824258@qq.com
GitHub: https://github.com/Lydian-Zhu/Metathin-Release
License | 许可证: MIT
Version | 版本: 0.5.0
"""

from setuptools import find_packages, setup

__version__ = "0.5.0"
__author__ = "Lydian-Zhu"
__license__ = "MIT"
__contact__ = "1799824258@qq.com"

try:
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "Metathin - interface-first cognitive agent framework"

core_requires = [
    "numpy>=1.19.0",
]

test_requires = [
    "pytest>=6.0.0",
    "pytest-cov>=2.0.0",
]

yaml_requires = [
    "pyyaml>=5.1",
]

docs_requires = [
    "sphinx>=4.0.0",
    "sphinx-rtd-theme>=1.0.0",
]

dev_requires = test_requires + [
    "black>=21.0.0",
    "flake8>=3.9.0",
    "mypy>=0.900",
    "isort>=5.0.0",
]

extras = {
    "core": core_requires,
    "test": test_requires,
    "yaml": yaml_requires,
    "docs": docs_requires,
    "dev": dev_requires + docs_requires + yaml_requires,
    "full": core_requires + yaml_requires,
    "all": core_requires + test_requires + dev_requires + docs_requires + yaml_requires,
}

packages = find_packages(
    include=["metathin", "metathin.*"],
    exclude=["tests", "tests.*", "docs", "docs.*", "examples", "examples.*"],
)

setup(
    name="metathin",
    version=__version__,
    author=__author__,
    author_email=__contact__,
    maintainer=__author__,
    maintainer_email=__contact__,
    description="Metathin - interface-first cognitive agent framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Lydian-Zhu/Metathin-Release",
    project_urls={
        "Documentation | 文档": "https://github.com/Lydian-Zhu/Metathin-Release#readme",
        "Source | 源码": "https://github.com/Lydian-Zhu/Metathin-Release",
        "Bug Reports | 问题反馈": "https://github.com/Lydian-Zhu/Metathin-Release/issues",
        "Author's Email | 作者邮箱": f"mailto:{__contact__}",
    },
    license=__license__,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        f"License :: OSI Approved :: {__license__} License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    keywords=(
        "ai agent-framework cognitive-architecture metacognition "
        "interfaces reinforcement-learning"
    ),
    packages=packages,
    include_package_data=True,
    package_data={"metathin": ["py.typed"]},
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=core_requires,
    extras_require=extras,
    entry_points={
        "console_scripts": [],
        "metathin.plugins": [],
    },
)
