"""
Metathin Ecosystem Installation Configuration
=============================================

Metathin 生态系统安装配置

This package includes:
    - metathin: Core cognitive agent framework 
    - metathin_plus.chaos: Chaos theory and time series forecasting
    - metathin_plus.sci: Scientific discovery and pattern extraction 

本包包含：
    - metathin：核心认知代理框架
    - metathin_plus.chaos：混沌理论与时间序列预测
    - metathin_plus.sci：科学发现与模式提取

Author | 作者: Lydian-Zhu
Contact | 联系方式: 1799824258@qq.com
GitHub: https://github.com/Lydian-Zhu/Metathin-Release
License | 许可证: MIT
Version | 版本: 0.4.0 (重构版 | Refactored)
"""

from setuptools import setup, find_packages
import os

# ============================================================
# Package Metadata | 包元数据
# ============================================================

__version__ = "0.4.0"
__author__ = "Lydian-Zhu"
__license__ = "MIT"
__contact__ = "1799824258@qq.com"

# Safely read README for PyPI description | 安全读取 README 作为 PyPI 描述
try:
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "Metathin - A meta-cognitive agent framework for building intelligent systems"


# ============================================================
# Dependency Groups | 依赖组
# ============================================================

# Core dependencies (required for all installations)
# 核心依赖（所有安装都需要）
core_requires = [
    "numpy>=1.19.0",           # Fundamental package for numerical computing | 数值计算基础包
]

# Chaos module dependencies | 混沌模块依赖
chaos_requires = [
    "scipy>=1.5.0",            # Scientific computing | 科学计算
    "scikit-learn>=0.24.0",    # Machine learning tools | 机器学习工具
]

# Scientific discovery module dependencies | 科学发现模块依赖
sci_requires = [
    "scipy>=1.5.0",            # Curve fitting, optimization | 曲线拟合、优化
    "scikit-learn>=0.24.0",    # Feature extraction, preprocessing | 特征提取、预处理
    "matplotlib>=3.3.0",       # Report visualization | 报告可视化
    "reportlab>=3.6.0",        # PDF report generation (optional) | PDF 报告生成（可选）
]

# Deep learning dependencies | 深度学习依赖
deep_requires = [
    "torch>=1.9.0",            # PyTorch deep learning framework | PyTorch 深度学习框架
]

# Visualization dependencies | 可视化依赖
viz_requires = [
    "matplotlib>=3.3.0",       # Core plotting library | 核心绘图库
    "seaborn>=0.11.0",         # Statistical data visualization | 统计数据可视化
]

# Data processing dependencies | 数据处理依赖
data_requires = [
    "pandas>=1.0.0",           # Data manipulation and analysis | 数据处理与分析
    "openpyxl>=3.0.0",         # Excel file support | Excel 文件支持
]

# Testing dependencies | 测试依赖
test_requires = [
    "pytest>=6.0.0",           # Testing framework | 测试框架
    "pytest-cov>=2.0.0",       # Test coverage reporting | 测试覆盖率报告
    "pytest-xdist>=2.0.0",     # Parallel test execution | 并行测试执行
]

# Development dependencies (for contributors) | 开发依赖（为贡献者准备）
dev_requires = [
    "pytest>=6.0.0",
    "pytest-cov>=2.0.0",
    "pytest-xdist>=2.0.0",
    "black>=21.0.0",           # Code formatter | 代码格式化工具
    "flake8>=3.9.0",           # Code linter | 代码检查工具
    "mypy>=0.900",             # Static type checking | 静态类型检查
    "isort>=5.0.0",            # Import sorter | 导入排序工具
]

# Documentation dependencies | 文档依赖
docs_requires = [
    "sphinx>=4.0.0",           # Documentation generator | 文档生成器
    "sphinx-rtd-theme>=1.0.0", # ReadTheDocs theme | ReadTheDocs 主题
]

# YAML support (for config files) | YAML 支持（用于配置文件）
yaml_requires = [
    "pyyaml>=5.1",             # YAML parser for config files | 配置文件的 YAML 解析器
]


# ============================================================
# Optional Dependency Groups | 可选依赖组
# ============================================================

extras = {
    # Core framework | 核心框架
    "core": core_requires,
    
    # Chaos module | 混沌模块
    "chaos": chaos_requires,
    
    # Scientific discovery module | 科学发现模块
    "sci": sci_requires,
    
    # Deep learning support | 深度学习支持
    "deep": deep_requires,
    
    # Visualization tools | 可视化工具
    "viz": viz_requires,
    
    # Data processing | 数据处理
    "data": data_requires,
    
    # Testing tools | 测试工具
    "test": test_requires,
    
    # YAML configuration support | YAML 配置支持
    "yaml": yaml_requires,
    
    # Documentation tools | 文档工具
    "docs": docs_requires,
    
    # Development environment (all tools for contributors)
    # 开发环境（为贡献者准备的所有工具）
    "dev": dev_requires + docs_requires + yaml_requires,
    
    # Full installation (all modules)
    # 完整安装（所有模块）
    "full": (core_requires + 
             chaos_requires + 
             sci_requires + 
             viz_requires + 
             data_requires + 
             yaml_requires),
}

# All-in-one for contributors (includes everything)
# 贡献者全量安装（包含所有）
extras["all"] = extras["full"] + dev_requires + docs_requires


# ============================================================
# Package Discovery Configuration | 包发现配置
# ============================================================

# Define packages to include | 定义要包含的包
packages = find_packages(
    include=[
        'metathin',
        'metathin.*',
        'metathin_plus',
        'metathin_plus.*',
    ],
    exclude=[
        'tests',
        'tests.*',
        'docs',
        'docs.*',
        'examples',
        'examples.*',
    ]
)


# ============================================================
# Main Setup Configuration | 主安装配置
# ============================================================

setup(
    # ===== Package Identity | 包标识 =====
    name="metathin",
    version=__version__,
    author=__author__,
    author_email=__contact__,
    maintainer=__author__,
    maintainer_email=__contact__,
    
    # ===== Description | 描述 =====
    description="Metathin - A meta-cognitive agent framework for building intelligent systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    # ===== Project URLs | 项目链接 =====
    url="https://github.com/Lydian-Zhu/Metathin-Release",
    project_urls={
        "Documentation | 文档": "https://github.com/Lydian-Zhu/Metathin-Release#readme",
        "Source | 源码": "https://github.com/Lydian-Zhu/Metathin-Release",
        "Bug Reports | 问题反馈": "https://github.com/Lydian-Zhu/Metathin-Release/issues",
        "Author's Email | 作者邮箱": f"mailto:{__contact__}",
    },
    
    # ===== License | 许可证 =====
    license=__license__,
    classifiers=[
        # Development status | 开发状态
        "Development Status :: 3 - Alpha",
        
        # Intended audience | 目标用户
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        
        # Topics | 主题
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        
        # License | 许可证
        f"License :: OSI Approved :: {__license__} License",
        
        # Python versions | Python 版本
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        
        # Operating systems | 操作系统
        "Operating System :: OS Independent",
    ],
    
    # ===== Keywords for PyPI search | PyPI 搜索关键词 =====
    keywords=" ".join([
        "ai", "artificial-intelligence", "cognitive-architecture",
        "agent", "agent-framework", "machine-learning",
        "metacognition", "cognitive-science",
        "chaos-theory", "nonlinear-dynamics", "time-series",
        "forecasting", "prediction",
        "scientific-discovery", "symbolic-regression",
        "pytorch", "scikit-learn", "scientific-computing",
        "physics", "mathematics", "research"
    ]),
    
    # ===== Package Discovery | 包发现 =====
    packages=packages,
    
    # ===== Include Package Data | 包含包数据 =====
    include_package_data=True,
    package_data={
        "metathin": ["py.typed"],
        "metathin_plus": ["py.typed"],
        "metathin_plus.sci.memory.pretrained": ["*.json"],
    },
    zip_safe=False,
    
    # ===== Python Version Requirement | Python 版本要求 =====
    python_requires=">=3.8",
    
    # ===== Core Dependencies | 核心依赖 =====
    install_requires=core_requires,
    
    # ===== Optional Dependencies | 可选依赖 =====
    extras_require=extras,
    
    # ===== Entry Points | 入口点 =====
    entry_points={
        "console_scripts": [
            # CLI tools (if any) | 命令行工具（如果有）
        ],
        "metathin.plugins": [
            # Plugin discovery | 插件发现
        ],
    },
)


# ============================================================
# Installation Instructions | 安装说明
# ============================================================

"""
Installation Examples | 安装示例:

    # Core only | 仅核心
    pip install metathin
    
    # With chaos module | 带混沌模块
    pip install metathin[chaos]
    
    # With scientific discovery | 带科学发现模块
    pip install metathin[sci]
    
    # With YAML config support | 带 YAML 配置支持
    pip install metathin[yaml]
    
    # With testing tools | 带测试工具
    pip install metathin[test]
    
    # Full installation | 完整安装
    pip install metathin[full]
    
    # For contributors | 为贡献者准备
    pip install metathin[all]
    
    # Development mode | 开发模式
    pip install -e .

After installation, verify with | 安装后验证:
    >>> import metathin
    >>> print(metathin.__version__)
    0.4.0
"""