"""
Metathin Ecosystem Installation Configuration
=============================================

This package includes:
- metathin: Core cognitive agent framework
- metathin_plus.chaos: Chaos theory and time series forecasting
- metathin_plus.sci: Scientific discovery and pattern extraction

Author: Lydian-Zhu
Contact: 1799824258@qq.com
GitHub: https://github.com/Lydian-Zhu/Metathin-Release
"""

from setuptools import setup, find_packages
import os

# ============================================================
# Package Metadata
# ============================================================

__version__ = "0.3.0"
__author__ = "Lydian-Zhu"
__license__ = "MIT"
__contact__ = "1799824258@qq.com"

# Safely read README for PyPI description
try:
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "Metathin - A meta-cognitive agent framework for building intelligent systems"

# ============================================================
# Dependency Groups
# ============================================================

# Core dependencies (required for all installations)
core_requires = [
    "numpy>=1.19.0",
]

# Chaos module dependencies (nonlinear dynamics, forecasting)
chaos_requires = [
    "scipy>=1.5.0",
    "scikit-learn>=0.24.0",
]

# Scientific discovery module dependencies
sci_requires = [
    "scipy>=1.5.0",              # Curve fitting, optimization
    "scikit-learn>=0.24.0",       # Feature extraction, preprocessing
    "matplotlib>=3.3.0",          # Report visualization
    "reportlab>=3.6.0",           # PDF report generation (optional)
]

# Deep learning dependencies
deep_requires = [
    "torch>=1.9.0",
]

# Visualization dependencies
viz_requires = [
    "matplotlib>=3.3.0",
    "seaborn>=0.11.0",
]

# Data processing dependencies
data_requires = [
    "pandas>=1.0.0",
    "openpyxl>=3.0.0",
]

# Development dependencies (for contributors)
dev_requires = [
    "pytest>=6.0.0",
    "pytest-cov>=2.0.0",
    "black>=21.0.0",
    "flake8>=3.9.0",
    "mypy>=0.900",
    "isort>=5.0.0",
]

# Documentation dependencies
docs_requires = [
    "sphinx>=4.0.0",
    "sphinx-rtd-theme>=1.0.0",
]

# ============================================================
# Optional Dependency Groups
# ============================================================

extras = {
    # Core framework
    "core": core_requires,
    
    # Chaos module
    "chaos": chaos_requires,
    
    # Scientific discovery module
    "sci": sci_requires,
    
    # Deep learning support
    "deep": deep_requires,
    
    # Visualization tools
    "viz": viz_requires,
    
    # Data processing
    "data": data_requires,
    
    # Documentation tools
    "docs": docs_requires,
    
    # Development environment
    "dev": dev_requires + docs_requires,
    
    # Full installation (all modules)
    "full": (core_requires + 
             chaos_requires + 
             sci_requires + 
             viz_requires + 
             data_requires),
}

# All-in-one for contributors
extras["all"] = extras["full"] + dev_requires + docs_requires

# ============================================================
# Main Setup Configuration
# ============================================================

setup(
    # Package identity
    name="metathin",
    version=__version__,
    author=__author__,
    author_email=__contact__,
    maintainer=__author__,
    maintainer_email=__contact__,
    
    # Description
    description="Metathin - A meta-cognitive agent framework for building intelligent systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    # Project URLs
    url="https://github.com/Lydian-Zhu/Metathin-Release",
    project_urls={
        "Documentation": "https://github.com/Lydian-Zhu/Metathin-Release#readme",
        "Source": "https://github.com/Lydian-Zhu/Metathin-Release",
        "Bug Reports": "https://github.com/Lydian-Zhu/Metathin-Release/issues",
        "Examples": "https://github.com/Lydian-Zhu/Metathin-Release/tree/main/examples",
        "Author's Email": f"mailto:{__contact__}",
    },
    
    # License
    license=__license__,
    classifiers=[
        # Development status
        "Development Status :: 3 - Alpha",
        
        # Intended audience
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        
        # Topics
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        
        # License
        f"License :: OSI Approved :: {__license__} License",
        
        # Python versions
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        
        # Operating systems
        "Operating System :: OS Independent",
    ],
    
    # Keywords for PyPI search
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
    
    # Package discovery
    packages=find_packages(include=[
        'metathin',
        'metathin.*',
        'metathin_plus',
        'metathin_plus.*',
    ]),
    
    # Include package data
    include_package_data=True,
    package_data={
        "metathin": ["py.typed"],
        "metathin_plus": ["py.typed"],
        "metathin_plus.sci.memory.pretrained": ["*.json"],
    },
    zip_safe=False,
    
    # Python version requirement
    python_requires=">=3.7",
    
    # Core dependencies
    install_requires=core_requires,
    
    # Optional dependencies
    extras_require=extras,
    
    # PyPI classifiers for search
    # These are automatically generated from classifiers above
)