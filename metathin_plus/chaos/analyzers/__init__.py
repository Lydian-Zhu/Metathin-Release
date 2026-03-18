"""
Chaos Analyzer Module
=====================================================

This module provides analytical tools for characterizing chaotic systems, used to quantify
the degree of chaos and system complexity. These analyzers help understand the dynamical
properties of systems and provide important references for prediction.

Design Philosophy:
    - Quantitative Analysis: Use numerical metrics to describe chaotic characteristics
    - Plug-and-Play: Can be used independently or integrated into agents
    - Theoretical Foundation: Based on established chaos theory
    - Extensible: Additional chaos analysis tools can be added

Analyzer Types:
    - LyapunovEstimator: Lyapunov exponent estimator
        Measures sensitivity to initial conditions
        Positive exponent indicates chaos, negative indicates periodic, zero indicates critical

    - DimensionEstimator: Correlation dimension estimator
        Measures system complexity
        Higher dimension indicates more complex system

Usage:
    >>> from metathin_plus.chaos.analyzers import LyapunovEstimator, DimensionEstimator
    >>> 
    >>> # Create analyzers
    >>> lyap = LyapunovEstimator()
    >>> dim = DimensionEstimator()
    >>> 
    >>> # Analyze time series
    >>> le = lyap.estimate(time_series)
    >>> cd = dim.estimate(time_series)
    >>> 
    >>> if le > 0:
    ...     print(f"Chaotic system, Lyapunov exponent={le:.4f}, Correlation dimension={cd:.4f}")
    ... else:
    ...     print(f"Periodic system, Lyapunov exponent={le:.4f}")
"""

# ============================================================
# Chaos Analyzer Imports
# ============================================================

# -------------------------------------------------------------------
# 1. Lyapunov Exponent Estimator
#    Measures sensitivity to initial conditions, the core indicator of chaotic systems
# -------------------------------------------------------------------
from .lyapunov import LyapunovEstimator
"""
LyapunovEstimator - Lyapunov Exponent Estimator

The Lyapunov exponent is one of the most important indicators for characterizing chaotic systems.
It quantifies the exponential divergence rate of nearby trajectories in phase space.

Physical Interpretation:
    - λ > 0: Chaotic system (sensitive to initial conditions)
    - λ = 0: Critical state (e.g., period-doubling bifurcation point)
    - λ < 0: Periodic system (stable)

Algorithm Principle:
    1. Reconstruct phase space
    2. Track evolution of nearby trajectories
    3. Fit exponential divergence rate

Example:
    >>> lyap = LyapunovEstimator()
    >>> le = lyap.estimate(time_series)
    >>> print(f"Lyapunov exponent: {le:.4f}")
"""

# -------------------------------------------------------------------
# 2. Correlation Dimension Estimator
#    Measures system complexity, reflects the geometric structure of the attractor
# -------------------------------------------------------------------
from .dimension import DimensionEstimator
"""
DimensionEstimator - Correlation Dimension Estimator

The correlation dimension is an important indicator of system complexity.
It reflects the spatial distribution characteristics of points in phase space.

Physical Interpretation:
    - Higher dimension indicates more complex system
    - Non-integer dimensions are characteristic of chaotic systems
    - Integer dimensions typically correspond to periodic or quasi-periodic motion

Algorithm Principle:
    1. Compute correlation integral at different scales
    2. Fit slope of log-log curve
    3. Obtain correlation dimension estimate

Example:
    >>> dim = DimensionEstimator()
    >>> cd = dim.estimate(time_series)
    >>> print(f"Correlation dimension: {cd:.4f}")
"""

# ============================================================
# Export Interface Definition
# ============================================================
# __all__ controls what gets imported with 'from metathin_plus.chaos.analyzers import *'
# Organized by functionality for easy user reference
# ============================================================

__all__ = [
    'LyapunovEstimator',   # Lyapunov exponent estimator - measures degree of chaos
    'DimensionEstimator',   # Correlation dimension estimator - measures complexity
]

# ============================================================
# Usage Examples
# ============================================================
"""
>>> from metathin_plus.chaos.analyzers import LyapunovEstimator, DimensionEstimator
>>> import numpy as np
>>> 
>>> # 1. Generate test signals
>>> t = np.linspace(0, 100, 1000)
>>> 
>>> # Periodic signal
>>> periodic = np.sin(0.5 * t)
>>> 
>>> # Chaotic signal (Logistic map)
>>> chaotic = []
>>> x = 0.2
>>> for _ in range(1000):
...     x = 3.9 * x * (1 - x)
...     chaotic.append(x)
>>> 
>>> # Noise signal
>>> noise = np.random.randn(1000) * 0.5
>>> 
>>> # 2. Create analyzers
>>> lyap = LyapunovEstimator()
>>> dim = DimensionEstimator()
>>> 
>>> # 3. Analyze different signals
>>> signals = [
...     ("Periodic", periodic),
...     ("Chaotic", chaotic),
...     ("Noise", noise)
... ]
>>> 
>>> print("Signal Type\tLyapunov Exp\tCorrelation Dim")
>>> print("-" * 50)
>>> for name, signal in signals:
...     le = lyap.estimate(signal.tolist())
...     cd = dim.estimate(signal.tolist())
...     print(f"{name}\t{le:.4f}\t\t{cd:.4f}")
... 
>>> # 4. Use within MetaChaosBasic
>>> forecaster = MetaChaosBasic(model=my_model)
>>> 
>>> # Get current buffer data
>>> if len(forecaster.value_buffer) > 100:
...     data = list(forecaster.value_buffer)
...     le = lyap.estimate(data)
...     cd = dim.estimate(data)
...     print(f"Current system state: Lyapunov exponent={le:.4f}, Correlation dimension={cd:.4f}")
...     if le > 0.01:
...         print("System is in chaotic state")
...     else:
...         print("System is in periodic state")
"""

# ============================================================
# Chaos Metrics Interpretation Guide
# ============================================================
"""
Lyapunov Exponent Interpretation:
    λ > 0.1  : Strong chaos
    0.01 < λ < 0.1 : Weak chaos
    -0.01 < λ < 0.01 : Critical state
    λ < -0.01 : Periodic state

Correlation Dimension Interpretation:
    D > 5    : High-dimensional chaos (complex system)
    2 < D < 5 : Medium-dimensional chaos
    1 < D < 2 : Low-dimensional chaos
    D ≈ 1    : Periodic motion
    D ≈ 2    : Quasi-periodic motion

Important Notes:
    - These metrics require sufficient data length for accurate estimation
    - Practical applications should combine multiple indicators for comprehensive judgment
    - Noise can affect estimation accuracy
    - For short time series, results should be interpreted with caution
"""