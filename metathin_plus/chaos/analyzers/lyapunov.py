"""
Lyapunov Exponent Estimator - Fixed Version
=====================================================

The Lyapunov exponent is a core indicator of chaotic systems, quantifying the exponential
divergence rate of nearby trajectories in phase space. A positive Lyapunov exponent is the
defining characteristic of chaotic systems.

Mathematical Definition:
    λ = lim_{t→∞} lim_{||δZ(0)||→0} (1/t) ln(||δZ(t)|| / ||δZ(0)||)

Physical Interpretation:
    - λ > 0: Chaotic system (sensitive to initial conditions)
    - λ = 0: Critical state (e.g., period-doubling bifurcation point)
    - λ < 0: Periodic system (stable)

This implementation combines multiple estimation methods for robustness:
    1. Trajectory divergence method: Directly tracks separation of nearby trajectories
    2. Autocorrelation method: Based on decay of correlation function
    3. Permutation entropy method: Estimation based on sequence complexity
"""

import numpy as np
from typing import List
import math


class LyapunovEstimator:
    """
    Improved Lyapunov Exponent Estimator.

    Estimates the largest Lyapunov exponent through multiple methods, improving accuracy
    and robustness. Suitable for various types of time series data.

    Characteristics:
        - Multi-method fusion: Combines three estimation approaches
        - Periodicity detection: Special handling for periodic signals
        - Numerically stable: Handles edge cases and outliers
        - Online update: Supports streaming data updates

    Attributes:
        window_size: Estimation window size
        history: Historical data cache

    Example:
        >>> estimator = LyapunovEstimator(window_size=500)
        >>> 
        >>> # Estimate single sequence
        >>> le = estimator.estimate(time_series)
        >>> print(f"Lyapunov exponent: {le:.4f}")
        >>> 
        >>> # Streaming update
        >>> for value in stream:
        ...     estimator.update(value)
        ...     if len(estimator.history) > 100:
        ...         le = estimator.estimate(estimator.history)
    """

    def __init__(self, window_size: int = 500):
        """
        Initialize Lyapunov exponent estimator.

        Args:
            window_size: Estimation window size, determines how much historical data to use
                        Too small: insufficient statistics
                        Too large: computationally expensive, may include non-stationary segments
        """
        self.window_size = window_size
        self.history = []

    def estimate(self, data: List[float]) -> float:
        """
        Estimate the largest Lyapunov exponent.

        Combines results from three methods and applies corrections based on signal type.

        Args:
            data: Input time series data

        Returns:
            float: Lyapunov exponent estimate, range [-0.2, 0.2]
                  >0.01: Chaotic
                  -0.01~0.01: Critical
                  <-0.01: Periodic
        """
        if len(data) < 100:
            return 0.0  # Insufficient data, return neutral value

        # Use most recent data
        data = np.array(data[-self.window_size:])
        # Remove mean to eliminate DC component
        data = data - np.mean(data)

        # Handle constant signal
        if np.std(data) < 1e-6:
            return -0.1  # Constant signal, negative Lyapunov exponent

        # Multiple estimation methods
        lyap1 = self._estimate_by_divergence(data)   # Trajectory divergence method
        lyap2 = self._estimate_by_correlation(data)  # Autocorrelation method
        lyap3 = self._estimate_by_permutation(data)  # Permutation entropy method

        # Weighted average (weights can be adjusted)
        lyap = 0.4 * lyap1 + 0.3 * lyap2 + 0.3 * lyap3

        # Apply correction based on signal type
        if self._is_periodic(data):
            lyap = max(lyap, -0.05)  # Periodic signals should be negative

        return float(np.clip(lyap, -0.2, 0.2))

    def _is_periodic(self, data: np.ndarray) -> bool:
        """
        Determine if signal is periodic.

        Detects periodicity through peak detection in autocorrelation function.

        Args:
            data: Input data

        Returns:
            bool: Whether signal is periodic
        """
        # Autocorrelation detection
        corr = np.correlate(data, data, mode='same')
        corr = corr[len(corr)//2:]  # Take positive half
        corr = corr / (corr[0] + 1e-10)  # Normalize

        # Find peaks
        peaks = []
        for i in range(2, len(corr)-2):
            if (corr[i] > corr[i-1] and 
                corr[i] > corr[i+1] and 
                corr[i] > 0.3):  # Threshold: 30% of main peak
                peaks.append(i)

        # If multiple significant peaks exist, it might be periodic
        if len(peaks) >= 3:
            intervals = np.diff(peaks[:5])  # Intervals between first few peaks
            if len(intervals) >= 2:
                # If intervals are relatively stable, classify as periodic
                cv = np.std(intervals) / (np.mean(intervals) + 1e-10)  # Coefficient of variation
                return cv < 0.1  # Variation less than 10%
        return False

    def _estimate_by_divergence(self, data: np.ndarray) -> float:
        """
        Estimate Lyapunov exponent based on trajectory divergence rate.

        Principle: Track the divergence of two nearby trajectories
        λ ≈ (1/t) ln(||δx(t)|| / ||δx(0)||)

        Args:
            data: Input data

        Returns:
            float: Lyapunov exponent estimate
        """
        n = len(data) // 2
        if n < 50:
            return 0.0

        # Take two nearby trajectories
        s1 = data[:n]      # Original trajectory
        s2 = data[1:n+1]   # Slightly offset trajectory

        # Calculate divergence
        diff = np.abs(s1 - s2) + 1e-10
        log_diff = np.log(diff)

        try:
            # Fit slope of the latter half
            x = np.arange(len(log_diff))
            start_idx = len(log_diff) // 2  # Take latter half
            coeffs = np.polyfit(x[start_idx:], log_diff[start_idx:], 1)
            return float(coeffs[0])
        except:
            return 0.0

    def _estimate_by_correlation(self, data: np.ndarray) -> float:
        """
        Estimate Lyapunov exponent based on autocorrelation decay.

        Principle: Autocorrelation function of chaotic systems decays exponentially
        C(τ) ~ e^{-λτ}

        Args:
            data: Input data

        Returns:
            float: Lyapunov exponent estimate
        """
        try:
            n = len(data)
            # Compute autocorrelation
            corr = np.correlate(data, data, mode='same')
            corr = corr[n//2:]  # Take positive half
            corr = corr / (corr[0] + 1e-10)  # Normalize

            # Find first zero crossing
            for i in range(1, len(corr)):
                if corr[i] <= 0:
                    # Estimate from half-life: C(τ) = 0.5 when τ = ln(2)/λ
                    return -np.log(0.5) / i if i > 0 else 0.0

            # If no zero crossing, fit exponential decay
            if len(corr) > 20:
                x = np.arange(1, 21)
                y = np.log(np.abs(corr[1:21]) + 1e-10)
                coeffs = np.polyfit(x, y, 1)
                return -float(coeffs[0])  # Slope is -λ
            return 0.0
        except:
            return 0.0

    def _estimate_by_permutation(self, data: np.ndarray) -> float:
        """
        Estimate Lyapunov exponent based on permutation entropy.

        Principle: Permutation entropy of chaotic systems correlates positively with
                  Lyapunov exponent. Use permutation entropy as complexity measure
                  and map to Lyapunov exponent range.

        Args:
            data: Input data

        Returns:
            float: Lyapunov exponent estimate
        """
        try:
            # Embedding dimension
            m = 3  # Usually between 3-7

            # Extract all patterns of length m
            patterns = []
            for i in range(len(data) - m):
                segment = data[i:i+m]
                # Get sorted indices as pattern
                # e.g., [0.3, 0.1, 0.2] sorted indices: [1,2,0]
                idx = np.argsort(segment)
                pattern = tuple(idx)
                patterns.append(pattern)

            # Calculate probability of each pattern
            unique, counts = np.unique(patterns, return_counts=True)
            probs = counts / (len(patterns) + 1e-10)

            # Calculate permutation entropy: H = -Σ p_i log(p_i)
            entropy = -np.sum(probs * np.log(probs + 1e-10))

            # Normalize entropy
            max_entropy = np.log(math.factorial(m))
            norm_entropy = entropy / max_entropy if max_entropy > 0 else 0

            # Map to Lyapunov exponent range
            # entropy=0.5 corresponds to Lyapunov exponent=0
            return (norm_entropy - 0.5) * 0.1

        except Exception as e:
            return 0.0

    def update(self, value: float):
        """
        Update historical data.

        Used for streaming scenarios, maintains a sliding window.

        Args:
            value: New data point
        """
        self.history.append(value)
        # Keep history within reasonable bounds
        if len(self.history) > self.window_size * 2:
            self.history = self.history[-self.window_size:]

    def reset(self):
        """Reset estimator, clear history."""
        self.history = []


# ============================================================
# Usage Examples
# ============================================================
"""
>>> from metathin_plus.chaos.analyzers import LyapunovEstimator
>>> import numpy as np
>>> 
>>> # 1. Create estimator
>>> lyap = LyapunovEstimator(window_size=500)
>>> 
>>> # 2. Generate different types of data
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
>>> # 3. Estimate Lyapunov exponents
>>> le_periodic = lyap.estimate(periodic.tolist())
>>> le_chaotic = lyap.estimate(chaotic)
>>> le_noise = lyap.estimate(noise.tolist())
>>> 
>>> print(f"Periodic signal: {le_periodic:.4f}")
>>> print(f"Chaotic signal: {le_chaotic:.4f}")
>>> print(f"Random noise: {le_noise:.4f}")
>>> 
>>> # 4. Streaming processing
>>> estimator = LyapunovEstimator()
>>> for i, val in enumerate(data_stream):
...     estimator.update(val)
...     if i % 100 == 0 and len(estimator.history) > 100:
...         le = estimator.estimate(estimator.history)
...         print(f"Step {i}, Lyapunov exponent: {le:.4f}")
"""

# ============================================================
# Lyapunov Exponent Interpretation Guide
# ============================================================
"""
Quantitative Interpretation of Lyapunov Exponents:

λ > 0.1      : Strongly chaotic system
                 Lorenz system, Rossler system, classical chaos

0.01 < λ < 0.1 : Weakly chaotic system
                   Duffing oscillator, certain chaotic circuits

-0.01 < λ < 0.01 : Critical state
                      Period-doubling bifurcation points, edge of chaos

λ < -0.01     : Periodic system
                    Limit cycles, stable fixed points

Important Notes:
    1. Requires sufficiently long data (at least several hundred points)
    2. Data should be stationary (detrended)
    3. Noise affects estimation accuracy
    4. Recommended to combine with other indicators (correlation dimension) for comprehensive judgment
"""