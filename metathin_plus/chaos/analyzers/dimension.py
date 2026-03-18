"""
Correlation Dimension Estimator
=====================================================

The correlation dimension is an important metric for measuring the complexity of chaotic systems,
reflecting the geometric structure of the attractor in phase space. Unlike topological dimension,
the correlation dimension can be fractional, which is a hallmark of chaotic systems.

Mathematical Definition:
    D2 = lim_{r→0} log C(r) / log r
    
    where C(r) is the correlation integral, representing the proportion of point pairs with distance < r:
    C(r) = (2/(N(N-1))) Σ_{i<j} Θ(r - ||Xi - Xj||)

Physical Interpretation:
    - Fractional dimension: Chaotic systems (e.g., Lorenz attractor dimension ≈ 2.06)
    - Integer dimension: Periodic systems (dimension ≈ 1) or quasi-periodic systems (dimension ≈ 2)
    - High dimension: High system complexity

This implementation combines multiple estimation methods for robustness:
    1. Hurst exponent method: Based on scaling behavior
    2. Zero-crossing method: Based on signal fluctuation frequency
    3. Spectral entropy method: Based on frequency distribution complexity
"""

import numpy as np
from typing import List, Optional
import logging


class DimensionEstimator:
    """
    Correlation Dimension Estimator.

    Estimates correlation dimension through multiple methods, suitable for various
    time series data types.

    Characteristics:
        - Multi-method fusion: Combines three estimation approaches
        - Numerically stable: Handles edge cases and outliers
        - Fast estimation: No phase space reconstruction needed, computationally efficient
        - Online update: Supports streaming data updates

    Attributes:
        window_size: Estimation window size
        history: Historical data cache
        min_data_points: Minimum data points required

    Example:
        >>> estimator = DimensionEstimator(window_size=500)
        >>> 
        >>> # Estimate single sequence
        >>> dim = estimator.estimate(time_series)
        >>> print(f"Correlation dimension: {dim:.4f}")
        >>> 
        >>> # Streaming update
        >>> for value in stream:
        ...     estimator.update(value)
        ...     if len(estimator.history) > 100:
        ...         dim = estimator.estimate(estimator.history)
    """

    def __init__(self, window_size: int = 500, min_data_points: int = 50):
        """
        Initialize correlation dimension estimator.

        Parameters:
            window_size: Estimation window size, determines how much historical data to use
                        Too small: insufficient statistics
                        Too large: may include non-stationary segments
            min_data_points: Minimum data points required, returns default if below this
        """
        self.window_size = window_size
        self.min_data_points = min_data_points
        self.history = []
        self.logger = logging.getLogger("metathin_plus.chaos.analyzers.DimensionEstimator")

    def estimate(self, data: Optional[List[float]] = None) -> float:
        """
        Estimate correlation dimension.

        Combines results from three methods for robust estimation.

        Parameters:
            data: Time series data, uses internal history if None

        Returns:
            float: Correlation dimension estimate, range [0.5, 3.0]
                  1.0: Periodic motion
                  1.5-2.0: Low-dimensional chaos
                  2.0-3.0: High-dimensional chaos
        """
        # Determine data to analyze
        if data is None:
            data = self.history
        else:
            data = list(data)

        # Check data validity
        if not data or len(data) < self.min_data_points:
            self.logger.debug(f"Insufficient data: {len(data) if data else 0} < {self.min_data_points}")
            return 1.0  # Conservative estimate

        try:
            # Use most recent data points
            data_array = np.array(data[-self.window_size:], dtype=np.float64)
            
            # Check data validity
            if len(data_array) < self.min_data_points:
                return 1.0

            # Remove mean to eliminate DC component
            data_array = data_array - np.mean(data_array)

            # Handle constant signal
            std_dev = np.std(data_array)
            if std_dev < 1e-6:
                self.logger.debug("Detected constant signal")
                return 0.5  # Constant signal, dimension near 0

            # Three estimation methods
            dim1 = self._estimate_by_hurst(data_array)       # Hurst exponent method
            dim2 = self._estimate_by_zerocross(data_array)   # Zero-crossing method
            dim3 = self._estimate_by_spectrum(data_array)    # Spectral entropy method

            # Filter invalid values
            valid_dims = []
            for d in [dim1, dim2, dim3]:
                if not np.isnan(d) and not np.isinf(d) and 0.5 <= d <= 3.0:
                    valid_dims.append(d)

            if not valid_dims:
                self.logger.warning("All estimation methods returned invalid values")
                return 1.0

            # Weighted average (weights can be adjusted)
            weights = [0.4, 0.3, 0.3][:len(valid_dims)]
            weights = np.array(weights) / np.sum(weights)
            dim = np.average(valid_dims, weights=weights)

            return float(np.clip(dim, 0.5, 3.0))

        except Exception as e:
            self.logger.error(f"Dimension estimation failed: {e}")
            return 1.0

    def _estimate_by_hurst(self, data: np.ndarray) -> float:
        """
        Estimate dimension based on Hurst exponent.

        Principle: The Hurst exponent H describes long-range correlation in time series,
                   Correlation dimension D = 2 - H

        Steps:
            1. Coarse-grain at different scales
            2. Calculate fluctuation at each scale
            3. Fit log(scale)-log(fluctuation) curve
            4. Slope is the Hurst exponent

        Parameters:
            data: Input data

        Returns:
            float: Correlation dimension estimate
        """
        try:
            # Different scales
            scales = [2, 4, 8, 16, 32, 64]
            # Filter scales larger than data length
            valid_scales = [s for s in scales if s < len(data) // 2]
            
            if len(valid_scales) < 2:
                return 1.0

            fluctuations = []

            for scale in valid_scales:
                n = len(data) // scale
                if n < 2:
                    continue

                # Coarse-graining: average data in segments
                coarse = np.array([np.mean(data[i*scale:(i+1)*scale]) 
                                  for i in range(n)])

                # Calculate fluctuation (standard deviation)
                if len(coarse) > 1:
                    fluct = np.std(np.diff(coarse))
                    if fluct > 0:
                        fluctuations.append(fluct)

            if len(fluctuations) >= 2:
                # Log-log plot fitting
                log_scales = np.log(valid_scales[:len(fluctuations)])
                log_flucts = np.log(fluctuations)

                # Fit to get Hurst exponent
                coeffs = np.polyfit(log_scales, log_flucts, 1)
                hurst = coeffs[0]

                # Correlation dimension = 2 - Hurst
                dim = 2.0 - hurst
                return float(np.clip(dim, 0.5, 3.0))
            else:
                return 1.0
                
        except Exception as e:
            self.logger.debug(f"Hurst estimation failed: {e}")
            return 1.0

    def _estimate_by_zerocross(self, data: np.ndarray) -> float:
        """
        Estimate dimension based on zero-crossing rate.

        Principle: The zero-crossing rate of a signal correlates with its complexity
                  Higher zero-crossing rate indicates higher dimension

        Steps:
            1. Count zero crossings
            2. Calculate zero-crossing rate
            3. Estimate dimension based on empirical thresholds

        Parameters:
            data: Input data

        Returns:
            float: Correlation dimension estimate
        """
        try:
            if len(data) < 10:
                return 1.0

            # Count zero crossings
            zero_crossings = 0
            for i in range(1, len(data)):
                if data[i-1] * data[i] < 0:
                    zero_crossings += 1

            zero_rate = zero_crossings / len(data)

            # Estimate dimension based on zero-crossing rate
            if zero_rate > 0.3:
                return 2.5  # High dimension (frequent zero crossings)
            elif zero_rate > 0.15:
                return 2.0  # Medium dimension
            elif zero_rate > 0.05:
                return 1.5  # Low dimension
            else:
                return 1.0  # Very low dimension (rare zero crossings)
                
        except Exception as e:
            self.logger.debug(f"Zero-crossing estimation failed: {e}")
            return 1.0

    def _estimate_by_spectrum(self, data: np.ndarray) -> float:
        """
        Estimate dimension based on power spectrum.

        Principle: The distribution of power spectrum reflects signal complexity
                  More spread out spectrum indicates higher dimension

        Steps:
            1. Compute FFT to get power spectrum
            2. Calculate power spectrum entropy (reflects distribution uniformity)
            3. Normalize entropy
            4. Map to dimension range

        Parameters:
            data: Input data

        Returns:
            float: Correlation dimension estimate
        """
        try:
            if len(data) < 20:
                return 1.0

            # Compute FFT
            fft_vals = np.fft.fft(data)
            # Take positive frequency power (exclude DC)
            power = np.abs(fft_vals[1:len(fft_vals)//2])**2

            if len(power) < 5:
                return 1.0

            # Normalize power spectrum
            power_sum = np.sum(power)
            if power_sum < 1e-10:
                return 1.0
                
            power = power / power_sum

            # Compute power spectrum entropy: H = -Σ p_i log(p_i)
            # Higher entropy indicates more uniform frequency distribution
            entropy = -np.sum(power * np.log(power + 1e-10))

            # Normalize entropy (divide by maximum possible entropy)
            max_entropy = np.log(len(power))
            if max_entropy > 0:
                norm_entropy = entropy / max_entropy
            else:
                norm_entropy = 0.5

            # Map to dimension range [1, 3]
            # entropy=0 (single frequency) → dimension≈1
            # entropy=1 (white noise) → dimension≈3
            dim = 1.0 + 2.0 * norm_entropy
            
            return float(np.clip(dim, 1.0, 3.0))
            
        except Exception as e:
            self.logger.debug(f"Spectral estimation failed: {e}")
            return 1.0

    def update(self, value: float):
        """
        Update historical data.

        Used for streaming scenarios, maintains a sliding window.

        Parameters:
            value: New data point
        """
        self.history.append(value)
        # Keep history within reasonable bounds
        if len(self.history) > self.window_size * 2:
            self.history = self.history[-self.window_size:]

    def reset(self):
        """Reset estimator, clear history."""
        self.history = []

    def get_status(self) -> dict:
        """Get estimator status."""
        return {
            'history_size': len(self.history),
            'window_size': self.window_size,
            'min_data_points': self.min_data_points,
            'last_estimate': self.estimate(self.history[-self.window_size:]) if len(self.history) >= self.min_data_points else None
        }


# ============================================================
# Usage Examples
# ============================================================
"""
>>> from metathin_plus.chaos.analyzers import DimensionEstimator
>>> import numpy as np
>>> 
>>> # 1. Create estimator
>>> dim = DimensionEstimator(window_size=500)
>>> 
>>> # 2. Generate different types of data
>>> t = np.linspace(0, 100, 1000)
>>> 
>>> # Periodic signal (dimension≈1)
>>> periodic = np.sin(0.5 * t)
>>> 
>>> # Chaotic signal (Logistic map)
>>> chaotic = []
>>> x = 0.2
>>> for _ in range(1000):
...     x = 3.9 * x * (1 - x)
...     chaotic.append(x)
>>> 
>>> # Noise signal (theoretically high dimension)
>>> noise = np.random.randn(1000) * 0.5
>>> 
>>> # 3. Estimate correlation dimension
>>> d_periodic = dim.estimate(periodic.tolist())
>>> d_chaotic = dim.estimate(chaotic)
>>> d_noise = dim.estimate(noise.tolist())
>>> 
>>> print(f"Periodic signal: {d_periodic:.4f}")
>>> print(f"Chaotic signal: {d_chaotic:.4f}")
>>> print(f"Random noise: {d_noise:.4f}")
>>> 
>>> # 4. Streaming processing
>>> estimator = DimensionEstimator()
>>> for i, val in enumerate(data_stream):
...     estimator.update(val)
...     if i % 100 == 0 and len(estimator.history) > 100:
...         d = estimator.estimate(estimator.history)
...         print(f"Step {i}, Correlation dimension: {d:.4f}")
"""

# ============================================================
# Correlation Dimension Interpretation Guide
# ============================================================
"""
Quantitative Interpretation of Correlation Dimension:

D > 2.5     : High-dimensional chaos
               Complex systems, possibly multiple degrees of freedom
               Examples: Turbulence, high-dimensional chaotic systems

2.0 < D < 2.5 : Medium-dimensional chaos
                  Lorenz system (≈2.06)
                  Rossler system (≈2.01)

1.5 < D < 2.0 : Low-dimensional chaos
                   Duffing oscillator
                   Certain chaotic circuits

1.0 < D < 1.5 : Quasi-periodic or weak chaos
                   Period-doubling bifurcation points
                   Torus attractors

D ≈ 1.0     : Periodic motion
                  Limit cycles
                  Stable periodic oscillations

D < 1.0     : Constant or extremely simple systems

Important Notes:
    1. Requires sufficiently long data (at least several hundred points)
    2. Data should be detrended and denoised
    3. In real systems, noise tends to increase the estimated dimension
    4. Recommended to combine with Lyapunov exponents for comprehensive judgment
"""