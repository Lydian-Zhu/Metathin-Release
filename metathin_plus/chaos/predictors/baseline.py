"""
Baseline Predictors (Simple Methods)
=====================================================

Provides the simplest prediction methods as baselines for performance comparison.
Although simple, these methods serve as lower bounds for evaluating more complex approaches.

Contains:
    - PersistentPredictor: Persistence prediction (naive forecast)
    - LinearTrendPredictor: Linear trend extrapolation

These baselines are essential for:
    - Establishing minimum acceptable performance
    - Quantifying improvement of complex methods
    - Detecting overfitting (if complex method performs worse than baseline)
"""

import numpy as np
from collections import deque
from ..base import ChaosPredictor, SystemState, PredictionResult


# ============================================================
# Persistent Predictor
# ============================================================

class PersistentPredictor(ChaosPredictor):
    """
    Persistent Predictor (Baseline Method)

    The simplest prediction method, assuming the next value equals the current value:
        x_{t+1} = x_t

    This is also known as the "naive forecast" or "random walk" prediction.
    In chaos theory, this corresponds to the assumption that the system remains
    in its current state - a reasonable baseline for highly stochastic systems.

    Characteristics:
        ✅ Zero computational cost
        ✅ No parameters to tune
        ✅ Universal applicability
        ⚠️ Only serves as a performance lower bound

    Theoretical Foundation:
        For a purely random walk, the optimal predictor is the last observed value.
        Any sophisticated method should outperform this baseline to be considered useful.

    Parameters:
        name: str, Predictor name (default: "Persistent")

    Example:
        >>> # Create persistent predictor
        >>> predictor = PersistentPredictor()
        >>> 
        >>> # Make prediction
        >>> state = SystemState(data=0.5)
        >>> result = predictor.predict(state)
        >>> print(f"Predicted value: {result.value}")  # 0.5
        >>> 
        >>> # Use as baseline for comparison
        >>> simple_error = persistent.get_recent_error()
        >>> complex_error = complex_predictor.get_recent_error()
        >>> improvement = (simple_error - complex_error) / simple_error * 100
    """

    def __init__(self, name: str = "Persistent"):
        """
        Initialize persistent predictor.

        Args:
            name: Predictor name (used for identification in logs and results)
        """
        super().__init__(name)

    def predict(self, state: SystemState, **kwargs) -> PredictionResult:
        """
        Execute persistent prediction.

        The prediction is simply the current value, as per the persistence assumption.
        This method ignores all historical information and only uses the current state.

        Args:
            state: Current system state containing the value to persist
            **kwargs: Additional parameters (ignored by this method)

        Returns:
            PredictionResult: Prediction result with value = current value
        """
        current_value = state.get_value()

        return PredictionResult(
            value=current_value,
            confidence=0.5,  # Confidence set to 0.5 (moderate certainty)
            method=self.name,
            metadata={
                'baseline_type': 'persistent',
                'current_value': current_value
            }
        )


# ============================================================
# Linear Trend Predictor
# ============================================================

class LinearTrendPredictor(ChaosPredictor):
    """
    Linear Trend Predictor (Baseline Method)

    Fits a linear trend to recent data and extrapolates the next value:
        x_{t+1} = a * (t+1) + b
    where a and b are estimated using least squares regression.

    This method captures simple linear trends and is particularly useful for
    systems with steady drift or seasonal components. It serves as a baseline
    for methods that claim to capture nonlinear dynamics.

    Characteristics:
        ✅ Accounts for linear trends
        ✅ Adapts to changing trends
        ✅ Minimal parameters (only window size)
        ⚠️ Only serves as a performance lower bound for nonlinear methods

    Parameters:
        window: int, Window size for trend fitting (default: 10)
                Must be >= 2 for meaningful linear regression.
                Larger windows produce smoother trends but slower adaptation.
                Smaller windows respond quickly to recent changes but are noisier.
        name: str, Predictor name (default: "LinearTrend")

    Example:
        >>> # Create linear trend predictor
        >>> predictor = LinearTrendPredictor(window=20)
        >>> 
        >>> # Predict for a system with trend
        >>> state = SystemState(data=0.5)
        >>> result = predictor.predict(state)
        >>> print(f"Predicted value: {result.value}")
        
    Mathematical Details:
        The linear model is: x_i = a * i + b + ε_i
        Parameters are estimated by minimizing Σ(x_i - a*i - b)²
        The next value is predicted as: x_{n+1} = a * (n+1) + b
    """

    def __init__(self, window: int = 10, name: str = "LinearTrend"):
        """
        Initialize linear trend predictor.

        Args:
            window: Window size for trend fitting. Must be >= 2.
                   - Larger window: More stable but slower to adapt
                   - Smaller window: More responsive but potentially noisy
            name: Predictor name for identification

        Raises:
            ValueError: If window < 2
        """
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}")

        super().__init__(name)
        self.window = window
        self.value_history = deque(maxlen=window * 2)  # Store extra history for stability

    def predict(self, state: SystemState, **kwargs) -> PredictionResult:
        """
        Execute linear trend prediction.

        Implementation steps:
            1. Store current value in history
            2. If insufficient history, fall back to persistence prediction
            3. Fit linear trend to recent window points using least squares
            4. Extrapolate to next time step
            5. Return prediction with confidence based on data availability

        Args:
            state: Current system state
            **kwargs: Additional parameters (ignored by this method)

        Returns:
            PredictionResult: Prediction with linear trend extrapolation
        """
        current_value = state.get_value()
        self.value_history.append(current_value)

        # Insufficient data for trend fitting - fall back to persistence
        if len(self.value_history) < self.window:
            return PredictionResult(
                value=current_value,
                confidence=0.5,
                method=self.name,
                metadata={
                    'baseline_type': 'linear_trend',
                    'status': 'insufficient_data',
                    'window_used': len(self.value_history)
                }
            )

        # Convert to numpy array for efficient computation
        values = np.array(list(self.value_history))

        try:
            # Time indices: 0, 1, 2, ..., n-1
            x = np.arange(len(values))
            
            # Polynomial fit of degree 1 (linear)
            # coeffs[0] = slope (a), coeffs[1] = intercept (b)
            coeffs = np.polyfit(x, values, 1)
            
            # Predict next value at index = len(values)
            prediction = coeffs[0] * len(values) + coeffs[1]
            
            # Calculate goodness of fit (R²) as confidence metric
            y_pred = coeffs[0] * x + coeffs[1]
            ss_res = np.sum((values - y_pred) ** 2)
            ss_tot = np.sum((values - np.mean(values)) ** 2)
            r_squared = 1 - (ss_res / (ss_tot + 1e-10))
            
            # Confidence increases with more data and better fit
            confidence = 0.5 + 0.3 * (len(self.value_history) / (self.window * 2)) + 0.2 * r_squared
            confidence = min(0.9, confidence)  # Cap at 0.9

        except Exception as e:
            # Fallback to persistence if fitting fails
            prediction = current_value
            confidence = 0.5
            self.logger.debug(f"Linear trend fitting failed: {e}")

        return PredictionResult(
            value=float(prediction),
            confidence=float(confidence),
            method=self.name,
            metadata={
                'baseline_type': 'linear_trend',
                'window_size': self.window,
                'data_points': len(self.value_history),
                'slope': float(coeffs[0]) if 'coeffs' in locals() else None,
                'intercept': float(coeffs[1]) if 'coeffs' in locals() else None,
                'r_squared': float(r_squared) if 'r_squared' in locals() else None
            }
        )

    def get_trend_strength(self) -> float:
        """
        Get the strength of the current trend.
        
        Returns:
            float: Absolute slope normalized by data range (0-1)
        """
        if len(self.value_history) < self.window:
            return 0.0

        values = np.array(list(self.value_history))
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        slope = abs(coeffs[0])
        data_range = np.ptp(values) + 1e-10  # peak-to-peak range
        
        return min(1.0, slope / data_range)

    def reset(self) -> None:
        """Reset the predictor's internal state."""
        self.value_history.clear()
        self.logger.debug(f"{self.name} reset")