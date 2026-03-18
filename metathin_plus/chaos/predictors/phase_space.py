"""
Phase Space Reconstruction Predictors (Optimized Edition)
======================================================================

Contains:
    - FullPhaseSpacePredictor: Complete phase space reconstruction (recommended for production)
    - PhaseSpacePredictor: Simplified version (for comparison only)

Theoretical Background:
    Takens' Embedding Theorem: For a time series {x(t)}, construct delay vectors
    X(t) = [x(t), x(t-τ), x(t-2τ), ..., x(t-(m-1)τ)]
    If m > 2d, the reconstructed phase space is diffeomorphic to the original system.
"""

import numpy as np
from collections import deque
from typing import List, Tuple, Optional, Dict, Any
from scipy.spatial import KDTree
import logging
from ..base import ChaosPredictor, SystemState, PredictionResult


# ============================================================
# Full Version: Complete Phase Space Reconstruction (Recommended)
# ============================================================

class FullPhaseSpacePredictor(ChaosPredictor):
    """
    Complete Phase Space Reconstruction Predictor ⭐ Recommended for Production

    A full-featured phase space reconstruction implementation featuring:
        - Delay vector construction from time series
        - KD-Tree accelerated nearest neighbor search
        - Trajectory evolution tracking
        - Weighted average prediction based on neighbor distances

    Parameters:
        embed_dim: int, Embedding dimension (default: 5)
                   Determines the dimension of the reconstructed phase space.
                   Should be > 2d where d is the attractor dimension.
        delay: int, Time delay (default: 3)
                Determines the sampling interval in the reconstructed space.
                Often chosen using mutual information.
        k_neighbors: int, Number of nearest neighbors (default: 5)
                    More neighbors provide smoother predictions but may miss local dynamics.
        min_history: int, Minimum history required for prediction (default: 100)
                    Below this threshold, predictions fall back to current value.
        name: str, Predictor name (default: "FullPhaseSpace")

    Example:
        >>> # Create full phase space predictor
        >>> predictor = FullPhaseSpacePredictor(embed_dim=7, delay=2, k_neighbors=8)
        >>>
        >>> # Predict next value
        >>> state = SystemState(data=0.5, timestamp=0.0)
        >>> result = predictor.predict(state)
        >>> print(f"Predicted: {result.value:.4f}, Confidence: {result.confidence:.2%}")

    Theory:
        Based on Takens' embedding theorem, this method reconstructs the system's
        attractor from a single time series. The dynamics in the reconstructed space
        are topologically equivalent to the original system, enabling prediction by
        tracking the evolution of nearby trajectories.
    """

    def __init__(self,
                 embed_dim: int = 5,
                 delay: int = 3,
                 k_neighbors: int = 5,
                 min_history: int = 100,
                 name: str = "FullPhaseSpace"):
        """
        Initialize full phase space predictor.

        Args:
            embed_dim: Embedding dimension (m) - number of delayed coordinates
            delay: Time delay (τ) - spacing between coordinates
            k_neighbors: Number of nearest neighbors for prediction
            min_history: Minimum history length before attempting prediction
            name: Predictor name for identification and logging
        """
        super().__init__(name)
        self.embed_dim = embed_dim
        self.delay = delay
        self.k_neighbors = k_neighbors
        self.min_history = min_history
        self.value_history = deque(maxlen=2000)

        # Phase space data structures
        self.vectors = None  # Phase space vectors
        self.targets = None  # Corresponding target values (next step)
        self.tree = None     # KD-Tree for fast neighbor search

        # Restart count (inherited from base class)
        self.restart_count = 0

    def _build_phase_space(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Construct phase space from time series.

        Converts the time series into delay vectors according to:
            X(t) = [x(t), x(t-τ), x(t-2τ), ..., x(t-(m-1)τ)]

        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]: (phase space vectors, target values)
                                                                Returns (None, None) if insufficient data
        """
        values = np.array(list(self.value_history))
        n = len(values)

        # Check if we have enough data
        min_required = self.embed_dim * self.delay + self.k_neighbors
        if n < min_required:
            return None, None

        vectors = []
        targets = []

        for i in range(self.embed_dim * self.delay, n - 1):
            vector = []
            valid = True
            for j in range(self.embed_dim):
                idx = i - j * self.delay
                if 0 <= idx < len(values):
                    vector.append(values[idx])
                else:
                    valid = False
                    break
            if valid:
                vectors.append(vector)
                targets.append(values[i + 1])  # Predict the next value

        if len(vectors) < self.k_neighbors:
            return None, None

        return np.array(vectors), np.array(targets)

    def _update_kdtree(self):
        """Update the KD-Tree index for fast neighbor search."""
        vectors, targets = self._build_phase_space()
        if vectors is not None and len(vectors) > self.k_neighbors:
            self.vectors = vectors
            self.targets = targets
            try:
                self.tree = KDTree(vectors)
            except Exception as e:
                self._logger.error(f"Failed to build KD-Tree: {e}")
                self.tree = None
        else:
            self.vectors = None
            self.targets = None
            self.tree = None

    def _find_neighbors(self, current_point: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Find nearest neighbors using KD-Tree for acceleration.

        Args:
            current_point: Current phase space point

        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]: (neighbor indices, neighbor distances)
                                                                Returns (None, None) if search fails
        """
        if self.tree is None or self.vectors is None or len(self.vectors) == 0:
            return None, None

        k = min(self.k_neighbors, len(self.vectors))
        if k == 0:
            return None, None

        try:
            # Ensure correct shape for query
            if current_point.ndim == 1:
                query_point = current_point.reshape(1, -1)
            else:
                query_point = current_point

            distances, indices = self.tree.query(query_point, k=k)

            # Flatten results if necessary
            if indices.ndim == 2:
                indices = indices[0]
            if distances.ndim == 2:
                distances = distances[0]

            return indices, distances

        except Exception as e:
            self._logger.error(f"Nearest neighbor search failed: {e}")
            return None, None

    def _predict_by_evolution(self,
                             neighbor_indices: np.ndarray,
                             distances: np.ndarray) -> Tuple[float, float]:
        """
        Predict by tracking trajectory evolution of neighbors.

        Args:
            neighbor_indices: Indices of nearest neighbors
            distances: Distances to nearest neighbors

        Returns:
            Tuple[float, float]: (predicted value, confidence)
        """
        if neighbor_indices is None or len(neighbor_indices) == 0:
            return 0.0, 0.0

        next_values = []
        weights = []

        for idx, dist in zip(neighbor_indices, distances):
            if idx + 1 < len(self.targets):
                next_values.append(self.targets[idx + 1])
                weight = 1.0 / (dist + 1e-8)  # Inverse distance weighting
                weights.append(weight)

        if not next_values:
            return 0.0, 0.0

        # Normalize weights
        weights = np.array(weights)
        weights = weights / (np.sum(weights) + 1e-8)

        # Weighted average prediction
        prediction = np.sum(np.array(next_values) * weights)

        # Confidence calculation based on:
        #   - Average neighbor distance (closer = more confident)
        #   - Number of valid neighbors found
        if len(distances) > 0:
            avg_distance = np.mean(distances)
            distance_confidence = 1.0 / (1.0 + avg_distance)
            neighbor_confidence = min(1.0, len(next_values) / self.k_neighbors)
            confidence = 0.5 * distance_confidence + 0.5 * neighbor_confidence
        else:
            confidence = 0.3

        return prediction, confidence

    def predict(self, state: SystemState, **kwargs) -> PredictionResult:
        """
        Execute phase space prediction.

        Complete prediction pipeline:
            1. Update history with current value
            2. Check if sufficient data available
            3. Construct current phase point
            4. Update KD-Tree index
            5. Find nearest neighbors
            6. Predict based on neighbor evolution

        Args:
            state: Current system state
            **kwargs: Additional parameters (ignored)

        Returns:
            PredictionResult: Prediction result with value and confidence
        """
        current_value = state.get_value()
        self.value_history.append(current_value)

        # Check if we have enough history
        if len(self.value_history) < self.min_history:
            self._logger.debug(f"Insufficient history: {len(self.value_history)} < {self.min_history}")
            result = PredictionResult(
                value=current_value,
                confidence=0.3,
                method=self.name
            )
            self.predictions.append(result)
            return result

        # Update KD-Tree
        self._update_kdtree()

        if self.tree is None or self.vectors is None:
            self._logger.debug("Phase space reconstruction failed")
            result = PredictionResult(
                value=current_value,
                confidence=0.3,
                method=self.name
            )
            self.predictions.append(result)
            return result

        # Construct current phase point
        values = np.array(list(self.value_history))
        current_point = []
        valid_point = True

        for j in range(self.embed_dim):
            idx = -1 - j * self.delay
            if abs(idx) <= len(values):
                current_point.append(values[idx])
            else:
                valid_point = False
                break

        if not valid_point:
            self._logger.debug("Failed to construct current phase point")
            result = PredictionResult(
                value=current_value,
                confidence=0.3,
                method=self.name
            )
            self.predictions.append(result)
            return result

        current_point = np.array(current_point)

        # Find nearest neighbors
        neighbor_indices, distances = self._find_neighbors(current_point)

        if neighbor_indices is None or len(neighbor_indices) == 0:
            self._logger.debug("No neighbors found")
            result = PredictionResult(
                value=current_value,
                confidence=0.3,
                method=self.name
            )
            self.predictions.append(result)
            return result

        # Predict based on evolution
        prediction, confidence = self._predict_by_evolution(
            neighbor_indices, distances
        )

        # Sanity check: if prediction is anomalous, fall back to current value
        if np.isnan(prediction) or np.isinf(prediction) or abs(prediction) > 10 * abs(np.mean(values[-100:])):
            self._logger.debug(f"Anomalous prediction: {prediction}, falling back to current value")
            prediction = current_value
            confidence = 0.2

        confidence = float(np.clip(confidence, 0.2, 0.95))

        # Update prediction records via base class predictions list
        result = PredictionResult(
            value=float(prediction),
            confidence=confidence,
            method=self.name
        )
        self.predictions.append(result)

        return result

    def get_stats(self) -> Dict[str, Any]:
        """
        Get predictor statistics (overrides base class method).

        Returns:
            Dict[str, Any]: Statistics including phase-space specific metrics
        """
        stats = super().get_stats()
        stats.update({
            'embed_dim': self.embed_dim,
            'delay': self.delay,
            'k_neighbors': self.k_neighbors,
            'min_history': self.min_history,
            'vectors_size': len(self.vectors) if self.vectors is not None else 0,
            'kd_tree_built': self.tree is not None,
        })
        return stats

    def reset(self):
        """Reset predictor state (overrides base class method)."""
        super().reset()
        self.value_history.clear()
        self.vectors = None
        self.targets = None
        self.tree = None
        self.restart_count = 0


# ============================================================
# Simplified Version: For Comparison Only (Not for Production)
# ============================================================

class PhaseSpacePredictor(ChaosPredictor):
    """
    Simplified Phase Space Predictor ⚠️ For Comparison Only

    WARNING: This is a simplified version that uses linear extrapolation,
    NOT a true phase space reconstruction method. It exists only for baseline
    comparison and backward compatibility. For actual predictions, use
    FullPhaseSpacePredictor which implements proper Takens' embedding.

    This implementation mimics some characteristics of phase space methods
    (using recent history) but lacks the theoretical foundation and accuracy
    of the full version. Performance is significantly worse for chaotic systems.

    Parameters:
        embed_dim: int, Embedding dimension (interface compatibility only)
        delay: int, Time delay (interface compatibility only)
        k_neighbors: int, Number of neighbors (interface compatibility only)
        name: str, Predictor name (default: "PhaseSpace")

    Example:
        >>> # Only for comparison testing
        >>> simple = PhaseSpacePredictor()  # For baseline comparison
        >>> full = FullPhaseSpacePredictor()  # For actual use
    """

    def __init__(self,
                 embed_dim: int = 5,
                 delay: int = 3,
                 k_neighbors: int = 5,
                 name: str = "PhaseSpace"):
        """
        Initialize simplified phase space predictor.

        Args:
            embed_dim: Embedding dimension (interface compatibility only, not actually used)
            delay: Time delay (interface compatibility only, not actually used)
            k_neighbors: Number of neighbors (interface compatibility only, not actually used)
            name: Predictor name
        """
        super().__init__(name)
        self.embed_dim = embed_dim
        self.delay = delay
        self.k_neighbors = k_neighbors
        self.value_history = deque(maxlen=1000)

        # Issue warning about simplified nature
        import warnings
        warnings.warn(
            "Using simplified PhaseSpacePredictor. For actual predictions, "
            "use FullPhaseSpacePredictor which implements proper phase space reconstruction.",
            UserWarning,
            stacklevel=2
        )

    def predict(self, state: SystemState, **kwargs) -> PredictionResult:
        """
        Simplified prediction using linear extrapolation of recent points.

        This is NOT a true phase space prediction but a simple linear trend
        approximation for baseline comparison.

        Args:
            state: Current system state
            **kwargs: Additional parameters (ignored)

        Returns:
            PredictionResult: Simplified prediction result
        """
        current_value = state.get_value()
        self.value_history.append(current_value)

        if len(self.value_history) < 5:
            result = PredictionResult(
                value=current_value,
                confidence=0.3,
                method=self.name
            )
            self.predictions.append(result)
            return result

        values = np.array(list(self.value_history))

        try:
            # Use last 5 points for linear fit
            if len(values) >= 5:
                x = np.arange(5)
                y = values[-5:]
                coeffs = np.polyfit(x, y, 1)
                prediction = coeffs[0] * 5 + coeffs[1]

                # Check if prediction is reasonable
                if np.isnan(prediction) or np.isinf(prediction):
                    prediction = current_value
                    confidence = 0.2
                else:
                    # Confidence based on R² of the fit
                    y_pred = coeffs[0] * x + coeffs[1]
                    ss_res = np.sum((y - y_pred) ** 2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    r2 = 1 - ss_res / (ss_tot + 1e-10)
                    confidence = 0.3 + 0.4 * max(0, r2)
            else:
                prediction = current_value
                confidence = 0.3

        except Exception as e:
            self._logger.debug(f"Linear fit failed: {e}")
            prediction = current_value
            confidence = 0.2

        confidence = float(np.clip(confidence, 0.2, 0.8))

        result = PredictionResult(
            value=float(prediction),
            confidence=confidence,
            method=self.name
        )
        self.predictions.append(result)

        return result