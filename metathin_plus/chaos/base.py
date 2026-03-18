"""
Chaos Prediction Base Classes - Supporting Custom States (Optimized Edition)
=============================================================================

This module defines the foundational abstract classes and data structures for
chaos prediction, serving as the cornerstone for all chaos-related agents.

Version: 0.2.0 (Optimized Edition)
Updates:
    - Unified predictor base class attributes and methods
    - Added comprehensive type hints
    - Introduced unified statistics interface
    - Fixed API inconsistency issues
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime
import logging

# ============================================================
# Type Variable Definitions
# ============================================================

T = TypeVar('T')
"""
Type variable T represents the system state type, which can be customized by the user.

Examples:
    - float: Single-variable systems (e.g., Logistic map)
    - Tuple[float, float]: Two-variable systems (e.g., Duffing oscillator)
    - Tuple[float, float, float]: Three-variable systems (e.g., Lorenz system)
    - Dict[str, float]: Dictionary-based state representation
    - Custom class: Any complex state representation
"""

# ============================================================
# System State Data Class
# ============================================================

@dataclass
class SystemState(Generic[T]):
    """
    System State Data Class - Generic, Supports User Customization.

    This is the core data structure of the entire chaos prediction module,
    encapsulating all information about the system at a specific moment.
    The generic type T allows users to customize the concrete representation
    of the state.

    Attributes:
        data: User-defined state data (can be numeric, tuple, dictionary, etc.)
        timestamp: Timestamp for temporal analysis
        prediction_target: Target value to predict (typically first component of data)
        metadata: Additional metadata for passing context information

    The prediction_target is automatically inferred in __post_init__ if not provided.
    """

    data: T
    """User-defined state data, type determined by generic parameter T"""

    timestamp: float = 0.0
    """Timestamp indicating when this state occurred, useful for temporal analysis"""

    prediction_target: Any = None
    """Target value to predict (typically first component of data)
       If None, automatically inferred in __post_init__"""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata for passing context information, such as:
        - Current experimental conditions
        - External disturbances
        - Custom parameters
    """

    def __post_init__(self):
        """Post-initialization processing - automatically infer prediction_target."""
        if self.prediction_target is None:
            # Safely extract prediction target
            try:
                # Dictionary type
                if isinstance(self.data, dict):
                    # Try to get first numeric value
                    for v in self.data.values():
                        if isinstance(v, (int, float)):
                            self.prediction_target = float(v)
                            break
                # Sequence type (list, tuple)
                elif hasattr(self.data, '__len__') and not isinstance(self.data, (str, bytes)):
                    if len(self.data) > 0:
                        try:
                            self.prediction_target = float(self.data[0])
                        except (TypeError, ValueError, IndexError):
                            pass
                # Single numeric value
                elif isinstance(self.data, (int, float)):
                    self.prediction_target = float(self.data)
                # Object with 'value' attribute
                elif hasattr(self.data, 'value'):
                    try:
                        self.prediction_target = float(self.data.value)
                    except (TypeError, ValueError):
                        pass
            except Exception:
                # If all attempts fail, keep as None
                pass

    def get_value(self) -> float:
        """
        Get the numeric value for prediction.

        This is the unified interface for predictors to obtain the target value,
        abstracting away the underlying data representation.

        Returns:
            float: Numeric value for prediction (typically angle, position, etc.)
        """
        if self.prediction_target is not None:
            try:
                return float(self.prediction_target)
            except (TypeError, ValueError):
                pass

        # Attempt to extract value from data
        try:
            # Dictionary type
            if isinstance(self.data, dict):
                for v in self.data.values():
                    if isinstance(v, (int, float)):
                        return float(v)
                return 0.0

            # Sequence type
            elif hasattr(self.data, '__len__') and not isinstance(self.data, (str, bytes)):
                if len(self.data) > 0:
                    return float(self.data[0])
                return 0.0

            # Single value
            else:
                return float(self.data)
        except (TypeError, ValueError, IndexError):
            return 0.0


# ============================================================
# Prediction Result Data Class
# ============================================================

@dataclass
class PredictionResult:
    """
    Prediction Result Data Class.

    Encapsulates all output information from a single prediction,
    including the predicted value, confidence, method used, etc.

    Attributes:
        value: Predicted value (numeric)
        confidence: Confidence level [0,1], higher means more reliable
        method: Name of prediction method used
        features: Features used for prediction (optional)
        error_estimate: Estimated error (optional)
        metadata: Additional metadata
    """

    value: float
    """Predicted value (numeric), the core output of prediction"""

    confidence: float = 1.0
    """Confidence level [0,1], indicating reliability of prediction
       1.0 = very confident, 0.0 = completely uncertain"""

    method: str = "unknown"
    """Name of prediction method used, for tracking and debugging"""

    features: Dict = None
    """Features used for prediction (optional), for subsequent analysis"""

    error_estimate: float = 0.0
    """Estimated error (optional), based on historical performance"""

    metadata: Dict = None
    """Additional metadata for passing context information"""

    def __post_init__(self):
        """Post-initialization processing, ensure dictionaries are not None."""
        if self.features is None:
            self.features = {}
        if self.metadata is None:
            self.metadata = {}


# ============================================================
# Chaos System Model Base Class
# ============================================================

class ChaosModel(ABC, Generic[T]):
    """
    Chaos System Model Base Class.

    Defines the abstract interface for chaotic systems. Users need to inherit
    this class to implement their own chaos models. This is the core abstraction
    connecting physical systems and machine learning.

    Methods that must be implemented:
        dynamics: System dynamics equations

    Methods that can be optionally implemented:
        extract_features: Extract features from state
        get_feature_names: Get feature names
        get_parameters: Get system parameters
    """

    @abstractmethod
    def dynamics(self, t: float, state: T) -> T:
        """
        System dynamics equations.

        Describes how the system evolves over time - the heart of a chaotic system.

        Args:
            t: Current time
            state: Current state (type determined by generic parameter T)

        Returns:
            T: State derivative (continuous systems) or next state (discrete systems)
        """
        pass

    def extract_features(self, state: SystemState[T]) -> List[float]:
        """
        Extract feature vector from state.

        These features will be used as input to predictors. The default implementation
        attempts automatic extraction, but users are encouraged to override this method
        for better feature representation.

        Args:
            state: SystemState object encapsulating current state and metadata

        Returns:
            List[float]: Feature vector for predictor input
        """
        data = state.data
        features = []

        # Handle different data types
        try:
            # Dictionary type
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, (int, float)):
                        features.append(float(v))

            # Sequence type
            elif hasattr(data, '__iter__') and not isinstance(data, (str, bytes)):
                for item in data:
                    try:
                        features.append(float(item))
                    except (TypeError, ValueError):
                        pass

            # Single value
            else:
                try:
                    features.append(float(data))
                except (TypeError, ValueError):
                    pass
        except Exception:
            pass

        # Add numeric values from metadata
        for k, v in state.metadata.items():
            if isinstance(v, (int, float)):
                features.append(float(v))

        return features if features else [0.0]

    def get_feature_names(self) -> List[str]:
        """
        Get feature names list.

        Used for debugging and visualization, helps understand the meaning of each feature.

        Returns:
            List[str]: Feature names list, should have same length as extract_features output
        """
        return []

    def get_parameters(self) -> Dict:
        """
        Get system parameters.

        Returns:
            Dict: System parameters dictionary, e.g., {'omega': 3.05, 'damping': 0.1}
        """
        return {}


# ============================================================
# State Processor Interface
# ============================================================

class StateProcessor(ABC, Generic[T]):
    """
    State Processor Interface.

    Used for custom state processing logic when ChaosModel's default implementation
    is insufficient. This interface can be viewed as a complement or alternative to ChaosModel.

    Methods that must be implemented:
        extract_features: Extract feature vector from state

    Methods that can be optionally implemented:
        compute_energy: Compute system energy
        get_feature_names: Get feature names
    """

    @abstractmethod
    def extract_features(self, state: SystemState[T]) -> np.ndarray:
        """
        Extract feature vector from state.

        Args:
            state: System state object

        Returns:
            np.ndarray: Feature vector for predictor input
        """
        pass

    def compute_energy(self, state: SystemState[T]) -> float:
        """
        Compute system energy (optional).

        Used for energy analysis to help characterize system state.

        Args:
            state: System state object

        Returns:
            float: Current energy of the system
        """
        return 0.0

    def get_feature_names(self) -> List[str]:
        """
        Get feature names list (optional).

        Returns:
            List[str]: Feature names list
        """
        return []


# ============================================================
# Chaos Predictor Base Class (Optimized Edition)
# ============================================================

class ChaosPredictor(ABC, Generic[T]):
    """
    Chaos Predictor Base Class (Optimized Edition).

    All concrete prediction methods (phase space, Volterra, neural networks, etc.)
    must inherit from this class. Provides unified history tracking, error logging,
    statistics collection, and other foundational functionality.

    Attributes:
        name: Predictor name
        history: Historical state records
        predictions: Historical prediction results
        errors: Historical prediction errors
        value_history: Historical numeric values (for quick access)
        restart_count: Number of restarts (specific to some predictors)

    Methods that must be implemented:
        predict: Perform single-step prediction

    Methods that can be optionally implemented:
        update: Update with actual value (for learning)
        reset: Reset predictor state
    """

    def __init__(self, name: str):
        """
        Initialize predictor.

        Args:
            name: Predictor name for identification and logging
        """
        self.name = name
        """Predictor name for identification and logging"""

        self.history: List[SystemState[T]] = []
        """Historical state records, storing all processed states"""

        self.predictions: List[PredictionResult] = []
        """Historical prediction results, storing all prediction outputs"""

        self.errors: List[float] = []
        """Historical prediction errors, for performance evaluation and learning"""

        self.value_history: List[float] = []
        """Historical numeric values, for quick access to prediction targets"""

        self.restart_count: int = 0
        """Number of restarts (specific to some predictors), recording algorithm resets"""

        self._logger = logging.getLogger(f"metathin_plus.chaos.predictors.{name}")

    @property
    def total_predictions(self) -> int:
        """
        Total number of predictions made (unified attribute).

        Returns:
            int: Number of predictions performed
        """
        return len(self.predictions)

    @property
    def total_errors(self) -> int:
        """
        Total number of error records (unified attribute).

        Returns:
            int: Number of errors recorded
        """
        return len(self.errors)

    @abstractmethod
    def predict(self, state: SystemState[T], **kwargs) -> PredictionResult:
        """
        Perform single-step prediction.

        This is the core method of the predictor, must be implemented by subclasses.

        Args:
            state: Current system state
            **kwargs: Additional parameters, such as time step, context information

        Returns:
            PredictionResult: Prediction result containing value, confidence, etc.
        """
        pass

    def update(self, actual: SystemState[T]):
        """
        Update with actual observed value (for learning).

        When actual observations become available, call this method to update
        internal state and error records.

        Args:
            actual: Actually observed state
        """
        if self.predictions and actual:
            last_pred = self.predictions[-1]
            actual_value = actual.get_value()
            error = abs(last_pred.value - actual_value)
            self.errors.append(error)
            self.value_history.append(actual_value)

    def get_recent_error(self, window: int = 10) -> float:
        """
        Get recent average error.

        Used to evaluate the predictor's recent performance, often used in
        adaptive switching decisions.

        Args:
            window: Window size for averaging recent errors

        Returns:
            float: Recent average error, returns 1.0 if insufficient data
        """
        if len(self.errors) > window:
            return float(np.mean(self.errors[-window:]))
        elif self.errors:
            return float(np.mean(self.errors))
        else:
            return 1.0  # Maximum value when no data

    def get_stats(self) -> Dict[str, Any]:
        """
        Get predictor statistics (unified interface).

        Returns a dictionary containing predictor runtime status and performance metrics.

        Returns:
            Dict[str, Any]: Statistics dictionary containing:
                - name: Predictor name
                - total_predictions: Total number of predictions
                - total_errors: Total number of error records
                - recent_error: Recent average error
                - history_size: Number of historical states
                - predictions_size: Number of historical predictions
                - errors_size: Number of historical errors
                - restart_count: Number of restarts (if applicable)
        """
        stats = {
            'name': self.name,
            'total_predictions': self.total_predictions,
            'total_errors': self.total_errors,
            'recent_error': self.get_recent_error(10),
            'history_size': len(self.history),
            'predictions_size': len(self.predictions),
            'errors_size': len(self.errors),
        }

        # Add restart count if it exists
        if hasattr(self, 'restart_count'):
            stats['restart_count'] = self.restart_count

        return stats

    def reset(self):
        """Reset predictor, clearing all historical records."""
        self.history = []
        self.predictions = []
        self.errors = []
        self.value_history = []
        self.restart_count = 0
        self._logger.debug(f"Predictor {self.name} reset")