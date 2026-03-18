"""
MetaChaosBasic - Universal Chaos Prediction Agent (Optimized Edition)
======================================================================

This is the core agent of Metathin+, specifically designed for chaotic time series prediction.
It integrates multiple prediction methods, adaptively selects the optimal method based on
system state, and continuously optimizes the selection strategy through learning mechanisms.

Version: 0.2.0 (Optimized Edition)
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, TypeVar, Generic
from collections import deque
import time
import logging

# ============================================================
# Metathin Core Imports
# ============================================================
try:
    from metathin import Metathin, MetathinConfig
    from metathin.core import MemoryManager
    from metathin.core.interfaces import FeatureVector
    METATHIN_AVAILABLE = True
except ImportError as e:
    METATHIN_AVAILABLE = False
    print(f"Warning: Metathin core not found ({e}), using simplified mode")
    FeatureVector = np.ndarray

# ============================================================
# Local Module Component Imports
# ============================================================

from .base import ChaosModel, SystemState, PredictionResult, ChaosPredictor, StateProcessor, T
from .predictors import (
    PhaseSpacePredictor,    # Phase space reconstruction
    VolterraPredictor,      # Volterra series
    NeuralPredictor,        # Neural network
    SpectralPredictor,      # Spectral analysis
    PersistentPredictor,    # Persistent prediction (baseline)
    LinearTrendPredictor,   # Linear trend
    FullPhaseSpacePredictor,
    FullVolterraPredictor,
    FullNeuralPredictor,
    FullSpectralPredictor
)
from .analyzers import LyapunovEstimator, DimensionEstimator

# ============================================================
# Type Definitions
# ============================================================

from typing import TypedDict, Optional

class ChaosStatusDict(TypedDict):
    """Type definition for chaos agent status dictionary."""
    name: str
    total_predictions: int
    current_method: Optional[str]
    last_error: Optional[float]
    avg_error: Optional[float]
    error_count: int
    method_usage: Dict[str, int]
    switches: int
    buffer_size: int
    buffer_capacity: int
    lyapunov_exponent: Optional[float]
    correlation_dimension: Optional[float]


# ============================================================
# Main Agent Class (Optimized Edition - Inherits from Metathin)
# ============================================================

class MetaChaosBasic(Metathin if METATHIN_AVAILABLE else object, Generic[T]):
    """
    Universal Chaos Prediction Agent - Supports Custom State Types (Optimized Edition)

    This agent orchestrates multiple chaos prediction methods, providing:
        - Adaptive method selection based on recent performance
        - Integrated learning capabilities (when Metathin core available)
        - Chaos-specific analysis (Lyapunov exponents, correlation dimension)
        - Unified prediction interface for various system types
    """

    def __init__(self,
                 model: Optional[ChaosModel[T]] = None,
                 state_processor: Optional[StateProcessor[T]] = None,
                 feature_extractor: Optional[Callable[[SystemState[T]], List[float]]] = None,
                 error_threshold: float = 0.5,
                 methods: Optional[List[str]] = None,
                 enable_learning: bool = True,
                 memory_size: int = 1000,
                 name: str = "MetaChaosBasic",
                 **kwargs):
        """
        Initialize Chaos Prediction Agent.

        Args:
            model: Chaos system model (one of three options must be provided)
            state_processor: State processor (one of three options must be provided)
            feature_extractor: Feature extraction function (one of three options must be provided)
            error_threshold: Error switching threshold - switch methods when error exceeds this
            methods: List of prediction methods to use, None means use all available
            enable_learning: Whether to enable learning functionality (if Metathin available)
            memory_size: State buffer size for history tracking
            name: Agent name for identification and logging
            **kwargs: Additional parameters to pass to Metathin base class

        Raises:
            ValueError: If none of model, state_processor, or feature_extractor is provided
        """
        # Validate input
        if model is None and state_processor is None and feature_extractor is None:
            raise ValueError("Must provide one of: model, state_processor, or feature_extractor")

        # ===== Step 1: Initialize basic attributes (self.name must exist first) =====
        self.name = name  # Critical: name must be set before anything else
        self.model = model
        self.state_processor = state_processor
        self.feature_extractor = feature_extractor
        self.error_threshold = error_threshold

        # Create logger (now that name exists)
        self._logger = logging.getLogger(f"metathin_plus.chaos.{self.name}")

        # ===== Initialize feature names =====
        self.feature_names = self._init_feature_names()
        self._logger.debug(f"Feature names: {self.feature_names}")

        # ===== Step 2: Create pattern space for Metathin base class =====
        pattern_space = self._create_pattern_space()

        # ===== Step 3: Initialize Metathin base class (if available) =====
        if METATHIN_AVAILABLE:
            config = MetathinConfig(
                enable_learning=enable_learning,
                enable_memory=True,
                **kwargs
            )
            super().__init__(
                pattern_space=pattern_space,
                name=self.name,  # Use the already-assigned self.name
                config=config
            )

        # ===== Step 4: Chaos-specific initialization =====
        self._init_chaos_components(methods, memory_size)

        self._logger.info(f"✅ MetaChaosBasic initialized successfully")
        self._logger.info(f"   Prediction methods: {[p.name for p in self.predictors]}")
        self._logger.info(f"   Feature dimension: {len(self.feature_names)}")
        self._logger.info(f"   Error threshold: {error_threshold}")

    def _init_feature_names(self) -> List[str]:
        """
        Initialize feature names.

        Retrieves feature names from model, state_processor, or generates defaults.

        Returns:
            List[str]: List of feature names
        """
        names = []

        # Get from model
        if self.model and hasattr(self.model, 'get_feature_names'):
            model_names = self.model.get_feature_names()
            if model_names:
                names.extend(model_names)

        # Get from processor
        if self.state_processor and hasattr(self.state_processor, 'get_feature_names'):
            processor_names = self.state_processor.get_feature_names()
            if processor_names:
                names.extend(processor_names)

        # Generate default names
        if not names:
            # Try to infer feature dimension
            try:
                # Create temporary state to get feature dimension
                if self.model:
                    temp_state = SystemState(data=0.0)
                    features = self.model.extract_features(temp_state)
                    dim = len(features)
                else:
                    dim = 5  # Default dimension
            except:
                dim = 5  # Default dimension

            names = [f'feature_{i}' for i in range(dim)]

        return names

    def _init_chaos_components(self, methods: Optional[List[str]], memory_size: int):
        """
        Initialize chaos-specific components.

        Args:
            methods: List of prediction methods to use
            memory_size: Buffer size for state history
        """
        # 1. Initialize prediction methods
        self.predictors = self._init_predictors(methods)
        self.predictor_dict = {p.name: p for p in self.predictors}

        # 2. Initialize chaos analyzers
        self.lyapunov_estimator = LyapunovEstimator()
        self.dimension_estimator = DimensionEstimator()

        # 3. Initialize state buffers
        self.state_buffer: deque = deque(maxlen=memory_size)
        self.value_buffer: deque = deque(maxlen=memory_size)

        # 4. Initialize statistics
        self.stats = {
            'total_predictions': 0,
            'method_usage': {},
            'errors': [],
            'switches': 0,
            'last_error': 0.0,
            'start_time': time.time()
        }

        # 5. Initialize method selection state
        self.current_method = None
        self.method_fitness = {p.name: 0.5 for p in self.predictors}

    def _create_pattern_space(self):
        """Create pattern space for Metathin base class."""
        if not METATHIN_AVAILABLE:
            return None

        from metathin.components import SimplePatternSpace

        def extract_features(raw_input):
            if isinstance(raw_input, SystemState):
                return self._extract_features(raw_input)
            else:
                # If not SystemState, attempt to wrap it
                state = SystemState(data=raw_input)
                return self._extract_features(state)

        # self.name is now guaranteed to exist
        return SimplePatternSpace(
            extract_func=extract_features,
            name=f"{self.name}_pattern"
        )

    def _init_predictors(self, methods: Optional[List[str]]) -> List[ChaosPredictor]:
        """
        Initialize prediction methods.

        Creates predictor instances based on user-specified method list.

        Args:
            methods: List of method names, None means use all available

        Returns:
            List[ChaosPredictor]: List of predictor instances
        """
        all_methods = {
            'phase_space': FullPhaseSpacePredictor(),   # Use full version
            'volterra': FullVolterraPredictor(),       # Use full version
            'neural': FullNeuralPredictor(),           # Use full version
            'spectral': FullSpectralPredictor(),       # Use full version
            'persistent': PersistentPredictor(),
            'linear_trend': LinearTrendPredictor()
        }

        if methods is None:
            return list(all_methods.values())
        else:
            return [all_methods[m] for m in methods if m in all_methods]

    def _extract_features(self, state: SystemState[T]) -> np.ndarray:
        """
        Extract features from state.

        Priority order for feature extraction:
            1. Custom feature extractor (if provided)
            2. State processor (if provided)
            3. Model (if provided)
            4. Automatic extraction (fallback)

        Args:
            state: System state object

        Returns:
            np.ndarray: Feature vector
        """
        features = []

        # 1. Use custom feature extractor
        if self.feature_extractor is not None:
            try:
                feats = self.feature_extractor(state)
                if isinstance(feats, (list, tuple, np.ndarray)):
                    features.extend(feats)
            except Exception as e:
                self._logger.warning(f"Custom feature extraction failed: {e}")

        # 2. Use state processor
        elif self.state_processor is not None:
            try:
                feats = self.state_processor.extract_features(state)
                if isinstance(feats, np.ndarray):
                    features.extend(feats.tolist())
                elif isinstance(feats, (list, tuple)):
                    features.extend(feats)
            except Exception as e:
                self._logger.warning(f"State processor feature extraction failed: {e}")

        # 3. Use model
        elif self.model is not None:
            try:
                feats = self.model.extract_features(state)
                if isinstance(feats, (list, tuple, np.ndarray)):
                    features.extend(feats)
            except Exception as e:
                self._logger.warning(f"Model feature extraction failed: {e}")

        # 4. Automatic extraction (fallback)
        if not features:
            try:
                features.append(state.get_value())
            except:
                features.append(0.0)

            # Add numeric values from metadata
            for k, v in state.metadata.items():
                if isinstance(v, (int, float)):
                    features.append(float(v))

        # Ensure features exist
        if not features:
            features = [0.0]

        return np.array(features, dtype=np.float64)

    def _select_method(self, features: np.ndarray) -> str:
        """
        Select prediction method.

        Computes fitness based on recent error of each method and selects the best.
        Switches method if current method's error exceeds threshold.

        Args:
            features: Current feature vector (for future extensions)

        Returns:
            str: Name of selected method
        """
        # Update fitness for each method
        for predictor in self.predictors:
            recent_error = predictor.get_recent_error()
            if recent_error < self.error_threshold:
                fitness = 0.5 + 0.3 * (1 - recent_error / self.error_threshold)
            else:
                fitness = 0.3
            self.method_fitness[predictor.name] = fitness

        # Select method with highest fitness
        best_method = max(self.method_fitness.items(), key=lambda x: x[1])[0]

        # Switch if current method's error is too high
        if self.current_method:
            current_predictor = self.predictor_dict.get(self.current_method)
            if current_predictor and current_predictor.get_recent_error() > self.error_threshold:
                self.stats['switches'] += 1
                self._logger.info(f"Switching prediction mode: {self.current_method} -> {best_method}")
                self.current_method = best_method
        else:
            # First-time selection
            self.current_method = best_method

        return self.current_method

    def predict(self,
                state: Union[SystemState[T], T, Any],
                timestamp: Optional[float] = None,
                **kwargs) -> PredictionResult:
        """
        Predict next value.

        Core method of the agent, executing the complete prediction pipeline:
            1. Normalize input to SystemState
            2. Update state buffer
            3. Extract features
            4. Select optimal method
            5. Execute prediction
            6. Update statistics

        Args:
            state: Current state (can be SystemState, raw data, or any type)
            timestamp: Timestamp (used when state is not SystemState)
            **kwargs: Additional parameters, stored in metadata

        Returns:
            PredictionResult: Prediction result containing value, method, etc.
        """
        # 1. Ensure state is SystemState
        if not isinstance(state, SystemState):
            state = SystemState(
                data=state,
                timestamp=timestamp or time.time(),
                metadata=kwargs
            )

        # 2. Get target value
        target_value = state.get_value()

        # 3. Update buffers
        self.state_buffer.append(state)
        self.value_buffer.append(target_value)

        # 4. Update all predictors' history
        for predictor in self.predictors:
            predictor.history.append(state)

        # 5. Extract features
        features = self._extract_features(state)

        # 6. Select method
        method_name = self._select_method(features)
        selected_predictor = self.predictor_dict.get(method_name)

        # 7. Execute prediction
        if selected_predictor:
            result = selected_predictor.predict(state, **kwargs)
        else:
            # Fallback: return current value as prediction
            result = PredictionResult(
                value=target_value,
                method="fallback"
            )

        # 8. Update statistics
        self.stats['total_predictions'] += 1
        self.stats['method_usage'][method_name] = self.stats['method_usage'].get(method_name, 0) + 1

        # 9. Update predictor's prediction records
        for predictor in self.predictors:
            if predictor.name == method_name:
                predictor.predictions.append(result)

        return result

    def update(self, actual: Union[SystemState[T], T, Any], timestamp: Optional[float] = None):
        """
        Update with actual observed value for learning and evaluation.

        Called when actual observations become available. Used for:
            - Computing prediction error
            - Updating each predictor's learning
            - Updating statistics

        Args:
            actual: Actually observed state
            timestamp: Timestamp (used when actual is not SystemState)
        """
        # Ensure actual is SystemState
        if not isinstance(actual, SystemState):
            actual = SystemState(
                data=actual,
                timestamp=timestamp or time.time()
            )

        actual_value = actual.get_value()

        # Compute error
        if self.stats['total_predictions'] > 0:
            last_prediction = None
            last_method = None

            # Find most recent prediction from any predictor
            for name, predictor in self.predictor_dict.items():
                if predictor.predictions:
                    last_pred = predictor.predictions[-1]
                    if last_pred:
                        last_prediction = last_pred.value
                        last_method = name
                        break

            if last_prediction is not None:
                error = abs(last_prediction - actual_value)
                self.stats['errors'].append(error)
                self.stats['last_error'] = error

                # Update corresponding predictor
                if last_method and last_method in self.predictor_dict:
                    self.predictor_dict[last_method].update(actual)

    def get_status(self) -> ChaosStatusDict:
        """
        Get agent status (unified interface).

        Returns a dictionary containing agent runtime status and performance metrics.

        Returns:
            ChaosStatusDict: Status information dictionary
        """
        # Calculate average error
        avg_error = None
        if self.stats['errors']:
            avg_error = float(np.mean(self.stats['errors'][-100:]))

        # Calculate Lyapunov exponent and correlation dimension (if sufficient data)
        le = None
        cd = None
        if len(self.value_buffer) > 50:
            data = list(self.value_buffer)
            le = self.lyapunov_estimator.estimate(data)
            cd = self.dimension_estimator.estimate(data)

        return {
            'name': self.name,
            'total_predictions': self.stats['total_predictions'],
            'current_method': self.current_method,
            'last_error': self.stats.get('last_error'),
            'avg_error': avg_error,
            'error_count': len(self.stats['errors']),
            'method_usage': self.stats['method_usage'].copy(),
            'switches': self.stats['switches'],
            'buffer_size': len(self.state_buffer),
            'buffer_capacity': self.state_buffer.maxlen,
            'lyapunov_exponent': le,
            'correlation_dimension': cd,
        }

    def reset(self):
        """
        Reset predictor.

        Clears all buffers, statistics, and history, returning to initial state.
        """
        self.state_buffer.clear()
        self.value_buffer.clear()
        self.stats = {
            'total_predictions': 0,
            'method_usage': {},
            'errors': [],
            'switches': 0,
            'last_error': 0.0,
            'start_time': time.time()
        }
        self.current_method = None
        self.method_fitness = {p.name: 0.5 for p in self.predictors}

        # Reset all predictors
        for predictor in self.predictors:
            predictor.reset()

        self._logger.info("🔄 Predictor reset")