"""
Selector Components
===================

Provides various fitness calculators that evaluate the suitability of behaviors in the current state.
The selector serves as the agent's "evaluator," determining how appropriate each behavior is in the given context.

Design Philosophy:
    - Learnability: Selector parameters can be adjusted by learning mechanisms for adaptation
    - Diversity: Supports linear, polynomial, rule-based, and other evaluation methods
    - Composability: Multiple selectors can be combined for improved robustness
    - Stability: Numerically stable computations prevent overflow

The selector outputs α(Bi) ∈ [0,1], representing the fitness of behavior Bi.
Higher fitness indicates the behavior is more suitable for the current state.

Selector Types:
    - SimpleSelector: Linear weighted sum + sigmoid, learnable
    - PolynomialSelector: Polynomial regression, captures nonlinear relationships
    - RuleBasedSelector: Expert system based on predefined rules
    - EnsembleSelector: Combines multiple selectors
    - AdaptiveSelector: Dynamically switches between selectors
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
from abc import ABC, abstractmethod
import logging
import warnings

from ..core.interfaces import Selector, MetaBehavior, FeatureVector, FitnessScore, ParameterDict
from ..core.interfaces import FitnessComputationError, ParameterUpdateError


# ============================================================
# Helper Functions
# ============================================================

def sigmoid(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Safe sigmoid function.
    
    Maps any real number to the (0,1) interval, commonly used to convert linear combinations to probabilities.
    
    Args:
        x: Input array
        temperature: Temperature parameter, higher values produce smoother outputs
        
    Returns:
        np.ndarray: Sigmoid output in (0,1)
    """
    x = np.clip(x / temperature, -500, 500)  # Prevent overflow
    return 1.0 / (1.0 + np.exp(-x))


def normalize_scores(scores: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Normalize scores to [0,1] range.
    
    Linearly maps arbitrary scores to the [0,1] interval.
    
    Args:
        scores: Input scores
        epsilon: Small value to prevent division by zero
        
    Returns:
        np.ndarray: Normalized scores
    """
    min_score = np.min(scores)
    max_score = np.max(scores)
    
    if max_score - min_score < epsilon:
        return np.ones_like(scores) * 0.5
    
    return (scores - min_score) / (max_score - min_score + epsilon)


# ============================================================
# Simple Selector
# ============================================================

class SimpleSelector(Selector):
    """
    Simple Selector: Computes fitness based on weighted sum.
    
    Uses linear weighted sum: α = sigmoid(w·x + b)
    Supports dynamic expansion for new behaviors, suitable for most scenarios.
    
    Characteristics:
        - Learnable: Weights and bias can be adjusted by learning mechanisms
        - Dynamic expansion: Automatically expands weight matrix when new behaviors are added
        - Temperature control: Temperature parameter adjusts output smoothness
    
    Attributes:
        temperature: Temperature parameter, higher values produce smoother outputs
        max_weight: Maximum weight value to prevent gradient explosion
        weights: Weight matrix [n_behaviors, n_features]
        bias: Bias vector [n_behaviors]
    """
    
    def __init__(self, 
                 n_features: Optional[int] = None,
                 n_behaviors: Optional[int] = None,
                 temperature: float = 2.0,
                 max_weight: float = 10.0,
                 init_scale: float = 0.1):
        """
        Initialize simple selector.
        
        Args:
            n_features: Number of features, None for dynamic determination
            n_behaviors: Number of behaviors, None for dynamic determination
            temperature: Temperature parameter, higher values produce smoother outputs
            max_weight: Maximum weight value to prevent gradient explosion
            init_scale: Weight initialization scaling factor
            
        Raises:
            ValueError: If parameters are out of valid range
        """
        super().__init__()
        
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        
        if max_weight <= 0:
            raise ValueError(f"max_weight must be > 0, got {max_weight}")
        
        self.temperature = temperature
        self.max_weight = max_weight
        self.init_scale = init_scale
        self._logger = logging.getLogger("metathin.selector.SimpleSelector")
        
        # Behavior index mapping
        self._behavior_indices: Dict[str, int] = {}
        
        # Initialize weight matrix
        if n_features is not None and n_behaviors is not None:
            if n_features <= 0 or n_behaviors <= 0:
                raise ValueError("Number of features and behaviors must be positive")
            
            self.weights = np.random.randn(n_behaviors, n_features) * init_scale
            self.bias = np.zeros(n_behaviors)
            self._feature_dim = n_features
            self._logger.debug(f"Initialized weight matrix: {n_behaviors} x {n_features}")
        else:
            self.weights = None
            self.bias = None
            self._feature_dim = None
            self._logger.debug("Using dynamic expansion mode")
        
        # Parameter update history
        self._update_history: List[Dict[str, float]] = []
    
    def _get_or_create_index(self, behavior_name: str, feature_dim: int) -> int:
        """
        Get or create behavior index.
        
        Automatically expands weight matrix when new behaviors appear.
        
        Args:
            behavior_name: Name of the behavior
            feature_dim: Feature dimension
            
        Returns:
            int: Index of the behavior
        """
        if behavior_name in self._behavior_indices:
            return self._behavior_indices[behavior_name]
        
        idx = len(self._behavior_indices)
        self._behavior_indices[behavior_name] = idx
        
        self._logger.debug(f"Assigning index {idx} to new behavior '{behavior_name}'")
        
        # Dynamically expand weight matrix
        if self.weights is None:
            # First creation
            self.weights = np.random.randn(idx + 1, feature_dim) * self.init_scale
            self.bias = np.zeros(idx + 1)
            self._feature_dim = feature_dim
            
        elif idx >= len(self.weights):
            # Need expansion
            old_shape = self.weights.shape
            new_weights = np.vstack([
                self.weights,
                np.random.randn(idx + 1 - len(self.weights), self.weights.shape[1]) * self.init_scale
            ])
            self.weights = new_weights
            
            # Expand bias
            new_bias = np.append(self.bias, np.zeros(idx + 1 - len(self.bias)))
            self.bias = new_bias
        
        return idx
    
    def compute_fitness(self, behavior: MetaBehavior, features: FeatureVector) -> FitnessScore:
        """
        Compute fitness: α = sigmoid((w·x + b) / temperature)
        
        Args:
            behavior: Behavior to evaluate
            features: Feature vector
            
        Returns:
            FitnessScore: Fitness value in [0,1]
            
        Raises:
            FitnessComputationError: If fitness computation fails
        """
        try:
            # Validate features
            if not isinstance(features, np.ndarray):
                features = np.array(features, dtype=np.float64)
            
            if features.ndim > 1:
                features = features.flatten()
            
            # Get or create behavior index
            idx = self._get_or_create_index(behavior.name, len(features))
            
            # Check index validity
            if idx >= len(self.weights):
                raise FitnessComputationError(f"Behavior index {idx} exceeds weight matrix size ({len(self.weights)})")
            
            # Compute linear combination
            z = np.dot(self.weights[idx], features)
            
            if idx < len(self.bias):
                z += self.bias[idx]
            
            # Apply temperature
            z = z / self.temperature
            
            # Compute sigmoid
            fitness = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
            fitness = float(np.clip(fitness, 0.0, 1.0))
            
            # Record fitness
            self.record_fitness(behavior.name, fitness)
            
            return fitness
            
        except Exception as e:
            self._logger.error(f"Fitness computation failed: {e}")
            raise FitnessComputationError(f"Fitness computation failed: {e}") from e
    
    def get_parameters(self) -> ParameterDict:
        """Get current learnable parameters."""
        params = {}
        
        if self.weights is not None:
            for i in range(len(self.weights)):
                for j in range(self.weights.shape[1]):
                    params[f'w_{i}_{j}'] = float(self.weights[i, j])
        
        if self.bias is not None:
            for i in range(len(self.bias)):
                params[f'b_{i}'] = float(self.bias[i])
        
        return params
    
    def update_parameters(self, delta: ParameterDict) -> None:
        """
        Update parameters during learning.
        
        Args:
            delta: Parameter adjustments
            
        Raises:
            ParameterUpdateError: If parameter update fails
        """
        try:
            update_record = {}
            
            for key, value in delta.items():
                if key.startswith('w_'):
                    parts = key.split('_')
                    if len(parts) == 3:
                        _, i_str, j_str = parts
                        try:
                            i, j = int(i_str), int(j_str)
                            
                            if self.weights is not None and i < len(self.weights) and j < self.weights.shape[1]:
                                old_val = self.weights[i, j]
                                new_val = old_val + value
                                self.weights[i, j] = np.clip(new_val, -self.max_weight, self.max_weight)
                        except ValueError:
                            pass
                
                elif key.startswith('b_'):
                    parts = key.split('_')
                    if len(parts) == 2:
                        _, i_str = parts
                        try:
                            i = int(i_str)
                            if self.bias is not None and i < len(self.bias):
                                self.bias[i] += value
                        except ValueError:
                            pass
        except Exception as e:
            raise ParameterUpdateError(f"Parameter update failed: {e}") from e


# ============================================================
# Polynomial Selector (Enhanced Version)
# ============================================================

class PolynomialSelector(Selector):
    """
    Polynomial Selector: Uses polynomial regression for fitness computation (enhanced version).
    
    Supports linear terms, square terms, cubic terms, and interaction terms,
    capturing nonlinear relationships between features.
    
    Characteristics:
        - Nonlinear: Captures relationships between features through polynomial expansion
        - Interaction terms: Learns interactions between features
        - Regularization: Prevents overfitting
        - Numerically stable: Feature normalization prevents overflow
    
    Attributes:
        degree: Polynomial degree (1-3)
        include_interaction: Whether to include interaction terms
        include_bias: Whether to include bias term
        regularization: L2 regularization coefficient
        feature_means: Feature means for normalization
        feature_stds: Feature standard deviations for normalization
        
    Example:
        >>> # Create quadratic polynomial selector
        >>> selector = PolynomialSelector(
        ...     degree=2,
        ...     n_features=3,
        ...     n_behaviors=2,
        ...     include_interaction=True
        ... )
        >>> 
        >>> # Can capture interactions between features
        >>> fitness = selector.compute_fitness(behavior, features)
    """
    
    def __init__(self, 
                 degree: int = 2,
                 n_features: Optional[int] = None,
                 n_behaviors: Optional[int] = None,
                 include_interaction: bool = True,
                 include_bias: bool = True,
                 regularization: float = 0.01,
                 normalize_features: bool = True):
        """
        Initialize polynomial selector (enhanced version).
        
        Args:
            degree: Polynomial degree, supports 1-3
            n_features: Number of original features
            n_behaviors: Number of behaviors
            include_interaction: Whether to include interaction terms
            include_bias: Whether to include bias term
            regularization: L2 regularization coefficient
            normalize_features: Whether to normalize features
            
        Raises:
            ValueError: If degree is out of valid range
        """
        super().__init__()
        
        if degree < 1 or degree > 3:
            raise ValueError(f"degree must be between 1 and 3, got {degree}")
        
        self.degree = degree
        self.include_interaction = include_interaction
        self.include_bias = include_bias
        self.regularization = regularization
        self.normalize_features = normalize_features
        self._logger = logging.getLogger("metathin.selector.PolynomialSelector")
        
        # Behavior index mapping
        self._behavior_indices: Dict[str, int] = {}
        
        # Feature normalization statistics
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None
        
        # Count polynomial features
        self.n_poly = 0
        if n_features:
            self.n_poly = self._count_poly_features(n_features)
            self._feature_dim = n_features
        else:
            self._feature_dim = None
        
        # Initialize weights
        if n_behaviors and self.n_poly:
            scale = 1.0 / np.sqrt(self.n_poly)
            self.weights = np.random.randn(n_behaviors, self.n_poly) * scale
            self._logger.debug(f"Initialized polynomial weights: {n_behaviors} x {self.n_poly}")
        else:
            self.weights = None
        
        # Parameter update history
        self._update_history: List[Dict[str, float]] = []
    
    def _count_poly_features(self, n: int) -> int:
        """
        Count polynomial features (enhanced version).
        
        Includes:
            - Linear terms: n
            - Quadratic terms: n (squares) + n*(n-1)/2 (interactions, if enabled)
            - Cubic terms: n (cubes) + n*(n-1) (square*linear) + n*(n-1)*(n-2)/6 (cubic interactions)
        """
        count = n  # Linear terms
        
        if self.degree >= 2:
            count += n  # Square terms
            if self.include_interaction:
                count += n * (n - 1) // 2  # Quadratic interactions
        
        if self.degree >= 3:
            count += n  # Cube terms
            if self.include_interaction:
                count += n * (n - 1)  # Square * linear (e.g., x₁² * x₂)
                count += n * (n - 1) * (n - 2) // 6  # Cubic interactions (e.g., x₁ * x₂ * x₃)
        
        if self.include_bias:
            count += 1  # Bias term
        
        return count
    
    def _normalize_features(self, features: np.ndarray, update_stats: bool = False) -> np.ndarray:
        """
        Normalize features to prevent numerical overflow.
        
        Args:
            features: Raw features
            update_stats: Whether to update running statistics
            
        Returns:
            np.ndarray: Normalized features
        """
        if not self.normalize_features:
            return features
        
        features = features.astype(np.float64)
        
        if update_stats:
            if self.feature_means is None:
                self.feature_means = np.mean(features)
                self.feature_stds = np.std(features) + 1e-8
            else:
                # Online update of mean and std
                n = len(features)
                delta = features - self.feature_means
                self.feature_means = self.feature_means + delta / n
                self.feature_stds = np.sqrt(
                    (self.feature_stds**2 * (n-1) + delta**2) / n
                ) + 1e-8
        
        if self.feature_means is not None and self.feature_stds is not None:
            features = (features - self.feature_means) / self.feature_stds
        
        return features
    
    def _expand_features(self, features: np.ndarray) -> np.ndarray:
        """
        Expand to polynomial features (enhanced version).
        
        Supports 1-3 degree polynomial expansion, including interaction terms.
        
        Args:
            features: Original feature vector
            
        Returns:
            np.ndarray: Expanded polynomial feature vector
        """
        n = len(features)
        expanded = []
        
        # 1. Linear terms
        expanded.extend(features)
        
        # 2. Quadratic terms
        if self.degree >= 2:
            # Square terms
            expanded.extend(features ** 2)
            
            # Interaction terms
            if self.include_interaction:
                for i in range(n):
                    for j in range(i + 1, n):
                        expanded.append(features[i] * features[j])
        
        # 3. Cubic terms
        if self.degree >= 3:
            # Cube terms
            expanded.extend(features ** 3)
            
            # Interaction terms
            if self.include_interaction:
                # Square * linear (x_i² * x_j)
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            expanded.append((features[i] ** 2) * features[j])
                
                # Cubic interactions (x_i * x_j * x_k)
                for i in range(n):
                    for j in range(i + 1, n):
                        for k in range(j + 1, n):
                            expanded.append(features[i] * features[j] * features[k])
        
        # 4. Bias term
        if self.include_bias:
            expanded.append(1.0)
        
        return np.array(expanded, dtype=np.float64)
    
    def _get_or_create_index(self, behavior_name: str) -> int:
        """Get or create behavior index."""
        if behavior_name in self._behavior_indices:
            return self._behavior_indices[behavior_name]
        
        idx = len(self._behavior_indices)
        self._behavior_indices[behavior_name] = idx
        return idx
    
    def compute_fitness(self, behavior: MetaBehavior, features: FeatureVector) -> FitnessScore:
        """
        Compute polynomial fitness (enhanced version).
        
        Complete pipeline:
            1. Feature normalization
            2. Polynomial expansion
            3. Weighted sum
            4. Regularization
            5. Sigmoid mapping
        
        Args:
            behavior: Behavior to evaluate
            features: Feature vector
            
        Returns:
            FitnessScore: Fitness value in [0,1]
        """
        try:
            # Get or create behavior index
            name = behavior.name
            idx = self._get_or_create_index(name)
            
            # Normalize features
            features_norm = self._normalize_features(features, update_stats=True)
            
            # Expand to polynomial features
            poly_features = self._expand_features(features_norm)
            
            # Dynamically expand weights
            if self.weights is None:
                self.n_poly = len(poly_features)
                self.weights = np.random.randn(idx + 1, self.n_poly) * 0.1
                self._feature_dim = len(features)
            elif idx >= len(self.weights):
                new_weights = np.vstack([
                    self.weights,
                    np.random.randn(idx + 1 - len(self.weights), self.weights.shape[1]) * 0.1
                ])
                self.weights = new_weights
            
            # Ensure dimension match
            if len(poly_features) != self.weights.shape[1]:
                if len(poly_features) > self.weights.shape[1]:
                    poly_features = poly_features[:self.weights.shape[1]]
                else:
                    poly_features = np.pad(poly_features, 
                                         (0, self.weights.shape[1] - len(poly_features)),
                                         mode='constant', constant_values=0)
            
            # Compute weighted sum
            z = np.dot(self.weights[idx], poly_features)
            
            # Apply L2 regularization
            z -= self.regularization * np.sum(self.weights[idx] ** 2)
            
            # Apply sigmoid with temperature control
            fitness = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
            fitness = float(np.clip(fitness, 0.0, 1.0))
            
            # Record fitness
            self.record_fitness(behavior.name, fitness)
            
            return fitness
            
        except Exception as e:
            self._logger.error(f"Polynomial fitness computation failed: {e}")
            raise FitnessComputationError(f"Polynomial fitness computation failed: {e}") from e
    
    def get_parameters(self) -> ParameterDict:
        """Get parameters."""
        params = {}
        if self.weights is not None:
            for i in range(len(self.weights)):
                for j in range(self.weights.shape[1]):
                    params[f'poly_w_{i}_{j}'] = float(self.weights[i, j])
        return params
    
    def update_parameters(self, delta: ParameterDict) -> None:
        """
        Update parameters.
        
        Args:
            delta: Parameter adjustments
            
        Raises:
            ParameterUpdateError: If parameter update fails
        """
        try:
            for key, value in delta.items():
                if key.startswith('poly_w_'):
                    parts = key.split('_')
                    if len(parts) == 4:
                        _, _, i_str, j_str = parts
                        try:
                            i, j = int(i_str), int(j_str)
                            if self.weights is not None and i < len(self.weights) and j < self.weights.shape[1]:
                                self.weights[i, j] += value
                        except ValueError:
                            pass
        except Exception as e:
            raise ParameterUpdateError(f"Polynomial parameter update failed: {e}") from e
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance (new feature).
        
        Analyzes the contribution of each original feature to decisions.
        
        Returns:
            Dict[str, float]: Mapping from feature names to importance values
        """
        if self.weights is None or self.weights.shape[1] == 0:
            return {}
        
        # Calculate average weight contribution for each original feature
        n_orig_features = self._feature_dim if self._feature_dim else 0
        if n_orig_features == 0:
            return {}
        
        importance = np.zeros(n_orig_features)
        
        for i in range(self.weights.shape[0]):
            for j in range(self.weights.shape[1]):
                # Rough estimate: distribute polynomial weights to original features
                # Simplified approach; real applications need more sophisticated analysis
                if j < n_orig_features:
                    importance[j] += abs(self.weights[i, j])
        
        importance = importance / np.sum(importance)
        
        return {f"feature_{i}": float(imp) for i, imp in enumerate(importance)}
    
    def __repr__(self) -> str:
        return f"PolynomialSelector(degree={self.degree}, features={self.n_poly}, reg={self.regularization})"


# ============================================================
# Rule-Based Selector
# ============================================================

class RuleBasedSelector(Selector):
    """
    Rule-Based Selector: Uses predefined rules to compute fitness.
    
    Suitable for expert systems or deterministic scenarios where rules are clear and non-learnable.
    
    Characteristics:
        - Deterministic: Clear rules, predictable results
        - Expert knowledge: Encodes domain expert experience
        - Non-learnable: Parameters cannot be adjusted
    """
    
    def __init__(self, 
                 rules: Dict[str, Callable[[FeatureVector], float]],
                 default_fitness: float = 0.5):
        """
        Initialize rule-based selector.
        
        Args:
            rules: Mapping from behavior names to rule functions
            default_fitness: Default fitness value when no rule exists
            
        Raises:
            TypeError: If any rule is not callable
        """
        super().__init__()
        
        self.rules = rules
        self.default_fitness = np.clip(default_fitness, 0.0, 1.0)
        self._logger = logging.getLogger("metathin.selector.RuleBasedSelector")
        
        # Validate rules
        for name, rule in rules.items():
            if not callable(rule):
                raise TypeError(f"Rule '{name}' must be callable")
    
    def compute_fitness(self, behavior: MetaBehavior, features: FeatureVector) -> FitnessScore:
        """
        Compute fitness using rules.
        
        Args:
            behavior: Behavior to evaluate
            features: Feature vector
            
        Returns:
            FitnessScore: Fitness value in [0,1]
        """
        try:
            name = behavior.name
            
            if name in self.rules:
                fitness = self.rules[name](features)
            else:
                self._logger.debug(f"Behavior '{name}' has no rule, using default fitness {self.default_fitness}")
                fitness = self.default_fitness
            
            fitness = float(np.clip(fitness, 0.0, 1.0))
            self.record_fitness(behavior.name, fitness)
            
            return fitness
            
        except Exception as e:
            self._logger.error(f"Rule-based fitness computation failed: {e}")
            raise FitnessComputationError(f"Rule-based fitness computation failed: {e}") from e
    
    def add_rule(self, behavior_name: str, rule: Callable[[FeatureVector], float]) -> None:
        """Add a rule."""
        if not callable(rule):
            raise TypeError("Rule must be callable")
        self.rules[behavior_name] = rule
    
    def remove_rule(self, behavior_name: str) -> bool:
        """Remove a rule."""
        if behavior_name in self.rules:
            del self.rules[behavior_name]
            return True
        return False


# ============================================================
# Ensemble Selector
# ============================================================

class EnsembleSelector(Selector):
    """
    Ensemble Selector: Combines results from multiple selectors.
    
    Aggregates evaluations from multiple selectors through weighted averaging or voting,
    improving evaluation stability and accuracy.
    
    Characteristics:
        - Robustness: Individual selector failures don't affect overall result
        - Diversity: Leverages strengths of different selectors
        - Flexibility: Supports multiple aggregation methods
    """
    
    def __init__(self,
                 selectors: List[Selector],
                 weights: Optional[List[float]] = None,
                 aggregation: str = 'weighted_average'):
        """
        Initialize ensemble selector.
        
        Args:
            selectors: List of component selectors
            weights: Weight for each selector, None for equal weights
            aggregation: Aggregation method:
                - 'weighted_average': Weighted average
                - 'max': Take maximum value
                - 'min': Take minimum value
                - 'median': Take median value
                - 'product': Product (logical AND)
                
        Raises:
            ValueError: If selectors list is empty or weights mismatch
        """
        super().__init__()
        
        if not selectors:
            raise ValueError("Selectors list cannot be empty")
        
        self.selectors = selectors
        self.aggregation = aggregation
        
        # Set weights
        if weights is None:
            self.weights = [1.0 / len(selectors)] * len(selectors)
        else:
            if len(weights) != len(selectors):
                raise ValueError(f"Number of weights ({len(weights)}) must match number of selectors ({len(selectors)})")
            total = sum(weights)
            self.weights = [w / total for w in weights]
        
        self._logger = logging.getLogger("metathin.selector.EnsembleSelector")
    
    def compute_fitness(self, behavior: MetaBehavior, features: FeatureVector) -> FitnessScore:
        """
        Compute ensemble fitness.
        
        Args:
            behavior: Behavior to evaluate
            features: Feature vector
            
        Returns:
            FitnessScore: Fitness value in [0,1]
        """
        try:
            # Collect evaluations from all selectors
            scores = []
            valid_weights = []
            
            for selector, weight in zip(self.selectors, self.weights):
                try:
                    score = selector.compute_fitness(behavior, features)
                    
                    if 0.0 <= score <= 1.0:
                        scores.append(score)
                        valid_weights.append(weight)
                except Exception as e:
                    self._logger.debug(f"Selector {type(selector).__name__} evaluation failed: {e}")
                    continue
            
            if not scores:
                return 0.5
            
            # Compute based on aggregation method
            if self.aggregation == 'weighted_average':
                valid_weights = np.array(valid_weights)
                valid_weights = valid_weights / valid_weights.sum()
                fitness = np.average(scores, weights=valid_weights)
            elif self.aggregation == 'max':
                fitness = max(scores)
            elif self.aggregation == 'min':
                fitness = min(scores)
            elif self.aggregation == 'median':
                fitness = np.median(scores)
            elif self.aggregation == 'product':
                fitness = np.prod(scores)
            else:
                fitness = np.mean(scores)
            
            fitness = float(np.clip(fitness, 0.0, 1.0))
            self.record_fitness(behavior.name, fitness)
            
            return fitness
            
        except Exception as e:
            self._logger.error(f"Ensemble fitness computation failed: {e}")
            raise FitnessComputationError(f"Ensemble fitness computation failed: {e}") from e
    
    def add_selector(self, selector: Selector, weight: Optional[float] = None) -> None:
        """Add a selector to the ensemble."""
        self.selectors.append(selector)
        
        if weight is None:
            new_weight = 1.0 / len(self.selectors)
            self.weights = [new_weight] * len(self.selectors)
        else:
            self.weights.append(weight)
            total = sum(self.weights)
            self.weights = [w / total for w in self.weights]


# ============================================================
# Adaptive Selector
# ============================================================

class AdaptiveSelector(Selector):
    """
    Adaptive Selector: Dynamically adjusts strategy based on performance.
    
    Dynamically switches between multiple sub-selectors, choosing the one with best recent performance.
    Uses ε-greedy strategy to balance exploration and exploitation.
    
    Characteristics:
        - Dynamic switching: Selects best-performing selector based on recent history
        - Exploration-exploitation: ε-greedy strategy balances exploration and exploitation
        - Performance tracking: Records historical performance of each selector
    """
    
    def __init__(self,
                 selectors: List[Selector],
                 performance_window: int = 100,
                 exploration_rate: float = 0.1):
        """
        Initialize adaptive selector.
        
        Args:
            selectors: List of sub-selectors
            performance_window: Performance evaluation window size
            exploration_rate: Exploration rate for ε-greedy strategy
        """
        super().__init__()
        
        self.selectors = selectors
        self.performance_window = performance_window
        self.exploration_rate = exploration_rate
        
        # Currently active selector index
        self.active_idx = 0
        
        # Performance records
        self.selector_performance: List[List[float]] = [[] for _ in selectors]
        
        self._logger = logging.getLogger("metathin.selector.AdaptiveSelector")
    
    def compute_fitness(self, behavior: MetaBehavior, features: FeatureVector) -> FitnessScore:
        """
        Compute fitness using the currently active selector.
        
        Args:
            behavior: Behavior to evaluate
            features: Feature vector
            
        Returns:
            FitnessScore: Fitness value in [0,1]
        """
        # ε-greedy exploration
        if np.random.random() < self.exploration_rate:
            # Exploration mode: random selection
            idx = np.random.randint(len(self.selectors))
            self._logger.debug(f"Exploration mode: using selector {idx}")
        else:
            # Exploitation mode: choose best performing
            idx = self._get_best_selector()
        
        self.active_idx = idx
        selector = self.selectors[idx]
        
        try:
            fitness = selector.compute_fitness(behavior, features)
            return fitness
        except Exception as e:
            self._logger.error(f"Selector {idx} failed: {e}")
            return self.selectors[0].compute_fitness(behavior, features)
    
    def _get_best_selector(self) -> int:
        """Get index of the best performing selector."""
        avg_performance = []
        
        for i, perf in enumerate(self.selector_performance):
            if perf:
                avg_perf = np.mean(perf[-self.performance_window:])
                avg_performance.append((avg_perf, i))
            else:
                avg_performance.append((0, i))
        
        return max(avg_performance, key=lambda x: x[0])[1]
    
    def record_performance(self, selector_idx: int, performance: float):
        """Record selector performance."""
        if 0 <= selector_idx < len(self.selector_performance):
            self.selector_performance[selector_idx].append(performance)