"""
Learning Mechanism Components
=============================

Provides various parameter update algorithms that enable the agent to learn from experience.
The learning mechanism serves as the agent's "brain," responsible for adjusting behavior
selection based on feedback.

Design Philosophy:
    - Diversity: Supports supervised learning, reinforcement learning, and unsupervised learning
    - Composability: Can combine multiple learning mechanisms to leverage their strengths
    - Observable: Provides visualizable statistics of the learning process
    - Stability: Numerically stable computations prevent gradient explosion

The learning mechanism Ψ calculates parameter adjustments γ̂ based on the discrepancy between
expected and actual results. These adjustments modify the selector S's parameters,
thereby influencing future decisions.

Learning Paradigms:
    - Supervised Learning: Uses differences between expected and actual (GradientLearning)
    - Reinforcement Learning: Uses reward signals (RewardLearning)
    - Unsupervised Learning: Discovers patterns in data (HebbianLearning)
    - Memory-based Learning: Based on historical experience (MemoryLearning)
    - Ensemble Learning: Combines multiple learners (EnsembleLearning)
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import deque, defaultdict
import logging
from dataclasses import dataclass, field
import heapq

from ..core.interfaces import LearningMechanism, ParameterDict, FeatureVector
from ..core.interfaces import LearningError


# ============================================================
# Helper Functions
# ============================================================
# These functions provide common loss calculation and gradient processing utilities
# ============================================================

def mse_loss(expected: float, actual: float) -> float:
    """
    Mean Squared Error loss.
    
    Computes the squared difference between expected and actual values.
    Characteristics: Penalizes large errors heavily, more tolerant of small errors.
    
    Args:
        expected: Expected value
        actual: Actual value
        
    Returns:
        float: MSE loss value
    """
    return (expected - actual) ** 2


def mae_loss(expected: float, actual: float) -> float:
    """
    Mean Absolute Error loss.
    
    Computes the absolute difference between expected and actual values.
    Characteristics: Insensitive to outliers, constant gradient.
    
    Args:
        expected: Expected value
        actual: Actual value
        
    Returns:
        float: MAE loss value
    """
    return abs(expected - actual)


def huber_loss(expected: float, actual: float, delta: float = 1.0) -> float:
    """
    Huber loss, combining advantages of MSE and MAE.
    
    Uses MSE when error < delta, MAE when error > delta.
    Characteristics: Smooth and differentiable, insensitive to outliers.
    
    Args:
        expected: Expected value
        actual: Actual value
        delta: Threshold for switching between MSE and MAE
        
    Returns:
        float: Huber loss value
    """
    error = expected - actual
    if abs(error) <= delta:
        return 0.5 * error ** 2
    else:
        return delta * (abs(error) - 0.5 * delta)


def gradient_clip(grad: float, max_norm: float = 1.0) -> float:
    """
    Gradient clipping to prevent gradient explosion.
    
    Limits gradient magnitude to max_norm.
    
    Args:
        grad: Gradient value
        max_norm: Maximum allowed norm
        
    Returns:
        float: Clipped gradient
    """
    if abs(grad) > max_norm:
        return max_norm if grad > 0 else -max_norm
    return grad


def safe_float_convert(value: Any, default: float = 0.0) -> float:
    """
    Safely convert any value to float.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        float: Converted float value
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_array_convert(data: Any, default_dim: int = 1) -> np.ndarray:
    """
    Safely convert any data to numpy array.
    
    Args:
        data: Data to convert
        default_dim: Default dimension if conversion fails
        
    Returns:
        np.ndarray: Converted numpy array
    """
    if data is None:
        return np.zeros(default_dim, dtype=np.float64)
    
    if isinstance(data, np.ndarray):
        return data.astype(np.float64).flatten()
    elif isinstance(data, (list, tuple)):
        try:
            return np.array([safe_float_convert(x) for x in data], dtype=np.float64)
        except:
            return np.zeros(len(data), dtype=np.float64)
    elif isinstance(data, (int, float)):
        return np.array([float(data)], dtype=np.float64)
    else:
        return np.zeros(default_dim, dtype=np.float64)


# ============================================================
# Experience Data Class
# ============================================================

@dataclass
class Experience:
    """
    Experience sample data class.
    
    Records a complete execution experience for memory-based learning.
    Similar to a transition in reinforcement learning, containing state, behavior, and results.
    
    Attributes:
        features: Feature vector (state representation)
        behavior: Behavior name
        expected: Expected result
        actual: Actual result
        reward: Reward value
        timestamp: Timestamp of the experience
        metadata: Additional metadata
    """
    
    features: FeatureVector
    behavior: str
    expected: Any = None
    actual: Any = None
    reward: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Ensure features is a numpy array
        if self.features is None:
            self.features = np.array([0.0], dtype=np.float64)
        elif not isinstance(self.features, np.ndarray):
            self.features = np.array(self.features, dtype=np.float64)
        
        # Ensure features are 1-dimensional
        if self.features.ndim > 1:
            self.features = self.features.flatten()
    
    def similarity(self, other: 'Experience') -> float:
        """
        Calculate similarity with another experience.
        
        Used for finding similar experiences in memory-based learning.
        Combines feature similarity, behavior consistency, and time decay.
        
        Args:
            other: Another experience to compare with
            
        Returns:
            float: Similarity score in [0,1]
        """
        # Feature similarity
        feat_sim = self._feature_similarity(other.features)
        
        # Same behavior?
        behavior_sim = 1.0 if self.behavior == other.behavior else 0.0
        
        # Time decay
        time_diff = abs(self.timestamp - other.timestamp)
        time_decay = np.exp(-time_diff / 3600)  # 1-hour decay
        
        # Combined similarity
        return 0.5 * feat_sim + 0.3 * behavior_sim + 0.2 * time_decay
    
    def _feature_similarity(self, other_features: FeatureVector) -> float:
        """Calculate feature similarity using cosine similarity."""
        if other_features is None:
            return 0.0
        
        if not isinstance(other_features, np.ndarray):
            other_features = np.array(other_features, dtype=np.float64)
        
        if other_features.ndim > 1:
            other_features = other_features.flatten()
        
        # Handle dimension mismatch
        if len(self.features) != len(other_features):
            min_len = min(len(self.features), len(other_features))
            v1 = self.features[:min_len]
            v2 = other_features[:min_len]
        else:
            v1 = self.features
            v2 = other_features
        
        # Compute cosine similarity
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cos_sim = np.dot(v1, v2) / (norm1 * norm2)
        return float(np.clip((cos_sim + 1.0) / 2.0, 0.0, 1.0))


# ============================================================
# Gradient Learning
# ============================================================

class GradientLearning(LearningMechanism):
    """
    Gradient Learning: Error-based gradient descent.
    
    Uses squared error as loss to compute parameter gradients.
    Supports momentum, learning rate decay, and gradient clipping.
    
    Characteristics:
        - Supervised learning: Requires difference between expected and actual
        - Gradient descent: Updates parameters using error gradients
        - Momentum: Accelerates convergence, smooths gradients
        - Learning rate decay: Fast learning initially, fine-tuning later
    
    Attributes:
        learning_rate: Step size for parameter updates
        momentum: Momentum coefficient for smoothing gradients
        decay: Learning rate decay factor
        clip_norm: Gradient clipping threshold
        loss_function: Type of loss function ('mse', 'mae', 'huber')
    """
    
    def __init__(self, 
                 learning_rate: float = 0.01, 
                 momentum: float = 0.0,
                 decay: float = 1.0,
                 clip_norm: float = 1.0,
                 loss_function: str = 'mse'):
        """
        Initialize gradient learning.
        
        Args:
            learning_rate: Learning rate, controls update step size
            momentum: Momentum coefficient, accelerates convergence
            decay: Learning rate decay rate, multiplied after each update
            clip_norm: Gradient clipping threshold, prevents explosion
            loss_function: Loss function type ('mse', 'mae', 'huber')
            
        Raises:
            ValueError: If parameters are out of valid range
        """
        super().__init__()
        
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {learning_rate}")
        
        if not 0 <= momentum < 1:
            raise ValueError(f"momentum must be in [0,1), got {momentum}")
        
        if not 0 < decay <= 1:
            raise ValueError(f"decay must be in (0,1], got {decay}")
        
        self.base_lr = learning_rate
        self.current_lr = learning_rate
        self.momentum = momentum
        self.decay = decay
        self.clip_norm = clip_norm
        self.loss_function = loss_function
        
        # Momentum cache
        self.velocity: Dict[str, float] = {}
        
        # Loss history
        self.loss_history: List[float] = []
        
        # Gradient history
        self.gradient_history: Dict[str, List[float]] = defaultdict(list)
        
        self._logger = logging.getLogger("metathin.learning.GradientLearning")
        self._step = 0
    
    def _compute_loss(self, expected: Any, actual: Any) -> float:
        """Compute loss based on selected loss function."""
        exp_val = safe_float_convert(expected)
        act_val = safe_float_convert(actual)
        
        error = exp_val - act_val
        
        if self.loss_function == 'mse':
            return error ** 2
        elif self.loss_function == 'mae':
            return abs(error)
        elif self.loss_function == 'huber':
            return huber_loss(exp_val, act_val)
        else:
            return error ** 2
    
    def _compute_gradient(self, 
                         key: str, 
                         value: float, 
                         error: float,
                         features: np.ndarray,
                         param_idx: int) -> float:
        """
        Compute parameter gradient.
        
        For linear models: gradient = -2 * error * feature
        """
        if features is None or len(features) == 0:
            features = np.array([1.0])
        
        feat_idx = param_idx % len(features)
        grad = -2 * error * features[feat_idx]
        
        # Gradient clipping
        grad = gradient_clip(grad, self.clip_norm)
        
        return grad
    
    def compute_adjustment(self, 
                          expected: Any, 
                          actual: Any,
                          context: Dict[str, Any]) -> Optional[ParameterDict]:
        """
        Compute parameter adjustments based on error.
        
        Implementation steps:
            1. Calculate loss
            2. Compute gradients for each parameter
            3. Apply momentum
            4. Calculate adjustments
            5. Decay learning rate
        
        Args:
            expected: Expected result
            actual: Actual result
            context: Context dictionary containing parameters, features, etc.
            
        Returns:
            Optional[ParameterDict]: Parameter adjustments, None if no adjustment needed
        """
        try:
            self._step += 1
            
            # Get context information
            params = context.get('parameters', {})
            features = context.get('features')
            
            # Safely process features
            features = safe_array_convert(features, default_dim=1)
            
            if not params:
                self._logger.debug("No learnable parameters")
                return None
            
            # Compute error
            loss = self._compute_loss(expected, actual)
            self.loss_history.append(loss)
            
            # Limit loss history size
            if len(self.loss_history) > 1000:
                self.loss_history = self.loss_history[-1000:]
            
            # Compute error value for gradient
            exp_val = safe_float_convert(expected)
            act_val = safe_float_convert(actual)
            error = exp_val - act_val
            
            # Get current learning rate
            lr = context.get('learning_rate', self.current_lr)
            
            # Compute gradients for each parameter
            adjustments = {}
            
            for i, (key, value) in enumerate(params.items()):
                if not key.startswith(('w_', 'b_')):
                    continue
                
                # Compute gradient
                grad = self._compute_gradient(key, value, error, features, i)
                
                # Record gradient history
                self.gradient_history[key].append(grad)
                if len(self.gradient_history[key]) > 100:
                    self.gradient_history[key] = self.gradient_history[key][-100:]
                
                # Apply momentum
                if self.momentum > 0:
                    if key in self.velocity:
                        grad = self.momentum * self.velocity[key] + (1 - self.momentum) * grad
                    self.velocity[key] = grad
                
                # Calculate adjustment
                delta = -lr * grad  # Gradient descent: subtract gradient
                
                if not np.isnan(delta) and not np.isinf(delta):
                    adjustments[key] = delta
            
            # Decay learning rate
            if self.decay < 1.0:
                self.current_lr = self.base_lr * (self.decay ** self._step)
            
            return adjustments if adjustments else None
            
        except Exception as e:
            self._logger.error(f"Gradient computation failed: {e}")
            return None
    
    def get_loss_stats(self) -> Dict[str, float]:
        """Get loss statistics."""
        if not self.loss_history:
            return {}
        
        recent = self.loss_history[-100:] if len(self.loss_history) > 100 else self.loss_history
        
        return {
            'current_loss': self.loss_history[-1],
            'avg_loss': float(np.mean(recent)),
            'min_loss': float(np.min(self.loss_history)),
            'max_loss': float(np.max(self.loss_history)),
            'loss_std': float(np.std(recent)),
        }
    
    def reset(self) -> None:
        """Reset learner state."""
        self.velocity.clear()
        self.gradient_history.clear()
        self.loss_history.clear()
        self.current_lr = self.base_lr
        self._step = 0


# ============================================================
# Reward Learning
# ============================================================

class RewardLearning(LearningMechanism):
    """
    Reward Learning: Adjusts parameters based on reward signals.
    
    Suitable for reinforcement learning scenarios, updating parameters according to rewards.
    Higher rewards increase parameters, lower rewards decrease them.
    
    Characteristics:
        - Reinforcement learning: Requires only reward signals, no expected results
        - Advantage function: Reward minus baseline reduces variance
        - Adaptive baseline: Uses historical average reward as baseline
    
    Attributes:
        learning_rate: Learning rate for parameter updates
        baseline: Reward baseline for advantage calculation
        use_advantage: Whether to use advantage function (reward - baseline)
    """
    
    def __init__(self, 
                 learning_rate: float = 0.1,
                 baseline: float = 0.5,
                 use_advantage: bool = True):
        """
        Initialize reward learning.
        
        Args:
            learning_rate: Learning rate
            baseline: Reward baseline for advantage calculation
            use_advantage: Whether to use advantage function (reward - baseline)
            
        Raises:
            ValueError: If learning_rate <= 0
        """
        super().__init__()
        
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {learning_rate}")
        
        self.learning_rate = learning_rate
        self.baseline = baseline
        self.use_advantage = use_advantage
        
        # Reward history
        self.reward_history: List[float] = []
        
        # Cumulative average reward (for adaptive baseline)
        self.cumulative_reward = 0.0
        self.reward_count = 0
        
        self._logger = logging.getLogger("metathin.learning.RewardLearning")
    
    def compute_adjustment(self, 
                          expected: Any, 
                          actual: Any,
                          context: Dict[str, Any]) -> Optional[ParameterDict]:
        """
        Compute parameter adjustments based on reward.
        
        Implementation steps:
            1. Get reward from context
            2. Update cumulative average reward
            3. Calculate advantage (reward - baseline)
            4. Adjust parameters based on advantage
        
        Args:
            expected: Expected result (ignored in reward learning)
            actual: Actual result (ignored in reward learning)
            context: Context dictionary containing reward, parameters, features
            
        Returns:
            Optional[ParameterDict]: Parameter adjustments, None if no adjustment needed
        """
        try:
            # Get reward
            reward = context.get('reward', 0.0)
            reward = safe_float_convert(reward)
            
            self.reward_history.append(reward)
            
            # Limit history size
            if len(self.reward_history) > 1000:
                self.reward_history = self.reward_history[-1000:]
            
            # Update cumulative average
            self.cumulative_reward = (self.cumulative_reward * self.reward_count + reward)
            self.reward_count += 1
            if self.reward_count > 0:
                self.cumulative_reward /= self.reward_count
            
            # Get parameters
            params = context.get('parameters', {})
            if not params:
                return None
            
            # Get features
            features = context.get('features')
            features = safe_array_convert(features, default_dim=1)
            
            # Calculate advantage
            if self.use_advantage:
                baseline = self.cumulative_reward
                advantage = reward - baseline
            else:
                advantage = reward - self.baseline
            
            self._logger.debug(f"reward={reward:.3f}, advantage={advantage:.3f}, baseline={self.cumulative_reward:.3f}")
            
            # Calculate adjustments
            adjustments = {}
            
            for i, (key, value) in enumerate(params.items()):
                if not key.startswith('w_'):
                    continue
                
                feat_idx = i % len(features)
                delta = self.learning_rate * advantage * features[feat_idx]
                
                if not np.isnan(delta) and not np.isinf(delta):
                    adjustments[key] = delta
            
            return adjustments if adjustments else None
            
        except Exception as e:
            self._logger.error(f"Reward learning failed: {e}")
            return None
    
    def get_average_reward(self, window: Optional[int] = None) -> float:
        """
        Get average reward.
        
        Args:
            window: Number of recent rewards to average, None for all
            
        Returns:
            float: Average reward
        """
        if not self.reward_history:
            return 0.0
        
        if window:
            recent = self.reward_history[-min(window, len(self.reward_history)):]
            return float(np.mean(recent))
        else:
            return float(np.mean(self.reward_history))
    
    def update_baseline(self, new_baseline: float) -> None:
        """Update reward baseline."""
        self.baseline = new_baseline


# ============================================================
# Memory Learning
# ============================================================

class MemoryLearning(LearningMechanism):
    """
    Memory Learning: Learns from historical experiences.
    
    Stores past experiences and reuses them in similar situations.
    Uses nearest neighbor search to find similar experiences and adjusts
    parameters based on their average error.
    
    Characteristics:
        - Case-based reasoning: Past similar experiences guide current decisions
        - Nearest neighbor search: Finds k most similar experiences
        - Weighted average: Higher similarity = higher weight
    
    Attributes:
        memory_size: Memory capacity
        similarity_threshold: Minimum similarity to consider
        k_neighbors: Number of nearest neighbors
        learning_rate: Learning rate for parameter updates
    """
    
    def __init__(self, 
                 memory_size: int = 1000,
                 similarity_threshold: float = 0.7,
                 k_neighbors: int = 5,
                 learning_rate: float = 0.01):
        """
        Initialize memory learning.
        
        Args:
            memory_size: Memory capacity
            similarity_threshold: Similarity threshold, below which experiences are ignored
            k_neighbors: Number of nearest neighbors to consider
            learning_rate: Learning rate
        """
        super().__init__()
        
        self.memory_size = memory_size
        self.similarity_threshold = similarity_threshold
        self.k_neighbors = k_neighbors
        self.learning_rate = learning_rate
        
        # Memory storage (deque automatically removes oldest)
        self.memory: deque = deque(maxlen=memory_size)
        
        # Statistics
        self.recall_count = 0
        self.hit_count = 0
        self.miss_count = 0
        
        self._logger = logging.getLogger("metathin.learning.MemoryLearning")
    
    def remember(self, experience: Experience) -> None:
        """Remember an experience."""
        if experience is not None:
            self.memory.append(experience)
    
    def remember_many(self, experiences: List[Experience]) -> None:
        """Batch remember experiences."""
        if experiences:
            for exp in experiences:
                if exp is not None:
                    self.memory.append(exp)
    
    def _find_similar(self, 
                     features: FeatureVector, 
                     behavior_name: str,
                     k: int) -> List[Tuple[float, Experience]]:
        """
        Find similar experiences using cosine similarity.
        
        Args:
            features: Query feature vector
            behavior_name: Behavior name to match
            k: Number of neighbors to return
            
        Returns:
            List[Tuple[float, Experience]]: (similarity, experience) pairs
        """
        if not self.memory or features is None:
            return []
        
        # Ensure features is numpy array
        if not isinstance(features, np.ndarray):
            features = np.array(features, dtype=np.float64)
        
        if features.ndim > 1:
            features = features.flatten()
        
        # Create temporary experience for comparison
        query_exp = Experience(features.copy(), behavior_name)
        
        # Compute all similarities
        similarities = []
        for exp in self.memory:
            if exp is None or exp.behavior != behavior_name:
                continue
            
            try:
                sim = query_exp.similarity(exp)
                if sim >= self.similarity_threshold:
                    similarities.append((sim, exp))
            except Exception as e:
                self._logger.debug(f"Similarity computation failed: {e}")
                continue
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        return similarities[:k]
    
    def compute_adjustment(self, 
                          expected: Any, 
                          actual: Any,
                          context: Dict[str, Any]) -> Optional[ParameterDict]:
        """
        Compute parameter adjustments based on memory.
        
        Implementation steps:
            1. Find similar experiences
            2. Compute weighted average error
            3. Adjust parameters based on error
        
        Args:
            expected: Expected result
            actual: Actual result
            context: Context dictionary containing features, behavior_name, parameters
            
        Returns:
            Optional[ParameterDict]: Parameter adjustments, None if no adjustment needed
        """
        try:
            # Get context
            features = context.get('features')
            behavior_name = context.get('behavior_name')
            
            if features is None or behavior_name is None:
                self._logger.debug("Missing required context: features or behavior_name")
                return None
            
            if len(self.memory) == 0:
                self._logger.debug("Memory is empty")
                return None
            
            # Ensure features is numpy array
            if not isinstance(features, np.ndarray):
                features = np.array(features, dtype=np.float64)
            
            features = features.flatten()
            
            # Find similar experiences
            similar = self._find_similar(features, behavior_name, self.k_neighbors)
            
            self.recall_count += 1
            
            if not similar:
                self.miss_count += 1
                self._logger.debug(f"No similar experiences found (threshold={self.similarity_threshold})")
                return None
            
            self.hit_count += 1
            self._logger.debug(f"Found {len(similar)} similar experiences")
            
            # Compute weighted average error
            total_weight = 0.0
            weighted_error = 0.0
            
            for sim, exp in similar:
                exp_val = safe_float_convert(exp.expected)
                act_val = safe_float_convert(exp.actual)
                error = exp_val - act_val
                
                weight = sim
                weighted_error += weight * error
                total_weight += weight
            
            if total_weight == 0:
                return None
            
            avg_error = weighted_error / total_weight
            
            self._logger.debug(f"Weighted average error: {avg_error:.4f}")
            
            # Get current parameters
            params = context.get('parameters', {})
            if not params:
                return None
            
            # Generate adjustments
            adjustments = {}
            for i, (key, value) in enumerate(params.items()):
                if key.startswith('w_'):
                    feat_idx = i % len(features)
                    delta = self.learning_rate * avg_error * features[feat_idx]
                    
                    if not np.isnan(delta) and not np.isinf(delta):
                        adjustments[key] = delta
            
            return adjustments if adjustments else None
            
        except Exception as e:
            self._logger.error(f"Memory learning failed: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        total_queries = self.hit_count + self.miss_count
        
        behavior_counts = defaultdict(int)
        for exp in self.memory:
            if exp is not None:
                behavior_counts[exp.behavior] += 1
        
        return {
            'memory_size': len(self.memory),
            'capacity': self.memory.maxlen,
            'recall_count': self.recall_count,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': self.hit_count / max(1, total_queries),
            'behavior_distribution': dict(behavior_counts),
        }
    
    def clear(self) -> None:
        """Clear memory."""
        self.memory.clear()
        self.recall_count = 0
        self.hit_count = 0
        self.miss_count = 0
    
    def prune(self, max_age: float) -> int:
        """
        Prune old memories.
        
        Args:
            max_age: Maximum age in seconds
            
        Returns:
            int: Number of memories removed
        """
        now = time.time()
        old_count = len(self.memory)
        
        new_memory = deque(maxlen=self.memory.maxlen)
        for exp in self.memory:
            if exp is not None and now - exp.timestamp <= max_age:
                new_memory.append(exp)
        
        removed = old_count - len(new_memory)
        self.memory = new_memory
        
        return removed


# ============================================================
# Hebbian Learning
# ============================================================

class HebbianLearning(LearningMechanism):
    """
    Hebbian Learning: "Cells that fire together, wire together."
    
    Adjusts weights based on correlation between input and output.
    When input and output are simultaneously active, connections strengthen.
    
    Characteristics:
        - Unsupervised learning: No expected results needed
        - Biologically plausible: Mimics neuronal plasticity
        - Correlation-driven: Weights proportional to input-output correlation
    
    Attributes:
        learning_rate: Learning rate for weight updates
        use_anti: Whether to use anti-Hebbian learning (negative correlation)
        normalize: Whether to normalize weights after updates
        max_weight: Maximum allowed weight value
    """
    
    def __init__(self, 
                 learning_rate: float = 0.01, 
                 use_anti: bool = False,
                 normalize: bool = True,
                 max_weight: float = 10.0):
        """
        Initialize Hebbian learning.
        
        Args:
            learning_rate: Learning rate
            use_anti: Whether to use anti-Hebbian learning
            normalize: Whether to normalize weights after updates
            max_weight: Maximum weight value
        """
        super().__init__()
        
        self.learning_rate = learning_rate
        self.use_anti = use_anti
        self.normalize = normalize
        self.max_weight = max_weight
        
        self._logger = logging.getLogger("metathin.learning.HebbianLearning")
        self._update_count = 0
    
    def _activation(self, x: float) -> float:
        """
        Compute activation value.
        
        Uses sigmoid function to map input to [0,1] range.
        
        Args:
            x: Input value
            
        Returns:
            float: Activation in [0,1]
        """
        # Use sigmoid function
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))
    
    def compute_adjustment(self, 
                          expected: Any, 
                          actual: Any,
                          context: Dict[str, Any]) -> Optional[ParameterDict]:
        """
        Compute parameter adjustments based on Hebbian rule.
        
        Hebbian rule: Δw = η * (pre * post)
        Anti-Hebbian rule: Δw = -η * (pre * post)
        
        Args:
            expected: Expected result (ignored in Hebbian learning)
            actual: Actual result used as post-synaptic activation
            context: Context dictionary containing features and parameters
            
        Returns:
            Optional[ParameterDict]: Parameter adjustments, None if no adjustment needed
        """
        try:
            self._update_count += 1
            
            # Get input
            features = context.get('features')
            params = context.get('parameters', {})
            
            if not params:
                return None
            
            # Safely process features
            features = safe_array_convert(features, default_dim=1)
            
            # Compute output activation (post)
            act_val = safe_float_convert(actual)
            post = self._activation(act_val)
            
            self._logger.debug(f"post activation: {post:.3f}")
            
            # Compute adjustments
            adjustments = {}
            
            for i, (key, value) in enumerate(params.items()):
                if not key.startswith('w_'):
                    continue
                
                feat_idx = i % len(features)
                
                # Compute input activation (pre)
                pre = self._activation(features[feat_idx])
                
                # Hebbian rule
                if self.use_anti:
                    delta = -self.learning_rate * pre * post
                else:
                    delta = self.learning_rate * pre * post
                
                if not np.isnan(delta) and not np.isinf(delta):
                    adjustments[key] = delta
            
            return adjustments if adjustments else None
            
        except Exception as e:
            self._logger.error(f"Hebbian learning failed: {e}")
            return None
    
    def post_update(self, params: ParameterDict) -> ParameterDict:
        """
        Post-update processing (normalization, etc.).
        
        Args:
            params: Parameters after updates
            
        Returns:
            ParameterDict: Processed parameters
        """
        if not self.normalize:
            return params
        
        # Extract all weights
        weights = []
        weight_keys = []
        
        for key, value in params.items():
            if key.startswith('w_'):
                weights.append(value)
                weight_keys.append(key)
        
        if not weights:
            return params
        
        # L2 normalization
        norm = np.sqrt(sum(w**2 for w in weights))
        if norm > 0:
            scale = min(1.0, self.max_weight / norm)
            for key in weight_keys:
                params[key] *= scale
        
        return params


# ============================================================
# Ensemble Learning
# ============================================================

class EnsembleLearning(LearningMechanism):
    """
    Ensemble Learning: Combines results from multiple learners.
    
    Can combine multiple learning mechanisms through voting or weighted averaging.
    
    Characteristics:
        - Diversity: Leverages strengths of different learners
        - Robustness: Individual learner failures don't affect overall
        - Flexibility: Weights can be adjusted dynamically
    
    Attributes:
        learners: List of component learners
        weights: Weight for each learner
        aggregation: Aggregation method ('weighted_average', 'max', 'min')
    """
    
    def __init__(self,
                 learners: List[LearningMechanism],
                 weights: Optional[List[float]] = None,
                 aggregation: str = 'weighted_average'):
        """
        Initialize ensemble learning.
        
        Args:
            learners: List of component learners
            weights: Weight for each learner, None means equal weights
            aggregation: Aggregation method ('weighted_average', 'max', 'min')
            
        Raises:
            ValueError: If learners list is empty or weights mismatch
        """
        super().__init__()
        
        if not learners:
            raise ValueError("Learners list cannot be empty")
        
        self.learners = learners
        
        # Set weights
        if weights is None:
            self.weights = [1.0 / len(learners)] * len(learners)
        else:
            if len(weights) != len(learners):
                raise ValueError("Number of weights must match number of learners")
            total = sum(weights)
            self.weights = [w / total for w in weights]
        
        self.aggregation = aggregation
        self._logger = logging.getLogger("metathin.learning.EnsembleLearning")
    
    def compute_adjustment(self, 
                          expected: Any, 
                          actual: Any,
                          context: Dict[str, Any]) -> Optional[ParameterDict]:
        """
        Compute ensemble adjustment.
        
        Implementation steps:
            1. Call all learners to compute adjustments
            2. Filter out failed learners
            3. Aggregate results using weights
        
        Args:
            expected: Expected result
            actual: Actual result
            context: Context dictionary
            
        Returns:
            Optional[ParameterDict]: Combined adjustments, None if no adjustments
        """
        try:
            # Collect adjustments from all learners
            all_adjustments = []
            valid_weights = []
            
            for learner, weight in zip(self.learners, self.weights):
                try:
                    adj = learner.compute_adjustment(expected, actual, context)
                    if adj:
                        all_adjustments.append(adj)
                        valid_weights.append(weight)
                except Exception as e:
                    self._logger.debug(f"Learner {type(learner).__name__} failed: {e}")
                    continue
            
            if not all_adjustments:
                return None
            
            # Normalize weights
            valid_weights = np.array(valid_weights)
            valid_weights = valid_weights / (valid_weights.sum() + 1e-8)
            
            # Merge all parameter keys
            all_keys = set()
            for adj in all_adjustments:
                all_keys.update(adj.keys())
            
            # Compute final adjustment
            final_adjustment = {}
            
            for key in all_keys:
                values = []
                for adj, weight in zip(all_adjustments, valid_weights):
                    if key in adj:
                        delta = adj[key]
                        if not np.isnan(delta) and not np.isinf(delta):
                            values.append(delta * weight)
                
                if values:
                    if self.aggregation == 'weighted_average':
                        final_adjustment[key] = sum(values)
                    elif self.aggregation == 'max':
                        final_adjustment[key] = max(values)
                    elif self.aggregation == 'min':
                        final_adjustment[key] = min(values)
                    else:
                        final_adjustment[key] = sum(values) / len(values)
            
            return final_adjustment if final_adjustment else None
            
        except Exception as e:
            self._logger.error(f"Ensemble learning failed: {e}")
            return None
    
    def add_learner(self, learner: LearningMechanism, weight: Optional[float] = None) -> None:
        """
        Add a learner to the ensemble.
        
        Args:
            learner: Learner to add
            weight: Weight for the learner, None for equal weights
        """
        self.learners.append(learner)
        
        if weight is None:
            new_weight = 1.0 / len(self.learners)
            self.weights = [new_weight] * len(self.learners)
        else:
            self.weights.append(weight)
            total = sum(self.weights)
            if total > 0:
                self.weights = [w / total for w in self.weights]