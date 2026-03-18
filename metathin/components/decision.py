"""
Decision Strategy Components
============================

Provides various behavior selection strategies that make decisions based on fitness scores.
Decision strategies embody the agent's "personality" and exploration-exploitation trade-off.

Design Philosophy:
    - Diversity: Supports deterministic, stochastic, and hybrid strategies
    - Tunable: Exploration degree can be controlled via parameters (temperature, ε, etc.)
    - Observable: Provides confidence and probability information
    - Robust: Handles edge cases and numerical issues gracefully

Decision Strategy Categories:
    - Deterministic strategies: MaxFitnessStrategy
    - Probabilistic strategies: ProbabilisticStrategy, BoltzmannStrategy
    - Exploration-exploitation strategies: EpsilonGreedyStrategy
    - Fair strategies: RoundRobinStrategy
    - Baseline strategies: RandomStrategy
    - Combined strategies: HybridStrategy

The decision strategy σ receives fitness scores α(Bi) for all behaviors and outputs the selected behavior B*.
"""

import time
import numpy as np
import random
from typing import List, Dict, Any, Optional, Tuple, Callable
from abc import ABC, abstractmethod
import logging
from collections import deque

from ..core.interfaces import DecisionStrategy, MetaBehavior, FeatureVector
from ..core.interfaces import DecisionError, NoBehaviorError


# ============================================================
# Helper Functions
# ============================================================

def softmax(scores: List[float], temperature: float = 1.0) -> np.ndarray:
    """
    Safe softmax function.
    
    Converts scores to probability distribution, commonly used in probabilistic selection strategies.
    
    Mathematical principle:
        p_i = exp(s_i / T) / Σ_j exp(s_j / T)
    
    Args:
        scores: List of raw scores
        temperature: Temperature parameter controlling randomness
            - Higher temperature → more uniform distribution (more exploration)
            - Lower temperature → more peaked distribution (more exploitation)
    
    Returns:
        np.ndarray: Probability distribution summing to 1
    """
    scores = np.array(scores, dtype=np.float64)
    
    # Subtract maximum for numerical stability to prevent exp overflow
    scores = scores - np.max(scores)
    
    # Apply temperature
    if temperature != 1.0:
        scores = scores / temperature
    
    # Compute exp, clip to prevent overflow
    exp_scores = np.exp(np.clip(scores, -500, 500))
    
    # Normalize
    probs = exp_scores / (np.sum(exp_scores) + 1e-8)
    
    # Ensure probabilities sum to 1
    probs = probs / np.sum(probs)
    
    return probs


def normalize_fitness(fitness_scores: List[float], epsilon: float = 1e-8) -> np.ndarray:
    """
    Normalize fitness scores to a probability distribution.
    
    Ensures non-negative values that sum to 1.
    
    Args:
        fitness_scores: List of fitness scores
        epsilon: Small value to avoid division by zero
    
    Returns:
        np.ndarray: Normalized probability distribution
    """
    scores = np.array(fitness_scores, dtype=np.float64)
    
    # Handle all zeros case
    if np.all(scores < epsilon):
        return np.ones_like(scores) / len(scores)
    
    # Ensure non-negative
    scores = np.maximum(scores, epsilon)
    
    return scores / (np.sum(scores) + epsilon)


# ============================================================
# Maximum Fitness Strategy
# ============================================================

class MaxFitnessStrategy(DecisionStrategy):
    """
    Maximum Fitness Strategy: Always selects the behavior with highest fitness.
    
    This is the most straightforward deterministic strategy, suitable for exploitation-heavy scenarios.
    When multiple behaviors have the same maximum fitness, tie-breaking is applied.
    
    Characteristics:
        - Deterministic: Same input always yields same output
        - Greedy: Always picks what's currently considered best
        - No exploration: Never tries suboptimal behaviors
    
    Attributes:
        tie_breaker: Tie-breaking method
            - 'first': Select the first occurrence
            - 'last': Select the last occurrence
            - 'random': Select randomly among ties
    """
    
    def __init__(self, tie_breaker: str = 'random'):
        """
        Initialize maximum fitness strategy.
        
        Args:
            tie_breaker: Tie-breaking method ('first', 'last', 'random')
        
        Raises:
            ValueError: If tie_breaker is invalid
        """
        if tie_breaker not in ['first', 'last', 'random']:
            raise ValueError(f"tie_breaker must be 'first', 'last', or 'random'")
        
        self.tie_breaker = tie_breaker
        self._logger = logging.getLogger("metathin.decision.MaxFitnessStrategy")
        self._selection_history: List[Dict] = []
    
    def select(self, 
               behaviors: List[MetaBehavior], 
               fitness_scores: List[float],
               features: FeatureVector) -> MetaBehavior:
        """
        Select the behavior with maximum fitness.
        
        Args:
            behaviors: List of candidate behaviors
            fitness_scores: Corresponding fitness scores
            features: Current feature vector
            
        Returns:
            MetaBehavior: Selected behavior
            
        Raises:
            NoBehaviorError: If behaviors list is empty
            DecisionError: If input validation fails
        """
        # Input validation
        if not behaviors:
            raise NoBehaviorError("No behaviors available")
        
        if len(behaviors) != len(fitness_scores):
            raise DecisionError(
                f"Number of behaviors ({len(behaviors)}) and fitness scores ({len(fitness_scores)}) mismatch"
            )
        
        # Handle all equal scores
        if len(set(fitness_scores)) == 1:
            self._logger.debug("All fitness scores equal, selecting randomly")
            return random.choice(behaviors)
        
        # Find maximum fitness
        max_fitness = max(fitness_scores)
        max_indices = [i for i, f in enumerate(fitness_scores) if abs(f - max_fitness) < 1e-6]
        
        # Handle ties
        if len(max_indices) > 1:
            self._logger.debug(f"Detected tie: {len(max_indices)} behaviors with same fitness {max_fitness:.3f}")
            
            if self.tie_breaker == 'first':
                idx = max_indices[0]
            elif self.tie_breaker == 'last':
                idx = max_indices[-1]
            else:  # random
                idx = random.choice(max_indices)
        else:
            idx = max_indices[0]
        
        selected = behaviors[idx]
        
        # Record decision
        self._selection_history.append({
            'timestamp': time.time(),
            'selected': selected.name,
            'fitness': max_fitness,
            'n_candidates': len(behaviors),
            'tie_resolved': len(max_indices) > 1
        })
        
        return selected
    
    def get_confidence(self, fitness_scores: List[float]) -> float:
        """
        Calculate decision confidence.
        
        Uses the difference between highest and second-highest scores as confidence.
        
        Args:
            fitness_scores: List of fitness scores
            
        Returns:
            float: Confidence in [0,1]
        """
        if len(fitness_scores) < 2:
            return 1.0
        
        sorted_scores = sorted(fitness_scores, reverse=True)
        diff = sorted_scores[0] - sorted_scores[1]
        
        return min(1.0, max(0.0, diff))


# ============================================================
# Probabilistic Strategy
# ============================================================

class ProbabilisticStrategy(DecisionStrategy):
    """
    Probabilistic Strategy: Selects behaviors proportionally to their fitness.
    
    Behaviors with higher fitness have higher selection probability, but exploration is still possible.
    Temperature parameter controls randomness.
    
    Characteristics:
        - Stochastic: Same input may yield different outputs
        - Exploratory: Suboptimal behaviors have chance to be selected
        - Tunable: Temperature controls exploration degree
    
    Attributes:
        temperature: Temperature parameter
            - temperature = 1.0: Direct fitness proportion
            - temperature > 1.0: More random (more exploration)
            - temperature < 1.0: More deterministic (more exploitation)
    """
    
    def __init__(self, temperature: float = 1.0):
        """
        Initialize probabilistic strategy.
        
        Args:
            temperature: Temperature parameter, must be > 0
            
        Raises:
            ValueError: If temperature <= 0
        """
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        
        self.temperature = temperature
        self._logger = logging.getLogger("metathin.decision.ProbabilisticStrategy")
        self._selection_history: List[Dict] = []
    
    def select(self, 
               behaviors: List[MetaBehavior], 
               fitness_scores: List[float],
               features: FeatureVector) -> MetaBehavior:
        """
        Select behavior probabilistically.
        
        Args:
            behaviors: List of candidate behaviors
            fitness_scores: Corresponding fitness scores
            features: Current feature vector
            
        Returns:
            MetaBehavior: Selected behavior
        """
        if not behaviors:
            raise NoBehaviorError("No behaviors available")
        
        # Calculate selection probabilities
        probs = self.get_probabilities(fitness_scores)
        
        self._logger.debug(f"Selection probabilities: {dict(zip([b.name for b in behaviors], probs))}")
        
        # Random selection
        try:
            idx = np.random.choice(len(behaviors), p=probs)
            selected = behaviors[idx]
        except ValueError as e:
            # Probability distribution may have issues, fallback to uniform random
            self._logger.warning(f"Probabilistic selection failed ({e}), falling back to uniform random")
            selected = random.choice(behaviors)
            idx = behaviors.index(selected)
        
        # Record decision
        self._selection_history.append({
            'timestamp': time.time(),
            'selected': selected.name,
            'probability': float(probs[idx]),
            'temperature': self.temperature
        })
        
        return selected
    
    def get_probabilities(self, fitness_scores: List[float]) -> List[float]:
        """
        Get selection probability for each behavior.
        
        Args:
            fitness_scores: List of fitness scores
            
        Returns:
            List[float]: Probabilities summing to 1
        """
        if not fitness_scores:
            return []
        
        # Use softmax to compute probabilities
        probs = softmax(fitness_scores, self.temperature)
        
        return probs.tolist()
    
    def get_confidence(self, fitness_scores: List[float]) -> float:
        """
        Calculate decision confidence.
        
        Uses the ratio between highest and second-highest probabilities.
        
        Args:
            fitness_scores: List of fitness scores
            
        Returns:
            float: Confidence in [0,1]
        """
        probs = self.get_probabilities(fitness_scores)
        
        if len(probs) < 2:
            return 1.0
        
        sorted_probs = sorted(probs, reverse=True)
        
        if sorted_probs[1] > 0:
            confidence = 1.0 - (sorted_probs[1] / sorted_probs[0])
        else:
            confidence = 1.0
        
        return min(1.0, max(0.0, confidence))
    
    def set_temperature(self, temperature: float) -> None:
        """
        Dynamically adjust temperature.
        
        Args:
            temperature: New temperature value
        """
        if temperature <= 0:
            raise ValueError(f"Temperature must be > 0")
        self.temperature = temperature


# ============================================================
# Epsilon-Greedy Strategy
# ============================================================

class EpsilonGreedyStrategy(DecisionStrategy):
    """
    Epsilon-Greedy Strategy: Explores randomly with probability ε, exploits optimally with probability 1-ε.
    
    Classic reinforcement learning exploration strategy balancing exploration and exploitation.
    Supports decay mechanism to gradually reduce exploration.
    
    Characteristics:
        - Simple and intuitive: Only two modes
        - Tunable: ε controls exploration degree
        - Decay: Automatically reduces exploration over time
    
    Attributes:
        epsilon: Current exploration probability
        decay: Decay factor (multiplied after each selection)
        min_epsilon: Minimum exploration probability
    """
    
    def __init__(self, 
                 epsilon: float = 0.1, 
                 decay: float = 1.0, 
                 min_epsilon: float = 0.01):
        """
        Initialize epsilon-greedy strategy.
        
        Args:
            epsilon: Initial exploration probability, must be in [0,1]
            decay: Decay factor, multiplied after each selection
            min_epsilon: Minimum exploration probability
            
        Raises:
            ValueError: If parameters are out of valid range
        """
        if not 0 <= epsilon <= 1:
            raise ValueError(f"epsilon must be in [0,1]")
        
        if decay <= 0 or decay > 1:
            raise ValueError(f"decay must be in (0,1]")
        
        if not 0 <= min_epsilon <= 1:
            raise ValueError(f"min_epsilon must be in [0,1]")
        
        self.epsilon = epsilon
        self.decay = decay
        self.min_epsilon = min_epsilon
        self.step = 0
        self._logger = logging.getLogger("metathin.decision.EpsilonGreedyStrategy")
        self._selection_history: List[Dict] = []
    
    def select(self, 
               behaviors: List[MetaBehavior], 
               fitness_scores: List[float],
               features: FeatureVector) -> MetaBehavior:
        """
        Epsilon-greedy selection.
        
        Args:
            behaviors: List of candidate behaviors
            fitness_scores: Corresponding fitness scores
            features: Current feature vector
            
        Returns:
            MetaBehavior: Selected behavior
        """
        if not behaviors:
            raise NoBehaviorError("No behaviors available")
        
        self.step += 1
        current_epsilon = self.epsilon
        
        # Decide between exploration and exploitation
        if random.random() < current_epsilon:
            # Exploration: random selection
            idx = random.randrange(len(behaviors))
            selected = behaviors[idx]
            mode = 'explore'
            self._logger.debug(f"Exploration mode: randomly selected {selected.name}")
        else:
            # Exploitation: select highest fitness
            idx = np.argmax(fitness_scores)
            selected = behaviors[idx]
            mode = 'exploit'
            self._logger.debug(f"Exploitation mode: selected optimal {selected.name} (fitness={fitness_scores[idx]:.3f})")
        
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)
        
        # Record decision
        self._selection_history.append({
            'step': self.step,
            'selected': selected.name,
            'mode': mode,
            'epsilon': current_epsilon,
            'fitness': fitness_scores[idx] if mode == 'exploit' else None
        })
        
        return selected
    
    def get_confidence(self, fitness_scores: List[float]) -> float:
        """
        Calculate decision confidence.
        
        Confidence = 1 - ε (current exploration rate)
        
        Args:
            fitness_scores: List of fitness scores
            
        Returns:
            float: Confidence in [0,1]
        """
        return 1.0 - self.epsilon
    
    def get_exploration_rate(self) -> float:
        """Get current exploration rate."""
        return self.epsilon
    
    def reset_epsilon(self, epsilon: float) -> None:
        """
        Reset exploration rate.
        
        Args:
            epsilon: New exploration rate
        """
        if not 0 <= epsilon <= 1:
            raise ValueError(f"epsilon must be in [0,1]")
        self.epsilon = epsilon
        self.step = 0


# ============================================================
# Round-Robin Strategy
# ============================================================

class RoundRobinStrategy(DecisionStrategy):
    """
    Round-Robin Strategy: Cycles through behaviors in order.
    
    Ignores fitness scores, fairly selecting each behavior in sequence.
    Suitable for testing, load balancing, and uniform evaluation.
    
    Characteristics:
        - Fair: Each behavior gets equal selection count
        - Simple: No parameters, easy to understand
        - Predictable: Selection order is fixed
    
    Attributes:
        current: Current index in the cycle
        cycle: Number of completed cycles
    """
    
    def __init__(self):
        """Initialize round-robin strategy."""
        self.current = 0
        self.cycle = 0
        self._logger = logging.getLogger("metathin.decision.RoundRobinStrategy")
        self._selection_history: List[str] = []
    
    def select(self, 
               behaviors: List[MetaBehavior], 
               fitness_scores: List[float],
               features: FeatureVector) -> MetaBehavior:
        """
        Round-robin selection.
        
        Args:
            behaviors: List of candidate behaviors
            fitness_scores: Corresponding fitness scores (ignored)
            features: Current feature vector
            
        Returns:
            MetaBehavior: Selected behavior
        """
        if not behaviors:
            raise NoBehaviorError("No behaviors available")
        
        # Select behavior at current index
        selected = behaviors[self.current]
        
        # Record selection
        self._selection_history.append(selected.name)
        
        self._logger.debug(f"Round-robin selection: {selected.name} (index={self.current})")
        
        # Update index
        self.current = (self.current + 1) % len(behaviors)
        if self.current == 0:
            self.cycle += 1
            self._logger.debug(f"Completed cycle {self.cycle}")
        
        return selected
    
    def get_cycle(self) -> int:
        """Get number of completed cycles."""
        return self.cycle
    
    def reset(self) -> None:
        """Reset round-robin state."""
        self.current = 0
        self.cycle = 0
        self._selection_history.clear()


# ============================================================
# Random Strategy
# ============================================================

class RandomStrategy(DecisionStrategy):
    """
    Random Strategy: Completely random behavior selection.
    
    Ignores all fitness information, selects uniformly at random.
    Suitable for baseline comparison and pure exploration scenarios.
    
    Characteristics:
        - Purely random: All behaviors equally probable
        - Memoryless: Each selection independent
        - Reproducible: Supports random seed
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize random strategy.
        
        Args:
            seed: Random seed for reproducible experiments
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.seed = seed
        self._logger = logging.getLogger("metathin.decision.RandomStrategy")
        self._selection_history: List[str] = []
    
    def select(self, 
               behaviors: List[MetaBehavior], 
               fitness_scores: List[float],
               features: FeatureVector) -> MetaBehavior:
        """
        Random selection.
        
        Args:
            behaviors: List of candidate behaviors
            fitness_scores: Corresponding fitness scores (ignored)
            features: Current feature vector
            
        Returns:
            MetaBehavior: Selected behavior
        """
        if not behaviors:
            raise NoBehaviorError("No behaviors available")
        
        selected = random.choice(behaviors)
        
        # Record selection
        self._selection_history.append(selected.name)
        
        self._logger.debug(f"Random selection: {selected.name}")
        
        return selected
    
    def get_confidence(self, fitness_scores: List[float]) -> float:
        """
        Random strategy confidence is always 0.
        
        Args:
            fitness_scores: List of fitness scores
            
        Returns:
            float: Always 0.0
        """
        return 0.0


# ============================================================
# Boltzmann Strategy
# ============================================================

class BoltzmannStrategy(DecisionStrategy):
    """
    Boltzmann Exploration: Probabilistic selection based on exponential weighting.
    
    Uses Boltzmann distribution (softmax) for probabilistic selection,
    a generalization of the probabilistic strategy that handles extreme values better.
    
    Characteristics:
        - Temperature control: Temperature adjusts exploration degree
        - Probability smoothing: Avoids overly peaked distributions
        - Theoretical foundation: Rooted in statistical mechanics
    
    Attributes:
        temperature: Temperature parameter controlling randomness
    """
    
    def __init__(self, temperature: float = 1.0):
        """
        Initialize Boltzmann strategy.
        
        Args:
            temperature: Temperature parameter, must be > 0
            
        Raises:
            ValueError: If temperature <= 0
        """
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        
        self.temperature = temperature
        self._logger = logging.getLogger("metathin.decision.BoltzmannStrategy")
        self._selection_history: List[Dict] = []
    
    def select(self, 
               behaviors: List[MetaBehavior], 
               fitness_scores: List[float],
               features: FeatureVector) -> MetaBehavior:
        """
        Boltzmann selection.
        
        Args:
            behaviors: List of candidate behaviors
            fitness_scores: Corresponding fitness scores
            features: Current feature vector
            
        Returns:
            MetaBehavior: Selected behavior
        """
        if not behaviors:
            raise NoBehaviorError("No behaviors available")
        
        # Calculate Boltzmann probabilities
        probs = self.get_probabilities(fitness_scores)
        
        self._logger.debug(f"Boltzmann probabilities: {dict(zip([b.name for b in behaviors], probs))}")
        
        # Random selection
        try:
            idx = np.random.choice(len(behaviors), p=probs)
            selected = behaviors[idx]
        except ValueError as e:
            self._logger.warning(f"Probabilistic selection failed ({e}), falling back to uniform random")
            selected = random.choice(behaviors)
            idx = behaviors.index(selected)
        
        # Record decision
        self._selection_history.append({
            'timestamp': time.time(),
            'selected': selected.name,
            'probability': float(probs[idx]),
            'temperature': self.temperature,
            'max_fitness': max(fitness_scores)
        })
        
        return selected
    
    def get_probabilities(self, fitness_scores: List[float]) -> List[float]:
        """
        Get selection probability for each behavior (Boltzmann distribution).
        
        Args:
            fitness_scores: List of fitness scores
            
        Returns:
            List[float]: Probabilities summing to 1
        """
        if not fitness_scores:
            return []
        
        # Use softmax (Boltzmann distribution)
        probs = softmax(fitness_scores, self.temperature)
        
        return probs.tolist()
    
    def get_confidence(self, fitness_scores: List[float]) -> float:
        """
        Calculate decision confidence.
        
        Uses the difference between highest and second-highest probabilities.
        
        Args:
            fitness_scores: List of fitness scores
            
        Returns:
            float: Confidence in [0,1]
        """
        probs = self.get_probabilities(fitness_scores)
        
        if len(probs) < 2:
            return 1.0
        
        sorted_probs = sorted(probs, reverse=True)
        confidence = sorted_probs[0] - sorted_probs[1]
        
        return min(1.0, max(0.0, confidence))


# ============================================================
# Hybrid Strategy
# ============================================================

class HybridStrategy(DecisionStrategy):
    """
    Hybrid Strategy: Combines multiple strategies, switching based on conditions.
    
    Dynamically switches between different strategies to adapt to various scenarios.
    
    Characteristics:
        - Flexible: Can choose different strategies based on conditions
        - Extensible: Can combine any number of strategies
        - Adaptive: Strategy selector can use context information
    
    Attributes:
        strategies: List of component strategies
        selector: Function that selects which strategy to use
        step: Number of selections performed
    """
    
    def __init__(self,
                 strategies: List[DecisionStrategy],
                 selector: Callable[[int, Dict], int]):
        """
        Initialize hybrid strategy.
        
        Args:
            strategies: List of component strategies
            selector: Selection function, receives (step, context) and returns strategy index
            
        Raises:
            ValueError: If strategies list is empty
        """
        if not strategies:
            raise ValueError("Strategies list cannot be empty")
        
        self.strategies = strategies
        self.selector = selector
        self.step = 0
        self._logger = logging.getLogger("metathin.decision.HybridStrategy")
        self._selection_history: List[Dict] = []
    
    def select(self, 
               behaviors: List[MetaBehavior], 
               fitness_scores: List[float],
               features: FeatureVector) -> MetaBehavior:
        """
        Select using currently chosen strategy.
        
        Args:
            behaviors: List of candidate behaviors
            fitness_scores: Corresponding fitness scores
            features: Current feature vector
            
        Returns:
            MetaBehavior: Selected behavior
        """
        self.step += 1
        
        # Choose current strategy
        context = {
            'behaviors': behaviors,
            'fitness_scores': fitness_scores,
            'features': features,
            'step': self.step
        }
        
        try:
            strategy_idx = self.selector(self.step, context)
            if strategy_idx < 0 or strategy_idx >= len(self.strategies):
                self._logger.warning(f"Invalid strategy index {strategy_idx}, using default 0")
                strategy_idx = 0
        except Exception as e:
            self._logger.error(f"Strategy selection failed: {e}, using default 0")
            strategy_idx = 0
        
        strategy = self.strategies[strategy_idx]
        
        self._logger.debug(f"Using strategy {strategy_idx}: {type(strategy).__name__}")
        
        # Execute selection
        selected = strategy.select(behaviors, fitness_scores, features)
        
        # Record decision
        self._selection_history.append({
            'step': self.step,
            'strategy_idx': strategy_idx,
            'strategy_name': type(strategy).__name__,
            'selected': selected.name
        })
        
        return selected
    
    def get_confidence(self, fitness_scores: List[float]) -> float:
        """
        Get confidence from current strategy.
        
        Simplified implementation using first strategy's confidence.
        
        Args:
            fitness_scores: List of fitness scores
            
        Returns:
            float: Confidence in [0,1]
        """
        if self.strategies:
            return self.strategies[0].get_confidence(fitness_scores)
        return 0.5