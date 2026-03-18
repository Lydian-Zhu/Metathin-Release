"""
Metathin Components Package
===========================

Provides ready-to-use component implementations. All components implement the interfaces
defined in core.interfaces. Each component corresponds to one part of the quintuple
(P, B, S, D, Ψ) and can be used directly or serve as a reference for custom implementations.

Component Categories:
    - Pattern Space: Feature extractors that convert raw input to feature vectors
    - MetaBehavior: Executable skill units that perform specific tasks
    - Selector: Fitness calculators that evaluate behavior suitability
    - DecisionStrategy: Behavior selection strategies based on fitness scores
    - LearningMechanism: Parameter update algorithms for learning from feedback

Usage Example:
    >>> from metathin.components import SimplePatternSpace, FunctionBehavior
    >>> from metathin.components import MaxFitnessStrategy, GradientLearning
    >>> 
    >>> # Create components
    >>> pattern = SimplePatternSpace(lambda x: [len(x)])
    >>> behavior = FunctionBehavior("greet", lambda f,**k: "Hello")
    >>> strategy = MaxFitnessStrategy()
    >>> learner = GradientLearning(learning_rate=0.01)
"""

# ============================================================
# Pattern Space Components Import
# ============================================================
# Pattern Space P: Converts raw input into feature vectors
# All these components implement the PatternSpace interface
# ============================================================

from metathin.components.pattern_space import (
    SimplePatternSpace,        # Function-based feature extractor - uses user-provided function
                              # Example: SimplePatternSpace(lambda x: [len(x), x.count(' ')])
    
    StatisticalPatternSpace,   # Statistical feature extractor - computes mean, variance, etc.
                              # Extracts statistical properties from numerical sequences
    
    NormalizedPatternSpace,    # Normalized pattern space - normalizes features to a specific range
                              # Useful when features need to be scaled (e.g., to [0,1] or [-1,1])
    
    CompositePatternSpace,     # Composite pattern space - combines multiple extractors
                              # Allows building complex feature vectors from multiple sources
    
    CachedPatternSpace,        # Cached pattern space - caches extraction results for efficiency
                              # Prevents redundant computation for repeated inputs
)

# ============================================================
# MetaBehavior Components Import
# ============================================================
# MetaBehavior B: Executable skill units
# All these components implement the MetaBehavior interface
# ============================================================

from metathin.components.behavior_library import (
    FunctionBehavior,      # Function-based behavior - wraps a regular Python function
                         # Example: FunctionBehavior("add", lambda f,**k: f[0] + f[1])
    
    LambdaBehavior,        # Lambda behavior - creates simple behaviors using lambda expressions
                         # Useful for quick, one-off behaviors
    
    CompositeBehavior,     # Composite behavior - executes multiple sub-behaviors in sequence
                         # Can be used to build complex workflows from simpler behaviors
    
    RetryBehavior,         # Retry behavior - automatically retries on failure
                         # Configurable retry count and delay between attempts
    
    TimeoutBehavior,       # Timeout behavior - limits execution time
                         # Raises exception if execution exceeds specified timeout
    
    ConditionalBehavior,   # Conditional behavior - chooses branches based on conditions
                         # Implements if-then-else logic for behaviors
    
    CachedBehavior,        # Cached behavior - caches execution results
                         # Returns cached result for identical inputs without re-execution
)

# ============================================================
# Selector Components Import
# ============================================================
# Selector S: Evaluates behavior suitability, computes fitness scores
# All these components implement the Selector interface
# ============================================================

from metathin.components.selector import (
    SimpleSelector,        # Simple linear selector - computes fitness as weighted sum
                         # Basic selector suitable for linear relationships
                         # Parameters: weights and bias for each behavior-feature pair
    
    PolynomialSelector,    # Polynomial selector - uses polynomial regression
                         # Captures non-linear relationships between features and fitness
    
    RuleBasedSelector,     # Rule-based selector - uses predefined rules
                         # Fitness determined by matching conditions against rules
    
    EnsembleSelector,      # Ensemble selector - combines multiple selectors
                         # Can use voting, averaging, or weighted combination
    
    AdaptiveSelector,      # Adaptive selector - dynamically adjusts strategy
                         # Switches between strategies based on performance metrics
)

# ============================================================
# Decision Strategy Components Import
# ============================================================
# Decision Strategy D: Selects optimal behavior based on fitness scores
# All these components implement the DecisionStrategy interface
# ============================================================

from metathin.components.decision import (
    MaxFitnessStrategy,     # Maximum fitness strategy - always selects highest fitness
                          # Pure exploitation, no exploration
    
    ProbabilisticStrategy,  # Probabilistic strategy - selects proportionally to fitness
                          # Higher fitness = higher selection probability
    
    EpsilonGreedyStrategy,  # ε-greedy strategy - explores with probability ε
                          # Balances exploration and exploitation
                          # Configurable epsilon and decay rate
    
    RoundRobinStrategy,     # Round-robin strategy - cycles through behaviors
                          # Ensures each behavior gets equal opportunity
    
    RandomStrategy,         # Random strategy - completely random selection
                          # Pure exploration, useful for baseline comparison
    
    BoltzmannStrategy,      # Boltzmann strategy - softmax with temperature
                          # Temperature controls exploration: higher = more random
    
    HybridStrategy,         # Hybrid strategy - combines multiple strategies
                          # Can switch based on context or confidence
)

# ============================================================
# Learning Mechanism Components Import
# ============================================================
# Learning Mechanism Ψ: Updates selector parameters based on feedback
# All these components implement the LearningMechanism interface
# ============================================================

from metathin.components.learning import (
    GradientLearning,      # Gradient learning - uses error gradient descent
                         # Standard supervised learning approach
                         # Parameters: learning rate, momentum, weight decay
    
    RewardLearning,        # Reward learning - based on reinforcement learning
                         # Uses reward signals rather than explicit error
                         # Suitable for scenarios without ground truth
    
    MemoryLearning,        # Memory learning - based on historical experience
                         # Stores and replays past experiences
                         # Implements experience replay for stability
    
    HebbianLearning,       # Hebbian learning - "cells that fire together wire together"
                         # Unsupervised learning based on correlations
                         # Good for pattern discovery
    
    EnsembleLearning,      # Ensemble learning - combines multiple learners
                         # Can use boosting, bagging, or stacking
    
    Experience,            # Experience sample - data class for memory learning
                         # Stores (features, behavior, reward, next_features) tuples
)

# ============================================================
# Export Interface Definition
# ============================================================
# __all__ controls what gets imported with 'from metathin.components import *'
# Grouped by functionality for easier user reference
# ============================================================

__all__ = [
    # ===== Pattern Space Components =====
    # Feature extractors that convert raw input to feature vectors
    'SimplePatternSpace',      # Function-based feature extraction
    'StatisticalPatternSpace', # Statistical feature extraction
    'NormalizedPatternSpace',  # Normalized feature extraction
    'CompositePatternSpace',   # Composite feature extraction
    'CachedPatternSpace',      # Cached feature extraction
    
    # ===== MetaBehavior Components =====
    # Executable skill units
    'FunctionBehavior',        # Function-based behavior
    'LambdaBehavior',          # Lambda behavior
    'CompositeBehavior',       # Composite behavior
    'RetryBehavior',           # Retry behavior
    'TimeoutBehavior',         # Timeout behavior
    'ConditionalBehavior',     # Conditional behavior
    'CachedBehavior',          # Cached behavior
    
    # ===== Selector Components =====
    # Fitness calculators
    'SimpleSelector',          # Simple linear selector
    'PolynomialSelector',      # Polynomial selector
    'RuleBasedSelector',       # Rule-based selector
    'EnsembleSelector',        # Ensemble selector
    'AdaptiveSelector',        # Adaptive selector
    
    # ===== Decision Strategy Components =====
    # Behavior selection strategies
    'MaxFitnessStrategy',      # Maximum fitness strategy
    'ProbabilisticStrategy',   # Probabilistic strategy
    'EpsilonGreedyStrategy',   # ε-greedy strategy
    'RoundRobinStrategy',      # Round-robin strategy
    'RandomStrategy',          # Random strategy
    'BoltzmannStrategy',       # Boltzmann strategy
    'HybridStrategy',          # Hybrid strategy
    
    # ===== Learning Mechanism Components =====
    # Parameter update algorithms
    'GradientLearning',        # Gradient learning
    'RewardLearning',          # Reward learning
    'MemoryLearning',          # Memory learning
    'HebbianLearning',         # Hebbian learning
    'EnsembleLearning',        # Ensemble learning
    'Experience',              # Experience sample
]

# ============================================================
# Usage Example
# ============================================================
"""
>>> from metathin.components import *
>>> 
>>> # Create a pattern space
>>> # Extracts two features: string length and word count
>>> pattern = SimplePatternSpace(lambda x: [len(str(x)), len(str(x).split())])
>>> 
>>> # Create behaviors
>>> greet = FunctionBehavior("greet", lambda f,**k: f"Hello (input length: {f[0]})")
>>> bye = FunctionBehavior("bye", lambda f,**k: "Goodbye")
>>> 
>>> # Create selector
>>> # 2 features, 2 behaviors
>>> selector = SimpleSelector(n_features=2, n_behaviors=2)
>>> 
>>> # Create decision strategy
>>> # 10% exploration rate
>>> strategy = EpsilonGreedyStrategy(epsilon=0.1)
>>> 
>>> # Create learning mechanism
>>> learner = GradientLearning(learning_rate=0.01)
>>> 
>>> # Test components
>>> features = pattern.extract("hello world")
>>> print(greet.execute(features))
>>> 
>>> # Compute fitness
>>> fitness_greet = selector.compute_fitness(greet, features)
>>> fitness_bye = selector.compute_fitness(bye, features)
>>> print(f"Greet fitness: {fitness_greet:.3f}, Bye fitness: {fitness_bye:.3f}")
>>> 
>>> # Select behavior
>>> selected = strategy.select([greet, bye], [fitness_greet, fitness_bye], features)
>>> print(f"Selected: {selected.name}")
"""