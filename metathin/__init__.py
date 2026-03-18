"""
Metathin - Meta-Cognitive Agent System Construction Framework
===============================================================

Metathin derives from Meta+Thinking, a thinking-oriented agent framework built from the 
ground up based on cognitive principles. The framework is designed around the quintuple
structure (P, B, S, D, Ψ), with all components interface-driven and supporting custom
implementations.

Core Components:
    - PatternSpace: Pattern space, converts input to feature vectors (perception layer)
    - MetaBehavior: Meta-behavior, executable skill units (action layer)
    - Selector: Selector, evaluates behavior suitability (evaluation layer)
    - DecisionStrategy: Decision strategy, selects optimal behavior (decision layer)
    - LearningMechanism: Learning mechanism, adjusts based on feedback (learning layer)

Design Philosophy:
    - Fixed interfaces, free implementation: Framework defines clear interfaces, users can implement their own algorithms arbitrarily
    - Modular: Components can be independently replaced and combined
    - Extensible: Supports custom components and third-party integration
    - Type-safe: Type annotations improve code reliability

Version: 0.1.0
Author: Lydian-Zhu
License: MIT
"""

# ============================================================
# Preprocessing Area: Version and Author Information
# ============================================================

__version__ = '0.1.0'           # Framework version, follows semantic versioning
__author__ = 'Lydian-Zhu'       # Author information
__license__ = 'MIT'              # Open source license

# ============================================================
# Function Area: Core Module Imports
# ============================================================

# -------------------------------------------------------------------
# 1. Core Framework Imports
#    Metathin main class and configuration, foundation for all agents
# -------------------------------------------------------------------
from metathin.core import Metathin           # Main class: Core agent integrating the quintuple
from metathin.core import MetathinConfig    # Configuration class: Controls agent behavior parameters

# -------------------------------------------------------------------
# 2. Core Interface Imports
#    Abstract base classes for the quintuple, all custom components must inherit from these interfaces
# -------------------------------------------------------------------
from metathin.core import (
    PatternSpace,          # Pattern space interface P: Window to perceive the environment
    MetaBehavior,          # Meta-behavior interface B: System's skill collection
    Selector,              # Selector interface S: Evaluates behavior suitability
    DecisionStrategy,      # Decision strategy interface D: Selects optimal behavior based on fitness
    LearningMechanism,     # Learning mechanism interface Ψ: Adjusts parameters based on feedback
)

# -------------------------------------------------------------------
# 3. Exception Class Imports
#    Various exceptions defined by the framework for error handling and debugging
# -------------------------------------------------------------------
from metathin.core import (
    MetathinError,                 # Base exception class, parent of all custom exceptions
    PatternExtractionError,        # Pattern extraction error: Raised when pattern space cannot extract features
    BehaviorExecutionError,        # Behavior execution error: Exception during behavior execution
    FitnessComputationError,       # Fitness computation error: Selector cannot compute fitness
    DecisionError,                  # Decision error: Decision strategy cannot make a choice
    NoBehaviorError,                # No available behavior: All behaviors unsuitable for current state
    LearningError,                  # Learning error: Learning mechanism fails to update parameters
    ParameterUpdateError,           # Parameter update error: Failed to update selector parameters
)

# -------------------------------------------------------------------
# 4. State and Record Class Imports
#    Used for tracking the agent's thinking process and internal state
# -------------------------------------------------------------------
from metathin.core import (
    Thought,                # Thought record: Records a complete thinking process
    ThinkingStage,          # Thinking stage: Enumerates different stages of thinking (perceive, hypothesize, decide, etc.)
    LearningStatus,         # Learning status: Enumerates learning outcomes (success, failed, skipped)
)

# -------------------------------------------------------------------
# 5. Memory System Imports
#    Provides persistent memory capabilities, supports multiple storage backends
# -------------------------------------------------------------------
from metathin.core import MemoryManager      # Memory manager: Cache + persistence
from metathin.core.memory import (
    MemoryBackend,          # Memory backend interface: Abstract base for all storage backends
    InMemoryBackend,        # In-memory backend: Fastest but non-persistent
    JSONMemoryBackend,      # JSON file backend: Human-readable persistence based on JSON files
    SQLiteMemoryBackend,    # SQLite backend: Production-ready persistence based on SQLite database
    TTLMemoryManager,       # TTL memory manager: Memory management with time-to-live
)

# ============================================================
# Function Area: Built-in Component Imports
# ============================================================

# -------------------------------------------------------------------
# 6. Pattern Space Components
#    Various pre-built feature extractors that convert raw input to feature vectors
# -------------------------------------------------------------------
try:
    from metathin.components.pattern_space import (
        SimplePatternSpace,         # Simple pattern space: Function-based feature extraction
        StatisticalPatternSpace,    # Statistical pattern space: Extracts statistical features (mean, variance, etc.)
        NormalizedPatternSpace,     # Normalized pattern space: Normalizes features to a consistent range
        CompositePatternSpace,      # Composite pattern space: Concatenates features from multiple pattern spaces
        CachedPatternSpace,         # Cached pattern space: Caches extraction results for efficiency
    )
except ImportError:
    # If component module doesn't exist, define placeholders
    SimplePatternSpace = None
    StatisticalPatternSpace = None
    NormalizedPatternSpace = None
    CompositePatternSpace = None
    CachedPatternSpace = None

# -------------------------------------------------------------------
# 7. Meta-Behavior Components
#    Various pre-built executable skill units
# -------------------------------------------------------------------
try:
    from metathin.components.behavior_library import (
        FunctionBehavior,       # Function-based behavior: Implements behavior using regular Python functions
        LambdaBehavior,         # Lambda behavior: Simple behaviors created with lambda expressions
        CompositeBehavior,      # Composite behavior: Executes multiple sub-behaviors in sequence
        RetryBehavior,          # Retry behavior: Adds retry mechanism to another behavior
        TimeoutBehavior,        # Timeout behavior: Adds timeout control to another behavior
        ConditionalBehavior,    # Conditional behavior: Selects different sub-behaviors based on conditions
        CachedBehavior,         # Cached behavior: Adds result caching to another behavior
    )
except ImportError:
    FunctionBehavior = None
    LambdaBehavior = None
    CompositeBehavior = None
    RetryBehavior = None
    TimeoutBehavior = None
    ConditionalBehavior = None
    CachedBehavior = None

# -------------------------------------------------------------------
# 8. Selector Components
#    Various fitness calculators that evaluate behavior suitability in the current state
# -------------------------------------------------------------------
try:
    from metathin.components.selector import (
        SimpleSelector,         # Simple selector: Computes fitness based on weighted sum
        PolynomialSelector,     # Polynomial selector: Uses polynomial regression for fitness computation
        RuleBasedSelector,      # Rule-based selector: Uses predefined rules to compute fitness
        EnsembleSelector,       # Ensemble selector: Combines results from multiple selectors
        AdaptiveSelector,       # Adaptive selector: Dynamically adjusts strategy based on performance
    )
except ImportError:
    SimpleSelector = None
    PolynomialSelector = None
    RuleBasedSelector = None
    EnsembleSelector = None
    AdaptiveSelector = None

# -------------------------------------------------------------------
# 9. Decision Strategy Components
#    Various behavior selection strategies that make decisions based on fitness scores
# -------------------------------------------------------------------
try:
    from metathin.components.decision import (
        MaxFitnessStrategy,     # Maximum fitness strategy: Always selects the behavior with highest fitness
        ProbabilisticStrategy,  # Probabilistic strategy: Selects proportionally to fitness scores
        EpsilonGreedyStrategy,  # ε-greedy strategy: Explores randomly with probability ε, exploits with probability 1-ε
        RoundRobinStrategy,     # Round-robin strategy: Cycles through behaviors in order
        RandomStrategy,         # Random strategy: Completely random behavior selection
        BoltzmannStrategy,      # Boltzmann strategy: Probabilistic selection based on exponential weighting
        HybridStrategy,         # Hybrid strategy: Combines multiple strategies, switches based on conditions
    )
except ImportError:
    MaxFitnessStrategy = None
    ProbabilisticStrategy = None
    EpsilonGreedyStrategy = None
    RoundRobinStrategy = None
    RandomStrategy = None
    BoltzmannStrategy = None
    HybridStrategy = None

# -------------------------------------------------------------------
# 10. Learning Mechanism Components
#     Various parameter update algorithms that enable the agent to learn from experience
# -------------------------------------------------------------------
try:
    from metathin.components.learning import (
        GradientLearning,       # Gradient learning: Error-based gradient descent
        RewardLearning,         # Reward learning: Adjusts parameters based on reward signals
        MemoryLearning,         # Memory learning: Learns from historical experiences
        HebbianLearning,        # Hebbian learning: "Cells that fire together, wire together"
        EnsembleLearning,       # Ensemble learning: Combines results from multiple learners
        Experience,             # Experience sample: Records a complete execution experience
    )
except ImportError:
    GradientLearning = None
    RewardLearning = None
    MemoryLearning = None
    HebbianLearning = None
    EnsembleLearning = None
    Experience = None

# ============================================================
# Export Interface Definition
# ============================================================
# __all__ controls what gets imported with 'from metathin import *'
# Grouped by functionality for easier user reference
# ============================================================

__all__ = [
    # -------------------------------------------------------------------
    # 1. Version Information
    # -------------------------------------------------------------------
    '__version__',           # Framework version number
    '__author__',            # Author information
    '__license__',           # License information
    
    # -------------------------------------------------------------------
    # 2. Core Classes
    #    Fundamental classes required when using the framework
    # -------------------------------------------------------------------
    'Metathin',              # Main class: Entry point for creating agents
    'MetathinConfig',        # Configuration class: Sets agent behavior parameters
    'Thought',               # Thought record: Debugging and analysis tool
    'ThinkingStage',         # Thinking stage enumeration
    'LearningStatus',        # Learning status enumeration
    
    # -------------------------------------------------------------------
    # 3. Core Interfaces
    #    Abstract base classes to inherit when creating custom components
    # -------------------------------------------------------------------
    'PatternSpace',          # Pattern space interface
    'MetaBehavior',          # Meta-behavior interface
    'Selector',              # Selector interface
    'DecisionStrategy',      # Decision strategy interface
    'LearningMechanism',     # Learning mechanism interface
    
    # -------------------------------------------------------------------
    # 4. Core Exceptions
    #    Exception types that may be raised by the framework
    # -------------------------------------------------------------------
    'MetathinError',                 # Base exception
    'PatternExtractionError',        # Feature extraction error
    'BehaviorExecutionError',        # Behavior execution error
    'FitnessComputationError',       # Fitness computation error
    'DecisionError',                  # Decision error
    'NoBehaviorError',                # No available behavior
    'LearningError',                  # Learning error
    'ParameterUpdateError',           # Parameter update error
    
    # -------------------------------------------------------------------
    # 5. Memory Management
    #    Persistent memory related classes
    # -------------------------------------------------------------------
    'MemoryManager',         # Memory manager
    'InMemoryBackend',       # In-memory backend
    'MemoryBackend',         # Memory backend interface
    'JSONMemoryBackend',     # JSON file backend
    'SQLiteMemoryBackend',   # SQLite database backend
    'TTLMemoryManager',      # TTL memory manager
    
    # -------------------------------------------------------------------
    # 6. Pattern Space Components
    #    Pre-built feature extractors
    # -------------------------------------------------------------------
    'SimplePatternSpace',        # Function-based feature extraction
    'StatisticalPatternSpace',   # Statistical feature extraction
    'NormalizedPatternSpace',    # Normalized feature extraction
    'CompositePatternSpace',     # Composite feature extraction
    'CachedPatternSpace',        # Cached feature extraction
    
    # -------------------------------------------------------------------
    # 7. Meta-Behavior Components
    #    Pre-built skill units
    # -------------------------------------------------------------------
    'FunctionBehavior',      # Function-based behavior
    'LambdaBehavior',        # Lambda behavior
    'CompositeBehavior',     # Composite behavior
    'RetryBehavior',         # Retry behavior
    'TimeoutBehavior',       # Timeout behavior
    'ConditionalBehavior',   # Conditional behavior
    'CachedBehavior',        # Cached behavior
    
    # -------------------------------------------------------------------
    # 8. Selector Components
    #    Pre-built fitness calculators
    # -------------------------------------------------------------------
    'SimpleSelector',        # Simple linear selector
    'PolynomialSelector',    # Polynomial selector
    'RuleBasedSelector',     # Rule-based selector
    'EnsembleSelector',      # Ensemble selector
    'AdaptiveSelector',      # Adaptive selector
    
    # -------------------------------------------------------------------
    # 9. Decision Strategy Components
    #    Pre-built behavior selection strategies
    # -------------------------------------------------------------------
    'MaxFitnessStrategy',    # Maximum fitness strategy
    'ProbabilisticStrategy', # Probabilistic strategy
    'EpsilonGreedyStrategy', # ε-greedy strategy
    'RoundRobinStrategy',    # Round-robin strategy
    'RandomStrategy',        # Random strategy
    'BoltzmannStrategy',     # Boltzmann strategy
    'HybridStrategy',        # Hybrid strategy
    
    # -------------------------------------------------------------------
    # 10. Learning Mechanism Components
    #     Pre-built parameter update algorithms
    # -------------------------------------------------------------------
    'GradientLearning',      # Gradient learning
    'RewardLearning',        # Reward learning
    'MemoryLearning',        # Memory learning
    'HebbianLearning',       # Hebbian learning
    'EnsembleLearning',      # Ensemble learning
    'Experience',            # Experience sample
]