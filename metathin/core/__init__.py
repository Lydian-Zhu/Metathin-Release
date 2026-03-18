"""
Metathin Core Package
=====================

The core layer contains the fundamental interfaces for the quintuple (P, B, S, D, Ψ) 
and the main class implementation. This serves as the foundation of the entire 
framework, defining the behavioral specifications for all core components.

Core Components:
    - Quintuple Interfaces: PatternSpace, MetaBehavior, Selector, DecisionStrategy, LearningMechanism
    - Main Class: Metathin - Agent implementation integrating the quintuple
    - Memory System: Provides persistent memory capabilities
    - Exception Hierarchy: Unified error handling mechanism

Design Principles:
    - Fixed interfaces, free implementation: All core components are abstract base classes
    - Type safety: Uses type aliases and annotations throughout
    - Modularity: Each functionality is independent and replaceable
    - Extensibility: Users can implement custom components by inheriting from interfaces
"""

# ============================================================
# Function Area: Core Interface and Exception Imports
# ============================================================

# -------------------------------------------------------------------
# 1. Type Aliases and Exception Classes
#    These form the foundational type definitions and error handling 
#    mechanisms of the framework
# -------------------------------------------------------------------
from metathin.core.interfaces import (
    # ===== Type Aliases =====
    # These type aliases provide clear semantics and enhance code readability.
    # They serve as documentation for what types are expected in different contexts.
    FeatureVector,      # Type alias for feature vectors: numpy.ndarray
                        # Represents features extracted by PatternSpace from raw input
    
    FitnessScore,       # Type alias for fitness scores: float, range [0,1]
                        # Indicates how suitable a behavior is for current context
    
    ParameterDict,      # Type alias for parameter dictionaries: Dict[str, float]
                        # Used for selector parameter adjustments and learning
    
    # ===== Exception Classes =====
    # Unified exception hierarchy for consistent error handling throughout the framework.
    # All custom exceptions inherit from MetathinError, allowing users to catch
    # all framework-specific exceptions with a single except clause.
    MetathinError,                  # Base exception class, parent of all custom exceptions
    PatternExtractionError,         # Raised when feature extraction fails
    BehaviorExecutionError,         # Raised when behavior execution encounters an exception
    FitnessComputationError,        # Raised when fitness calculation fails
    DecisionError,                  # Raised when decision process encounters an error
    NoBehaviorError,                # Raised when no behaviors are available for selection
    LearningError,                  # Raised when learning process encounters an error
    ParameterUpdateError,           # Raised when parameter update fails
    
    # ===== Core Interfaces =====
    # Abstract base classes for the quintuple. All custom components MUST inherit
    # from these interfaces to ensure compatibility with the framework.
    PatternSpace,       # Pattern Space Interface P: Converts raw input to feature vectors
                        # Must implement extract() method
    
    MetaBehavior,       # Meta-Behavior Interface B: Executable skill units
                        # Must implement execute() method, can optionally implement
                        # before_execute(), after_execute(), and on_error() hooks
    
    Selector,           # Selector Interface S: Evaluates behavior suitability
                        # Computes fitness scores for each behavior based on features
                        # Must implement compute_fitness() method
    
    DecisionStrategy,   # Decision Strategy Interface D: Selects optimal behavior
                        # Makes decisions based on fitness scores
                        # Must implement select() method
    
    LearningMechanism,  # Learning Mechanism Interface Ψ: Adjusts parameters based on feedback
                        # Updates selector parameters using expected vs actual results
                        # Must implement compute_adjustment() method
)

# ============================================================
# Function Area: Main Class Implementation Imports
# ============================================================

# -------------------------------------------------------------------
# 2. Metathin Main Class and Helper Classes
#    These are the core implementations that integrate the quintuple
#    to complete the agent's cognitive cycle
# -------------------------------------------------------------------
from metathin.core.metathin import (
    Metathin,           # Main class: Agent implementation integrating the quintuple (P,B,S,D,Ψ)
                        # Provides think() method to execute a complete cognitive cycle:
                        # Perceive → Hypothesize → Decide → Execute → Learn
                        # Also provides memory functions (remember/recall) and persistence (save/load)
    
    Thought,            # Thought record class: Records a complete thinking process
                        # Contains input, output, decision information, timings, etc.
                        # Used for debugging, analysis, and visualization
                        # Attributes include: features, selected_behavior, fitness_scores,
                        # decision_time, execution_time, success, error_message, etc.
    
    ThinkingStage,      # Thinking stage enumeration: PERCEIVE, HYPOTHESIS, DECIDE, EXECUTE, LEARN
                        # Tracks the agent's current stage during thinking process
                        # Useful for progress monitoring and error localization
    
    LearningStatus,     # Learning status enumeration: SUCCESS, FAILED, SKIPPED
                        # Indicates the outcome of learning operations
                        # SUCCESS: parameters were updated, FAILED: exception occurred,
                        # SKIPPED: learning conditions were not met
    
    MetathinConfig,     # Configuration class: Controls agent behavior parameters
                        # Includes learning rate, fitness threshold, memory configuration,
                        # logging settings, error handling behavior, etc.
                        # All parameters have sensible defaults but can be overridden
)

# ============================================================
# Function Area: Memory System Imports
# ============================================================

# -------------------------------------------------------------------
# 3. Memory System Components
#    Provides persistent memory capabilities for the agent,
#    supporting multiple storage backends with different characteristics
# -------------------------------------------------------------------
from metathin.core.memory import (
    # ===== Memory Backend Interface and Implementations =====
    # Backends handle the actual storage of memory data
    MemoryBackend,       # Memory backend abstract interface: Defines save, load, delete methods
                         # All concrete backends must implement these methods
    
    InMemoryBackend,     # In-memory backend: Stores data only in RAM
                         # Fastest but non-persistent, lost when program exits
                         # Suitable for temporary caching and testing
    
    JSONMemoryBackend,   # JSON file backend: Saves memory as human-readable JSON format
                         # Suitable for small amounts of data and debugging
                         # Easy to inspect and modify manually
    
    SQLiteMemoryBackend, # SQLite database backend: Supports large amounts of data and transactions
                         # Suitable for production environments requiring persistence
                         # Provides ACID guarantees and concurrent access
    
    # ===== Memory Managers =====
    # Managers provide higher-level memory functionality with caching
    MemoryManager,       # Memory manager: Provides two-tier memory architecture
                         # Combines fast in-memory cache with persistent backend storage
                         # Automatically handles cache hits/misses and synchronization
    
    TTLMemoryManager,    # TTL (Time-To-Live) memory manager: Memory with expiration
                         # Each memory item has a time-to-live value
                         # Automatically cleans up expired items
                         # Useful for temporary or time-sensitive information
)

# ============================================================
# Export Interface Definition
# ============================================================
# __all__ controls what gets imported when using 'from metathin.core import *'
# It's grouped by functionality for easier user reference
# ============================================================

__all__ = [
    # -------------------------------------------------------------------
    # 1. Type Aliases
    #    Used for type hints, improving code readability and IDE support
    # -------------------------------------------------------------------
    'FeatureVector',        # Feature vector type (numpy.ndarray)
    'FitnessScore',         # Fitness score type (float in [0,1])
    'ParameterDict',        # Parameter dictionary type (Dict[str, float])
    
    # -------------------------------------------------------------------
    # 2. Exception Classes
    #    Unified error handling mechanism
    # -------------------------------------------------------------------
    'MetathinError',                 # Base exception for all framework errors
    'PatternExtractionError',        # Error during feature extraction
    'BehaviorExecutionError',        # Error during behavior execution
    'FitnessComputationError',       # Error during fitness calculation
    'DecisionError',                  # Error during decision making
    'NoBehaviorError',                # No behaviors available for selection
    'LearningError',                  # Error during learning process
    'ParameterUpdateError',           # Error during parameter update
    
    # -------------------------------------------------------------------
    # 3. Core Interfaces
    #    Abstract base classes for the quintuple. Custom components MUST
    #    inherit from these interfaces.
    # -------------------------------------------------------------------
    'PatternSpace',         # Pattern Space Interface (P) - Feature extraction
    'MetaBehavior',         # Meta-Behavior Interface (B) - Executable skills
    'Selector',             # Selector Interface (S) - Fitness computation
    'DecisionStrategy',     # Decision Strategy Interface (D) - Behavior selection
    'LearningMechanism',    # Learning Mechanism Interface (Ψ) - Parameter adjustment
    
    # -------------------------------------------------------------------
    # 4. Main Classes
    #    Core implementations of the agent
    # -------------------------------------------------------------------
    'Metathin',             # Main agent class - integrates the quintuple
    'Thought',              # Thought record - captures complete thinking process
    'ThinkingStage',        # Thinking stage enumeration - tracks progress
    'LearningStatus',       # Learning status enumeration - records learning outcome
    'MetathinConfig',       # Configuration class - controls agent behavior
    
    # -------------------------------------------------------------------
    # 5. Memory Components
    #    Persistent memory related components
    # -------------------------------------------------------------------
    'MemoryBackend',        # Memory backend interface - defines storage contract
    'InMemoryBackend',      # In-memory backend implementation (fast, non-persistent)
    'JSONMemoryBackend',    # JSON file backend implementation (human-readable)
    'SQLiteMemoryBackend',  # SQLite database backend (production-ready)
    'MemoryManager',        # Memory manager - two-tier caching system
    'TTLMemoryManager',     # TTL memory manager - auto-expiring memory items
]

# ============================================================
# Usage Example
# ============================================================
"""
Basic usage example demonstrating how to create a simple agent:

>>> import numpy as np
>>> from metathin.core import Metathin, PatternSpace, MetaBehavior
>>> 
>>> # Custom PatternSpace implementation
>>> # Converts input string to its length as a feature vector
>>> class MyPattern(PatternSpace):
...     def extract(self, raw_input):
...         # Simple feature: length of input string
...         return np.array([len(raw_input)], dtype=np.float64)
>>> 
>>> # Custom MetaBehavior implementation
>>> # A simple greeting behavior
>>> class GreetBehavior(MetaBehavior):
...     def __init__(self):
...         self.name = "greet"
...     
...     def execute(self, features, **context):
...         return f"Hello, input length = {features[0]}"
...     
...     def get_complexity(self):
...         return 1.0
>>> 
>>> # Create agent with custom pattern space
>>> agent = Metathin(pattern_space=MyPattern())
>>> 
>>> # Register behavior
>>> agent.register_behavior(GreetBehavior())
>>> 
>>> # Execute thinking cycle
>>> result = agent.think("hello world")
>>> print(result)  # Output: "Hello, input length = 11"
>>> 
>>> # Check agent status
>>> status = agent.get_status()
>>> print(f"Total thoughts: {status['stats']['total_thoughts']}")
"""