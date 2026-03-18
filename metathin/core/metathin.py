"""
Metathin Main Class Implementation
==================================

Integrates the quintuple (P, B, S, D, Ψ) to implement a complete cognitive cycle.
Provides core functionalities such as thinking, learning, and memory.

Design Philosophy:
    - Modular: Components are decoupled through interfaces, allowing independent replacement
    - Extensible: Supports custom components for various application scenarios
    - Observable: Comprehensive logging and statistics for debugging and analysis
    - Safe: Robust error handling and type checking ensure stability

Cognitive Cycle Flow:
    1. Perceive: Extract features through PatternSpace
    2. Hypothesize: Calculate fitness for each behavior via Selector
    3. Decide: Select optimal behavior through DecisionStrategy
    4. Execute: Run the chosen MetaBehavior
    5. Learn: Update parameters via LearningMechanism
"""

import time
import uuid
import logging
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
import pickle
import json
import traceback

import numpy as np

from .interfaces import (
    PatternSpace, MetaBehavior, Selector, DecisionStrategy, LearningMechanism,
    FeatureVector, ParameterDict,
    MetathinError, NoBehaviorError, PatternExtractionError,
    BehaviorExecutionError, FitnessComputationError, DecisionError, LearningError
)
from .memory import MemoryManager, JSONMemoryBackend, MemoryBackend, SQLiteMemoryBackend


# ============================================================
# Constants
# ============================================================
# These constants control the default behavior of the agent and can be overridden by user configuration
# ============================================================

DEFAULT_LEARNING_RATE = 0.1
"""Default learning rate - Controls parameter update speed. Higher values learn faster but may be unstable"""

DEFAULT_MIN_FITNESS = 0.0
"""Default minimum fitness threshold - Behaviors below this value will not be considered"""

MAX_WEIGHT_VALUE = 20.0
"""Maximum weight value - Prevents gradient explosion by limiting parameter range"""

DEFAULT_HISTORY_SIZE = 1000
"""Default history size - Keeps recent thought records for analysis and debugging"""


# ============================================================
# Enum Definitions
# ============================================================
# These enums track the agent's state and stages during the thinking process
# ============================================================

class ThinkingStage(Enum):
    """
    Thinking stage enumeration for tracking the cognitive process.
    
    Each stage corresponds to a step in the cognitive cycle:
    - PERCEIVE: Feature extraction stage
    - HYPOTHESIS: Fitness computation stage
    - DECIDE: Behavior selection stage
    - EXECUTE: Behavior execution stage
    - LEARN: Parameter learning stage
    - COMPLETE: Thinking completed
    - ERROR: Error occurred
    """
    PERCEIVE = "perceive"      # Perceive: extract features from raw input
    HYPOTHESIS = "hypothesis"   # Hypothesize: calculate fitness scores
    DECIDE = "decide"          # Decide: select optimal behavior
    EXECUTE = "execute"        # Execute: run the chosen behavior
    LEARN = "learn"            # Learn: update parameters based on feedback
    COMPLETE = "complete"      # Complete: thinking cycle finished successfully
    ERROR = "error"            # Error: exception occurred during thinking


class LearningStatus(Enum):
    """Learning status enumeration, recording the outcome of learning operations"""
    SUCCESS = "success"        # Learning succeeded: parameters were updated
    FAILED = "failed"          # Learning failed: an exception occurred
    SKIPPED = "skipped"        # Learning skipped: conditions not met


# ============================================================
# Data Class Definitions
# ============================================================
# These classes encapsulate the agent's state and configuration information
# ============================================================

@dataclass
class Thought:
    """
    Thought record data class.
    
    Records a complete thinking process, including inputs, outputs, decisions, etc.
    Used for debugging, analysis, and visualization - essentially the agent's "thought log".
    
    Attribute Groups:
        - Basic Information: Unique ID, timestamp, etc.
        - Input/Output: Raw input, feature vector, execution result
        - Decision Information: Selected behavior, fitness scores, etc.
        - Learning Information: Learning status, parameter changes
        - Status Information: Success/failure, error details
    """
    
    # ===== Basic Information =====
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    """Unique identifier for distinguishing different thought records"""
    
    timestamp: datetime = field(default_factory=datetime.now)
    """Thought start time for temporal analysis and performance evaluation"""
    
    stage: ThinkingStage = ThinkingStage.PERCEIVE
    """Current thinking stage, useful for tracking execution progress"""
    
    # ===== Input/Output =====
    raw_input: Any = None
    """Original input data, preserving the raw information provided by the user"""
    
    features: Optional[FeatureVector] = None
    """Extracted feature vector, output from the pattern space"""
    
    result: Any = None
    """Execution result, return value of the selected behavior"""
    
    expected: Any = None
    """Expected result, used for error calculation during learning"""
    
    # ===== Decision Information =====
    selected_behavior: Optional[str] = None
    """Name of the selected behavior, recording the decision outcome"""
    
    fitness_scores: Dict[str, float] = field(default_factory=dict)
    """Fitness scores for all candidate behaviors, reflecting evaluation results"""
    
    candidate_behaviors: List[str] = field(default_factory=list)
    """List of candidate behavior names, recording all possible choices"""
    
    decision_time: float = 0.0
    """Decision duration (seconds), used for performance analysis"""
    
    execution_time: float = 0.0
    """Execution duration (seconds), used for performance analysis"""
    
    total_time: float = 0.0
    """Total duration (seconds), time for the complete thinking cycle"""
    
    # ===== Learning Information =====
    learning_status: LearningStatus = LearningStatus.SKIPPED
    """Learning status, recording the outcome of the learning operation"""
    
    parameter_changes: Optional[ParameterDict] = None
    """Parameter changes, recording adjustments made during learning"""
    
    # ===== Status Information =====
    success: bool = True
    """Whether the overall thinking process completed successfully"""
    
    error_message: Optional[str] = None
    """Error message, description of the exception when failed"""
    
    error_traceback: Optional[str] = None
    """Error traceback, for debugging purposes"""
    
    # ===== Metadata =====
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata, user-customizable extra information"""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for serialization)"""
        data = asdict(self)
        # Handle special types: numpy arrays, datetime, enums need special conversion
        if self.features is not None:
            data['features'] = self.features.tolist()  # Convert numpy array to list
        data['timestamp'] = self.timestamp.isoformat()  # Convert datetime to ISO string
        data['stage'] = self.stage.value  # Convert enum to string
        data['learning_status'] = self.learning_status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Thought':
        """Create from dictionary (for deserialization)"""
        # Restore special types
        if 'features' in data and data['features'] is not None:
            data['features'] = np.array(data['features'], dtype=np.float64)
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['stage'] = ThinkingStage(data['stage'])
        data['learning_status'] = LearningStatus(data['learning_status'])
        return cls(**data)


@dataclass
class MetathinConfig:
    """
    Agent configuration class.
    
    Controls all aspects of the agent's behavior: learning, decision-making, storage, etc.
    All configuration items have default values and can be overridden by the user.
    
    Configuration Groups:
        - Learning Configuration: Controls learning behavior
        - Decision Configuration: Controls decision behavior
        - History Configuration: Controls historical record keeping
        - Memory Configuration: Controls persistent memory
        - Logging Configuration: Controls log output
        - Error Handling: Controls exception behavior
        - Debug Configuration: Controls debug information
    """
    
    # ===== Learning Configuration =====
    learning_rate: float = DEFAULT_LEARNING_RATE
    """Learning rate, affects parameter update speed (0.01-0.5 recommended)"""
    
    enable_learning: bool = True
    """Whether to enable learning functionality. If False, the agent won't update parameters"""
    
    weight_decay: float = 0.01
    """Weight decay coefficient for regularization, prevents overfitting"""
    
    # ===== Decision Configuration =====
    min_fitness_threshold: float = DEFAULT_MIN_FITNESS
    """Minimum fitness threshold. Behaviors below this won't be considered (0-1 range)"""
    
    # ===== History Configuration =====
    keep_history: bool = True
    """Whether to keep thought history for analysis and debugging"""
    
    max_history_size: Optional[int] = DEFAULT_HISTORY_SIZE
    """Maximum number of history records. None means unlimited (prevents memory overflow)"""
    
    # ===== Memory Configuration =====
    enable_memory: bool = True
    """Whether to enable memory functionality. If False, the agent has no persistent memory"""
    
    memory_backend: str = 'json'
    """Memory backend type: 'json' (human-readable) or 'sqlite' (better performance)"""
    
    memory_path: Optional[str] = None
    """Memory storage path. None means use default path (current directory)"""
    
    auto_save: bool = False
    """Whether to automatically save state for checkpoint/resume training"""
    
    auto_save_interval: int = 100
    """Auto-save interval (number of thoughts). Saves automatically every N thoughts"""
    
    # ===== Logging Configuration =====
    log_level: int = logging.INFO
    """Log level: DEBUG, INFO, WARNING, ERROR"""
    
    log_file: Optional[str] = None
    """Log file path. None means output to console only"""
    
    enable_performance_logging: bool = False
    """Whether to log performance metrics like stage durations"""
    
    # ===== Error Handling =====
    raise_on_error: bool = True
    """Whether to raise exceptions when errors occur. If False, returns None"""
    
    retry_on_error: bool = False
    """Whether to automatically retry failed behaviors"""
    
    max_retries: int = 3
    """Maximum retry attempts when retry_on_error is True"""
    
    # ===== Debug Configuration =====
    debug_mode: bool = False
    """Debug mode. Logs more information (like error tracebacks)"""
    
    validate_inputs: bool = True
    """Whether to validate input data, checking for None, NaN, etc."""
    
    def __post_init__(self):
        """Post-initialization validation to ensure configuration parameters are valid"""
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0")
        if self.min_fitness_threshold < 0 or self.min_fitness_threshold > 1:
            raise ValueError("min_fitness_threshold must be in range [0, 1]")
        if self.max_history_size is not None and self.max_history_size <= 0:
            raise ValueError("max_history_size must be greater than 0")


# ============================================================
# Main Class Implementation
# ============================================================

class Metathin:
    """
    Metathin Main Class - Core implementation of the cognitive agent.
    
    Contains the quintuple (P, B, S, D, Ψ) along with memory functionality.
    Provides complete thinking cycle and extensibility capabilities.
    
    Core Functions:
        - think(): Execute a complete cognitive cycle
        - register_behavior(): Register new behaviors
        - remember()/recall(): Memory functions
        - save()/load(): State persistence
    
    Usage Example:
        >>> # Create basic components
        >>> pattern = SimplePatternSpace(lambda x: [len(str(x))])
        >>> selector = SimpleSelector(n_features=1, n_behaviors=2)
        >>> strategy = MaxFitnessStrategy()
        >>> learner = GradientLearning(learning_rate=0.01)
        >>> 
        >>> # Create agent
        >>> agent = Metathin(
        ...     pattern_space=pattern,
        ...     selector=selector,
        ...     decision_strategy=strategy,
        ...     learning_mechanism=learner,
        ...     name="MyAgent"
        ... )
        >>> 
        >>> # Register behaviors
        >>> agent.register_behavior(FunctionBehavior("greet", lambda f,**k: "Hello"))
        >>> agent.register_behavior(FunctionBehavior("bye", lambda f,**k: "Goodbye"))
        >>> 
        >>> # Think
        >>> result = agent.think("input")
        >>> print(result)
    """
    
    def __init__(self,
                 pattern_space: PatternSpace,
                 selector: Optional[Selector] = None,
                 decision_strategy: Optional[DecisionStrategy] = None,
                 learning_mechanism: Optional[LearningMechanism] = None,
                 config: Optional[MetathinConfig] = None,
                 name: Optional[str] = None,
                 memory_backend: Optional[MemoryBackend] = None):
        """
        Initialize a Metathin instance.
        
        Args:
            pattern_space: Pattern space (must implement extract method)
            selector: Selector (optional, evaluates behavior fitness)
            decision_strategy: Decision strategy (optional, selects optimal behavior)
            learning_mechanism: Learning mechanism (optional, updates parameters)
            config: Configuration object (optional, controls agent behavior)
            name: Instance name (optional, used for logging and identification)
            memory_backend: Memory backend (optional, takes precedence over config.memory_backend)
            
        Raises:
            TypeError: When component types are incorrect
            ValueError: When configuration parameters are invalid
        """
        # ====================================================
        # Validate core components
        # ====================================================
        self._validate_pattern_space(pattern_space)
        
        # ====================================================
        # Core Quintuple (P, B, S, D, Ψ)
        # ====================================================
        self.P = pattern_space
        """Pattern space P: Window to perceive the environment, converts raw input to feature vectors"""
        
        self.B: List[MetaBehavior] = []
        """Behavior library B: Skill collection, stores all executable behaviors"""
        
        self.S = selector
        """Selector S: Evaluates behavior suitability, computes fitness for each behavior"""
        
        self.D = decision_strategy
        """Decision strategy D: Selects optimal behavior based on fitness scores"""
        
        self.Ψ = learning_mechanism
        """Learning mechanism Ψ: Parameter adjustment, updates selector parameters based on feedback"""
        
        # ====================================================
        # Configuration
        # ====================================================
        self.config = config or MetathinConfig()
        """Agent configuration, controls learning, memory, logging, etc."""
        
        self.name = name or f"Metathin-{uuid.uuid4().hex[:8]}"
        """Agent name, used for logging and identification"""
        
        # ====================================================
        # Logging System
        # ====================================================
        self.logger = self._setup_logger()
        """Logger instance for outputting debug and status information"""
        
        # ====================================================
        # State Management
        # ====================================================
        self._history: List[Thought] = []
        """Thought history, stores recent thinking processes"""
        
        self._is_thinking = False
        """Whether currently thinking (prevents reentrancy), avoids concurrent calls"""
        
        self._thinking_count = 0
        """Thinking counter, used for statistics and auto-save"""
        
        self._last_thought_time: Optional[datetime] = None
        """Last thought timestamp, for temporal analysis"""
        
        self._error_count = 0
        """Error counter, used for monitoring system health"""
        
        # ====================================================
        # Statistics
        # ====================================================
        self.stats = self._init_stats()
        """Runtime statistics, records performance and usage information"""
        
        # ====================================================
        # Initialize Memory
        # ====================================================
        self._init_memory(memory_backend)
        
        self.logger.info(f"✅ Metathin '{self.name}' initialized successfully")
        self.logger.debug(f"    Configuration: {self.config}")
        self.logger.debug(f"    Components: P={type(self.P).__name__}, S={type(self.S).__name__ if self.S else 'None'}, "
                         f"D={type(self.D).__name__ if self.D else 'None'}, Ψ={type(self.Ψ).__name__ if self.Ψ else 'None'}")
    
    # ========================================================
    # Initialization Helper Methods
    # ========================================================
    
    def _validate_pattern_space(self, pattern_space: PatternSpace) -> None:
        """
        Validate pattern space validity.
        
        Args:
            pattern_space: Pattern space to validate
            
        Raises:
            TypeError: When pattern space is invalid
        """
        if not isinstance(pattern_space, PatternSpace):
            raise TypeError(f"pattern_space must be an instance of PatternSpace, got {type(pattern_space)}")
        
        # Check required methods
        if not hasattr(pattern_space, 'extract'):
            raise TypeError("pattern_space must implement extract method")
        
        if not callable(getattr(pattern_space, 'extract')):
            raise TypeError("pattern_space.extract must be callable")
    
    def _setup_logger(self) -> logging.Logger:
        """
        Set up logger instance.
        
        Creates console and file log handlers based on configuration.
        
        Returns:
            logging.Logger: Configured logger instance
        """
        logger = logging.getLogger(f"metathin.{self.name}")
        
        # Avoid duplicate handlers
        if logger.handlers:
            return logger
        
        # Set log format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler (if configured)
        if self.config and self.config.log_file:
            try:
                file_handler = logging.FileHandler(self.config.log_file, encoding='utf-8')
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except Exception as e:
                print(f"Cannot create log file: {e}")
        
        logger.setLevel(self.config.log_level if self.config else logging.INFO)
        
        return logger
    
    def _init_stats(self) -> Dict[str, Any]:
        """
        Initialize statistics dictionary.
        
        Returns:
            Dict: Statistics dictionary tracking the agent's runtime state
        """
        return {
            'total_thoughts': 0,           # Total number of thoughts
            'successful_thoughts': 0,       # Number of successful thoughts
            'failed_thoughts': 0,           # Number of failed thoughts
            'learn_attempts': 0,            # Number of learning attempts
            'learn_success': 0,              # Number of successful learning operations
            'avg_decision_time': 0.0,        # Average decision time
            'avg_execution_time': 0.0,       # Average execution time
            'total_time': 0.0,                # Total time spent
            'behaviors_used': {},             # Behavior usage count statistics
            'behaviors_success': {},          # Behavior success count statistics
            'behaviors_failure': {},          # Behavior failure count statistics
            'start_time': datetime.now().isoformat(),  # Start time
            'last_thought_time': None,        # Last thought timestamp
            'peak_memory_usage': 0,           # Peak memory usage (reserved for future use)
        }
    
    def _init_memory(self, memory_backend: Optional[MemoryBackend] = None) -> None:
        """
        Initialize memory manager.
        
        Design considerations:
            - Even if config.enable_memory is False, initialize if memory_backend is provided
            - Ensure self.memory is always properly set, avoiding "memory not enabled" warnings
            - Comprehensive error handling and logging
        
        Args:
            memory_backend: Custom memory backend, takes precedence over configuration
        """
        # Case 1: No backend provided and memory disabled by config -> don't initialize
        if not self.config.enable_memory and memory_backend is None:
            self.memory = None
            self.logger.debug("Memory functionality disabled (by configuration)")
            return
        
        try:
            # Case 2: Use custom backend (highest priority)
            if memory_backend:
                # Validate backend type
                if not isinstance(memory_backend, MemoryBackend):
                    raise TypeError(f"memory_backend must be an instance of MemoryBackend, got {type(memory_backend)}")
                
                self.memory = MemoryManager(backend=memory_backend)
                self.logger.info(f"✅ Using custom memory backend: {type(memory_backend).__name__}")
                self.logger.info("✅ Memory manager initialized successfully")
                return
            
            # Case 3: Create backend based on configuration
            if not self.config.enable_memory:
                self.memory = None
                self.logger.debug("Memory functionality disabled (by configuration)")
                return
            
            # Select backend type based on configuration
            if self.config.memory_backend == 'sqlite':
                db_path = self.config.memory_path or f"metathin_{self.name}.db"
                try:
                    from .memory import SQLiteMemoryBackend
                    backend = SQLiteMemoryBackend(db_path)
                    self.logger.info(f"📁 Using SQLite memory backend: {db_path}")
                except ImportError:
                    self.logger.warning("⚠️ SQLite not available, falling back to JSON backend")
                    json_path = self.config.memory_path or f"metathin_{self.name}_memory.json"
                    backend = JSONMemoryBackend(json_path)
            else:  # Default to JSON
                json_path = self.config.memory_path or f"metathin_{self.name}_memory.json"
                backend = JSONMemoryBackend(json_path)
                self.logger.info(f"📁 Using JSON memory backend: {json_path}")
            
            # Create memory manager
            self.memory = MemoryManager(backend=backend)
            self.logger.info("✅ Memory manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize memory: {e}")
            self.memory = None
            # If configuration requires raising exceptions, do so
            if self.config.raise_on_error:
                raise
    
    # ========================================================
    # Behavior Management
    # ========================================================
    
    def register_behavior(self, behavior: MetaBehavior) -> 'Metathin':
        """
        Register a single behavior.
        
        Adds the behavior to the behavior library and initializes statistics.
        
        Args:
            behavior: Behavior instance to register
            
        Returns:
            self: Supports method chaining
            
        Raises:
            TypeError: When behavior type is incorrect
            ValueError: When behavior name already exists
        """
        # Validate behavior
        if not isinstance(behavior, MetaBehavior):
            raise TypeError(f"Behavior must inherit from MetaBehavior, got {type(behavior)}")
        
        if not hasattr(behavior, 'name'):
            raise TypeError("Behavior must have a name attribute")
        
        if not isinstance(behavior.name, str) or not behavior.name:
            raise ValueError("Behavior name must be a valid string")
        
        # Check name uniqueness
        if any(b.name == behavior.name for b in self.B):
            raise ValueError(f"Behavior name '{behavior.name}' already exists")
        
        # Register behavior
        self.B.append(behavior)
        
        # Initialize statistics
        self.stats['behaviors_used'][behavior.name] = 0
        self.stats['behaviors_success'][behavior.name] = 0
        self.stats['behaviors_failure'][behavior.name] = 0
        
        self.logger.info(f"✅ Registered behavior: {behavior.name} (complexity: {behavior.get_complexity()})")
        
        return self
    
    def register_behaviors(self, behaviors: List[MetaBehavior]) -> 'Metathin':
        """
        Batch register behaviors.
        
        Args:
            behaviors: List of behaviors to register
            
        Returns:
            self: Supports method chaining
        """
        for behavior in behaviors:
            self.register_behavior(behavior)
        
        self.logger.info(f"✅ Batch registration of {len(behaviors)} behaviors completed")
        
        return self
    
    def unregister_behavior(self, behavior_name: str) -> bool:
        """
        Unregister a behavior.
        
        Args:
            behavior_name: Name of the behavior to unregister
            
        Returns:
            bool: Whether unregistration was successful
        """
        for i, behavior in enumerate(self.B):
            if behavior.name == behavior_name:
                self.B.pop(i)
                self.logger.info(f"🗑️ Unregistered behavior: {behavior_name}")
                return True
        
        self.logger.warning(f"⚠️ Behavior not found: {behavior_name}")
        return False
    
    def get_behavior(self, behavior_name: str) -> Optional[MetaBehavior]:
        """
        Get behavior by name.
        
        Args:
            behavior_name: Name of the behavior
            
        Returns:
            Optional[MetaBehavior]: Found behavior, None if not exists
        """
        for behavior in self.B:
            if behavior.name == behavior_name:
                return behavior
        return None
    
    def list_behaviors(self) -> List[Dict[str, Any]]:
        """
        List information about all behaviors.
        
        Returns:
            List[Dict]: Brief information for each behavior
        """
        return [
            {
                'name': b.name,
                'type': type(b).__name__,
                'complexity': b.get_complexity(),
                'execution_count': b._execution_count,
                'can_execute': True  # Needs dynamic evaluation
            }
            for b in self.B
        ]
    
    # ========================================================
    # Core Thinking Cycle
    # ========================================================
    
    def think(self, 
              raw_input: Any, 
              expected: Any = None,
              **context) -> Any:
        """
        Execute a complete thinking cycle.
        
        Flow:
            1. Perceive: Extract features from raw input
            2. Hypothesize: Calculate fitness scores for each behavior
            3. Decide: Select the optimal behavior based on fitness
            4. Execute: Run the selected behavior
            5. Learn: Update parameters based on expected vs actual results
        
        Args:
            raw_input: Raw input data to process
            expected: Expected result (used for learning)
            **context: Additional context parameters passed to behaviors
            
        Returns:
            Any: Result of behavior execution
            
        Raises:
            MetathinError: When errors occur during thinking and raise_on_error is True
        """
        # Prevent reentrancy
        if self._is_thinking:
            raise MetathinError("Agent is already thinking, cannot re-enter")
        
        self._is_thinking = True
        start_time = time.time()
        
        # Create thought record
        thought = self._create_thought(raw_input, expected, context)
        
        try:
            # ================================================
            # 1. Perceive Stage: Extract features
            # ================================================
            features = self._perceive(raw_input, thought)
            
            # ================================================
            # 2. Hypothesize Stage: Calculate fitness
            # ================================================
            behaviors, scores = self._hypothesize(features, thought)
            
            # ================================================
            # 3. Decide Stage: Select behavior
            # ================================================
            selected = self._decide(behaviors, scores, features, thought)
            
            # ================================================
            # 4. Execute Stage: Run behavior
            # ================================================
            exec_start = time.time()
            result = self._execute(selected, features, context, thought)
            thought.execution_time = time.time() - exec_start
            
            # ================================================
            # 5. Learn Stage: Update parameters
            # ================================================
            if expected is not None and self.config.enable_learning:
                self._learn(selected, features, expected, result, thought)
            
            # Record results
            thought.result = result
            thought.success = True
            thought.total_time = time.time() - start_time
            
            # Update statistics
            self._update_stats(thought, selected, True)
            
            self.logger.debug(f"✅ Thinking completed: input={raw_input}, output={result}, duration={thought.total_time:.3f}s")
            
            return result
            
        except Exception as e:
            # Handle errors
            thought.success = False
            thought.error_message = str(e)
            thought.error_traceback = traceback.format_exc()
            thought.total_time = time.time() - start_time
            
            self.stats['failed_thoughts'] += 1
            self._error_count += 1
            
            self.logger.error(f"❌ Thinking failed: {e}")
            self.logger.debug(traceback.format_exc())
            
            if self.config.raise_on_error:
                raise
            
            return None
            
        finally:
            # Save history
            if self.config.keep_history:
                self._add_to_history(thought)
            
            # Auto-save
            if (self.config.auto_save and 
                self._thinking_count > 0 and 
                self._thinking_count % self.config.auto_save_interval == 0):
                self._auto_save()
            
            self._is_thinking = False
            self._thinking_count += 1
            self._last_thought_time = datetime.now()
    
    def _create_thought(self, raw_input: Any, expected: Any, context: Dict) -> Thought:
        """
        Create a thought record.
        
        Args:
            raw_input: Raw input
            expected: Expected result
            context: Context dictionary
            
        Returns:
            Thought: Thought record object
        """
        return Thought(
            raw_input=raw_input,
            expected=expected,
            metadata=context.copy() if context else {}
        )
    
    def _perceive(self, raw_input: Any, thought: Thought) -> FeatureVector:
        """
        Perceive stage: Extract features from raw input.
        
        Calls the pattern space to extract features and performs validation.
        
        Args:
            raw_input: Raw input data
            thought: Thought record
            
        Returns:
            FeatureVector: Extracted feature vector
            
        Raises:
            PatternExtractionError: When feature extraction fails
        """
        thought.stage = ThinkingStage.PERCEIVE
        
        self.logger.debug(f"🔍 Perceive stage: input type={type(raw_input).__name__}")
        
        try:
            # Validate input
            if self.config.validate_inputs and raw_input is None:
                raise ValueError("Input cannot be None")
            
            # Extract features
            features = self.P.extract(raw_input)
            
            # Validate features
            if not isinstance(features, np.ndarray):
                features = np.array(features, dtype=np.float64)
            
            if features.ndim != 1:
                self.logger.warning(f"Feature vector is not 1-dimensional, flattening: {features.ndim}D -> 1D")
                features = features.flatten()
            
            if features.dtype != np.float64:
                features = features.astype(np.float64)
            
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                raise ValueError("Feature vector contains NaN or Inf")
            
            thought.features = features
            
            self.logger.debug(f"  Feature extraction successful: dimensions={len(features)}, features={features}")
            
            return features
            
        except Exception as e:
            error_msg = f"Feature extraction failed: {e}"
            self.logger.error(error_msg)
            raise PatternExtractionError(error_msg) from e
    
    def _hypothesize(self, 
                     features: FeatureVector, 
                     thought: Thought) -> Tuple[List[MetaBehavior], List[float]]:
        """
        Hypothesize stage: Calculate fitness scores for each behavior.
        
        Evaluates fitness for each behavior and filters those below threshold.
        
        Args:
            features: Feature vector
            thought: Thought record
            
        Returns:
            Tuple[List[MetaBehavior], List[float]]: Candidate behaviors and corresponding fitness scores
            
        Raises:
            NoBehaviorError: When no behaviors are available
        """
        thought.stage = ThinkingStage.HYPOTHESIS
        
        self.logger.debug(f"📊 Hypothesize stage: evaluating {len(self.B)} behaviors")
        
        valid_behaviors = []
        fitness_scores = []
        
        # If no behaviors registered, raise exception
        if not self.B:
            error_msg = "No behaviors registered"
            self.logger.warning(error_msg)
            raise NoBehaviorError(error_msg)
        
        # Evaluate each behavior
        for behavior in self.B:
            # Check if behavior can execute
            try:
                if hasattr(behavior, 'can_execute') and callable(behavior.can_execute):
                    if not behavior.can_execute(features):
                        self.logger.debug(f"  Behavior {behavior.name} cannot execute")
                        continue
            except Exception as e:
                self.logger.debug(f"  Error checking behavior {behavior.name} executability: {e}")
                continue
            
            # Compute fitness
            try:
                if self.S:
                    fitness = self.S.compute_fitness(behavior, features)
                else:
                    # Default fitness - all behaviors get default value
                    fitness = 0.5
                
                # Record fitness
                if self.S:
                    self.S.record_fitness(behavior.name, fitness)
                
                # Check threshold
                if fitness >= self.config.min_fitness_threshold:
                    valid_behaviors.append(behavior)
                    fitness_scores.append(fitness)
                    thought.fitness_scores[behavior.name] = fitness
                    thought.candidate_behaviors.append(behavior.name)
                    
                    self.logger.debug(f"  ✅ {behavior.name}: fitness={fitness:.3f}")
                else:
                    self.logger.debug(f"  ⚠️ {behavior.name}: fitness={fitness:.3f} (below threshold {self.config.min_fitness_threshold})")
                    
            except Exception as e:
                self.logger.debug(f"  Error computing fitness for behavior {behavior.name}: {e}")
                continue
        
        # Check if any behaviors are available
        if not valid_behaviors:
            # Even if no behaviors pass the threshold, if at least one behavior exists,
            # we can still select one
            if self.B and self.config.min_fitness_threshold > 0:
                self.logger.warning(f"All behaviors below threshold {self.config.min_fitness_threshold}, using default selection")
                # Return all behaviors but mark them with low fitness
                for behavior in self.B:
                    valid_behaviors.append(behavior)
                    fitness_scores.append(0.1)  # Give a low but non-zero fitness
                    thought.fitness_scores[behavior.name] = 0.1
                    thought.candidate_behaviors.append(behavior.name)
            else:
                error_msg = f"No behaviors available (evaluated {len(self.B)} behaviors)"
                self.logger.warning(error_msg)
                raise NoBehaviorError(error_msg)
        
        self.logger.debug(f"  Candidate behaviors: {len(valid_behaviors)}/{len(self.B)}")
        
        return valid_behaviors, fitness_scores
    
    def _decide(self, 
                behaviors: List[MetaBehavior], 
                fitness_scores: List[float],
                features: FeatureVector,
                thought: Thought) -> MetaBehavior:
        """
        Decide stage: Select optimal behavior.
        
        Uses decision strategy to select from candidate behaviors.
        
        Args:
            behaviors: List of candidate behaviors
            fitness_scores: Corresponding fitness scores
            features: Current feature vector
            thought: Thought record
            
        Returns:
            MetaBehavior: Selected behavior
            
        Raises:
            DecisionError: When decision fails
        """
        thought.stage = ThinkingStage.DECIDE
        start = time.time()
        
        self.logger.debug(f"🎯 Decide stage: selecting from {len(behaviors)} candidates")
        
        try:
            if self.D:
                selected = self.D.select(behaviors, fitness_scores, features)
            else:
                # Default strategy: select the one with highest fitness
                max_idx = max(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
                selected = behaviors[max_idx]
            
            thought.selected_behavior = selected.name
            thought.decision_time = time.time() - start
            
            # Calculate confidence if available
            if self.D and hasattr(self.D, 'get_confidence'):
                confidence = self.D.get_confidence(fitness_scores)
                self.logger.debug(f"  Decision confidence: {confidence:.3f}")
            
            self.logger.debug(f"  Selected behavior: {selected.name}")
            
            return selected
            
        except Exception as e:
            error_msg = f"Decision failed: {e}"
            self.logger.error(error_msg)
            raise DecisionError(error_msg) from e
    
    def _execute(self, 
                 behavior: MetaBehavior, 
                 features: FeatureVector,
                 context: Dict,
                 thought: Thought) -> Any:
        """
        Execute stage: Run the selected behavior.
        
        Calls the chosen behavior to perform the actual task.
        
        Args:
            behavior: Behavior to execute
            features: Feature vector
            context: Context dictionary
            thought: Thought record
            
        Returns:
            Any: Execution result
            
        Raises:
            BehaviorExecutionError: When execution fails
        """
        thought.stage = ThinkingStage.EXECUTE
        
        self.logger.debug(f"⚡ Execute stage: behavior={behavior.name}")
        
        try:
            # Pre-execution hook
            if hasattr(behavior, 'before_execute'):
                behavior.before_execute(features)
            
            # Execute behavior
            result = behavior.execute(features, **context)
            
            # Post-execution hook
            if hasattr(behavior, 'after_execute'):
                behavior.after_execute(result, thought.execution_time)
            
            self.logger.debug(f"  Execution successful: result type={type(result).__name__}")
            
            return result
            
        except Exception as e:
            # Error hook
            if hasattr(behavior, 'on_error'):
                behavior.on_error(e)
            
            error_msg = f"Behavior {behavior.name} execution failed: {e}"
            self.logger.error(error_msg)
            raise BehaviorExecutionError(error_msg) from e
    
    def _learn(self, 
               behavior: MetaBehavior, 
               features: FeatureVector,
               expected: Any, 
               actual: Any,
               thought: Thought) -> None:
        """
        Learn stage: Update parameters based on feedback.
        
        Compares expected and actual results to adjust selector parameters.
        
        Args:
            behavior: Executed behavior
            features: Feature vector
            expected: Expected result
            actual: Actual result
            thought: Thought record
        """
        thought.stage = ThinkingStage.LEARN
        self.stats['learn_attempts'] += 1
        
        self.logger.debug(f"📚 Learn stage: behavior={behavior.name}")
        
        try:
            # Check if learning is possible
            if not self.Ψ or not self.S:
                self.logger.debug("  Skipping learning: missing learning mechanism or selector")
                return
            
            if hasattr(self.Ψ, 'should_learn') and not self.Ψ.should_learn(expected, actual):
                self.logger.debug("  Skipping learning: should_learn returned False")
                return
            
            # Prepare context
            context = {
                'features': features,
                'behavior_name': behavior.name,
                'parameters': self.S.get_parameters(),
                'learning_rate': self.config.learning_rate,
                'timestamp': time.time(),
                'expected': expected,
                'actual': actual,
            }
            
            # Compute adjustment
            adjustment = self.Ψ.compute_adjustment(expected, actual, context)
            
            # Apply adjustment
            if adjustment:
                # Apply weight decay
                if self.config.weight_decay > 0:
                    current_params = self.S.get_parameters()
                    for key, value in current_params.items():
                        if key in adjustment:
                            adjustment[key] -= self.config.weight_decay * value
                
                # Update parameters
                self.S.update_parameters(adjustment)
                
                thought.learning_status = LearningStatus.SUCCESS
                thought.parameter_changes = adjustment
                self.stats['learn_success'] += 1
                
                self.logger.debug(f"  Learning successful: adjusted {len(adjustment)} parameters")
            else:
                self.logger.debug("  No parameter adjustment needed")
                
        except Exception as e:
            thought.learning_status = LearningStatus.FAILED
            self.logger.warning(f"  Learning failed: {e}")
            if self.config.debug_mode:
                self.logger.debug(traceback.format_exc())
    
    def _update_stats(self, thought: Thought, selected: MetaBehavior, success: bool) -> None:
        """
        Update statistics based on thought record.
        
        Args:
            thought: Thought record
            selected: Selected behavior
            success: Whether thinking was successful
        """
        self.stats['total_thoughts'] += 1
        self.stats['successful_thoughts'] += 1 if success else 0
        self.stats['total_time'] += thought.total_time
        
        # Update average decision time
        total = self.stats['avg_decision_time'] * (self.stats['total_thoughts'] - 1)
        total += thought.decision_time
        self.stats['avg_decision_time'] = total / self.stats['total_thoughts']
        
        # Update average execution time
        total = self.stats['avg_execution_time'] * (self.stats['total_thoughts'] - 1)
        total += thought.execution_time
        self.stats['avg_execution_time'] = total / self.stats['total_thoughts']
        
        # Update behavior statistics
        if selected:
            self.stats['behaviors_used'][selected.name] = (
                self.stats['behaviors_used'].get(selected.name, 0) + 1
            )
            if success:
                self.stats['behaviors_success'][selected.name] = (
                    self.stats['behaviors_success'].get(selected.name, 0) + 1
                )
            else:
                self.stats['behaviors_failure'][selected.name] = (
                    self.stats['behaviors_failure'].get(selected.name, 0) + 1
                )
        
        self.stats['last_thought_time'] = thought.timestamp.isoformat()
    
    def _add_to_history(self, thought: Thought) -> None:
        """
        Add thought record to history.
        
        Args:
            thought: Thought record to add
        """
        self._history.append(thought)
        
        # Maintain history size limit
        if self.config.max_history_size is not None and len(self._history) > self.config.max_history_size:
            self._history = self._history[-self.config.max_history_size:]
    
    def _auto_save(self) -> None:
        """Automatically save agent state for checkpointing."""
        try:
            save_path = f"metathin_{self.name}_autosave.pkl"
            self.save(save_path)
            self.logger.info(f"💾 Auto-save completed: {save_path}")
        except Exception as e:
            self.logger.error(f"Auto-save failed: {e}")
    
    # ========================================================
    # Memory Functions
    # ========================================================
    
    def remember(self, key: str, value: Any, permanent: bool = True) -> bool:
        """
        Remember information.
        
        Stores information in memory, optionally persistent.
        
        Design considerations:
            - Check if self.memory exists to avoid null pointer
            - Detailed logging for debugging
            - Comprehensive error handling
        
        Args:
            key: Memory key
            value: Memory value
            permanent: Whether to save permanently (True: store in backend, False: cache only)
            
        Returns:
            bool: Whether operation was successful
        """
        # Check if memory manager exists
        if not hasattr(self, 'memory') or self.memory is None:
            self.logger.warning(f"💭 Memory not enabled, cannot remember '{key}'")
            return False
        
        try:
            # Call memory manager's remember method
            success = self.memory.remember(key, value, permanent)
            
            if success:
                self.logger.debug(f"💭 Remembered successfully: '{key}' (permanent={permanent})")
            else:
                self.logger.warning(f"💭 Failed to remember: '{key}'")
            
            return success
            
        except Exception as e:
            self.logger.error(f"💭 Exception during remember: {e}")
            return False
    
    def recall(self, key: str, default: Any = None) -> Any:
        """
        Recall information.
        
        Retrieves previously stored information from memory.
        
        Design considerations:
            - Check if self.memory exists
            - Provide default value to avoid exceptions
            - Detailed logging
        
        Args:
            key: Memory key
            default: Default value (returned if key doesn't exist)
            
        Returns:
            Any: Stored value, or default if not found
        """
        # Check if memory manager exists
        if not hasattr(self, 'memory') or self.memory is None:
            self.logger.debug(f"💭 Memory not enabled, returning default value")
            return default
        
        try:
            value = self.memory.recall(key)
            
            if value is not None:
                self.logger.debug(f"💭 Recalled successfully: '{key}'")
                return value
            else:
                self.logger.debug(f"💭 Key not found: '{key}', returning default")
                return default
                
        except Exception as e:
            self.logger.error(f"💭 Exception during recall: {e}")
            return default
    
    def forget(self, key: str, permanent: bool = True) -> bool:
        """
        Forget information.
        
        Deletes specified information from memory.
        
        Args:
            key: Memory key
            permanent: Whether to also delete from backend
            
        Returns:
            bool: Whether operation was successful
        """
        if not hasattr(self, 'memory') or self.memory is None:
            self.logger.warning(f"💭 Memory not enabled, cannot forget '{key}'")
            return False
        
        try:
            success = self.memory.forget(key, permanent)
            
            if success:
                self.logger.debug(f"💭 Forgotten successfully: '{key}'")
            else:
                self.logger.warning(f"💭 Failed to forget: '{key}'")
            
            return success
            
        except Exception as e:
            self.logger.error(f"💭 Exception during forget: {e}")
            return False
    
    def clear_memory(self, permanent: bool = True) -> bool:
        """
        Clear all memory.
        
        Args:
            permanent: Whether to also clear backend
            
        Returns:
            bool: Whether operation was successful
        """
        if not hasattr(self, 'memory') or self.memory is None:
            self.logger.warning("💭 Memory not enabled, cannot clear")
            return False
        
        try:
            success = self.memory.clear(permanent)
            
            if success:
                self.logger.info("🧹 All memory cleared successfully")
            else:
                self.logger.warning("🧹 Failed to clear memory")
            
            return success
            
        except Exception as e:
            self.logger.error(f"🧹 Exception during memory clear: {e}")
            return False
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics.
        
        Returns:
            Dict: Memory statistics including cache hit rate, storage size, etc.
        """
        if not hasattr(self, 'memory') or self.memory is None:
            return {'enabled': False, 'reason': 'Memory not enabled'}
        
        try:
            stats = self.memory.get_stats()
            stats['enabled'] = True
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get memory stats: {e}")
            return {'enabled': True, 'error': str(e)}
    
    # ========================================================
    # History Management
    # ========================================================
    
    def get_history(self, limit: Optional[int] = None, success_only: bool = False) -> List[Thought]:
        """
        Get thought history.
        
        Args:
            limit: Limit the number of returned records
            success_only: Only return successful thoughts
            
        Returns:
            List[Thought]: Thought history records
        """
        history = self._history
        
        if success_only:
            history = [t for t in history if t.success]
        
        if limit is not None:
            history = history[-limit:]
        
        return history.copy()
    
    def get_last_thought(self) -> Optional[Thought]:
        """
        Get the last thought record.
        
        Returns:
            Optional[Thought]: Last thought record, or None if no history
        """
        return self._history[-1] if self._history else None
    
    def clear_history(self) -> None:
        """Clear thought history."""
        self._history.clear()
        self.logger.info("🧹 Thought history cleared")
    
    def export_history(self, filepath: str, format: str = 'json') -> bool:
        """
        Export thought history to file.
        
        Args:
            filepath: Export file path
            format: Export format ('json' or 'pickle')
            
        Returns:
            bool: Whether export was successful
        """
        try:
            history_data = [t.to_dict() for t in self._history]
            
            if format == 'json':
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(history_data, f, indent=2, ensure_ascii=False)
            elif format == 'pickle':
                with open(filepath, 'wb') as f:
                    pickle.dump(history_data, f)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"📤 Exported history to: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export history: {e}")
            return False
    
    # ========================================================
    # State Management
    # ========================================================
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get agent status.
        
        Returns:
            Dict: Status dictionary containing name, components, statistics, etc.
        """
        # Ensure stats contains required fields
        if not hasattr(self, 'stats'):
            self.stats = self._init_stats()
        
        required_stats = ['total_thoughts', 'successful_thoughts', 'failed_thoughts']
        for stat in required_stats:
            if stat not in self.stats:
                self.stats[stat] = 0
        
        # Ensure total_thoughts is integer
        self.stats['total_thoughts'] = int(self.stats.get('total_thoughts', 0))
        
        return {
            'name': self.name,
            'version': '0.1.0',
            'components': {
                'pattern_space': type(self.P).__name__,
                'selector': type(self.S).__name__ if self.S else None,
                'decision_strategy': type(self.D).__name__ if self.D else None,
                'learning_mechanism': type(self.Ψ).__name__ if self.Ψ else None,
            },
            'behaviors_count': len(self.B),
            'behaviors': self.list_behaviors(),
            'history_length': len(self._history),
            'stats': self.stats.copy(),
            'is_thinking': self._is_thinking,
            'thinking_count': self._thinking_count,
            'error_count': self._error_count,
            'memory_enabled': hasattr(self, 'memory') and self.memory is not None,
            'last_thought_time': self._last_thought_time.isoformat() if self._last_thought_time else None,
            'config': {
                'learning_rate': self.config.learning_rate,
                'enable_learning': self.config.enable_learning,
                'min_fitness_threshold': self.config.min_fitness_threshold,
                'keep_history': self.config.keep_history,
                'enable_memory': self.config.enable_memory,
            }
        }
    
    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self.stats = self._init_stats()
        self._error_count = 0
        self.logger.info("🔄 Statistics reset")
    
    def reset(self) -> None:
        """Completely reset the agent (clear history, reset statistics)."""
        self.clear_history()
        self.reset_stats()
        self._thinking_count = 0
        self._error_count = 0
        self._last_thought_time = None
        self.logger.info("🔄 Agent fully reset")
    
    # ========================================================
    # Serialization
    # ========================================================
    
    def save(self, path: Union[str, Path]) -> bool:
        """
        Save agent state to file.
        
        Args:
            path: Save path
            
        Returns:
            bool: Whether save was successful
        """
        try:
            path = Path(path)
            # Ensure directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare data for saving - only serializable content
            data = {
                'name': self.name,
                'config': self.config,
                'stats': self.stats,
                'history': [t.to_dict() for t in self._history],
                'thinking_count': self._thinking_count,
                'error_count': self._error_count,
                'timestamp': datetime.now().isoformat(),
                'behaviors': []  # Store behavior names rather than objects
            }
            
            # Record behavior names
            for behavior in self.B:
                data['behaviors'].append({
                    'name': behavior.name,
                    'type': type(behavior).__name__
                })
            
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            
            self.logger.info(f"💾 Saved to: {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Save failed: {e}")
            return False
    
    @classmethod
    def load(cls,
             path: Union[str, Path],
             pattern_space: Optional[PatternSpace] = None,
             selector: Optional[Selector] = None,
             decision_strategy: Optional[DecisionStrategy] = None,
             learning_mechanism: Optional[LearningMechanism] = None,
             behaviors: Optional[List[MetaBehavior]] = None) -> 'Metathin':
        """
        Load agent state from file.
        
        Args:
            path: File path
            pattern_space: Pattern space (must be provided)
            selector: Selector (optional)
            decision_strategy: Decision strategy (optional)
            learning_mechanism: Learning mechanism (optional)
            behaviors: Behavior list (must be provided to restore behaviors)
            
        Returns:
            Metathin: Loaded agent instance
            
        Raises:
            FileNotFoundError: When file doesn't exist
            ValueError: When file format is invalid or required parameters missing
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        if pattern_space is None:
            raise ValueError("pattern_space parameter must be provided")
        
        if behaviors is None:
            raise ValueError("behaviors parameter must be provided")
        
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            # Create new instance
            instance = cls(
                pattern_space=pattern_space,
                selector=selector,
                decision_strategy=decision_strategy,
                learning_mechanism=learning_mechanism,
                config=data.get('config'),
                name=data.get('name')
            )
            
            # Restore behaviors
            for behavior in behaviors:
                instance.register_behavior(behavior)
            
            # Restore state
            instance.stats = data.get('stats', instance._init_stats())
            instance._thinking_count = data.get('thinking_count', 0)
            instance._error_count = data.get('error_count', 0)
            
            # Restore history
            history_data = data.get('history', [])
            instance._history = [Thought.from_dict(t) for t in history_data]
            
            instance.logger.info(f"📂 Loaded successfully from: {path}")
            return instance
            
        except Exception as e:
            raise ValueError(f"Load failed: {e}")
    
    # ========================================================
    # Special Methods
    # ========================================================
    
    def __repr__(self) -> str:
        """String representation for developers."""
        memory_status = "✓" if (hasattr(self, 'memory') and self.memory) else "✗"
        return (f"Metathin(name='{self.name}', "
                f"behaviors={len(self.B)}, "
                f"thoughts={self.stats['total_thoughts']}, "
                f"memory={memory_status}, "
                f"success_rate={self.stats['successful_thoughts']/max(1,self.stats['total_thoughts']):.2%})")
    
    def __str__(self) -> str:
        """User-friendly string representation."""
        return self.__repr__()
    
    def __call__(self, raw_input: Any, **kwargs) -> Any:
        """Make instance callable, directly invoking think method."""
        return self.think(raw_input, **kwargs)
    
    def __len__(self) -> int:
        """Return number of behaviors."""
        return len(self.B)
    
    def __contains__(self, behavior_name: str) -> bool:
        """Check if behavior exists."""
        return any(b.name == behavior_name for b in self.B)
    
    def __getitem__(self, behavior_name: str) -> Optional[MetaBehavior]:
        """Get behavior by name using indexing syntax."""
        return self.get_behavior(behavior_name)


# ============================================================
# Export
# ============================================================

__all__ = [
    'Metathin',
    'Thought',
    'ThinkingStage',
    'LearningStatus',
    'MetathinConfig',
]