"""
Metathin Core Interfaces Definition
===================================

Defines the standard interfaces for the quintuple (P, B, S, D, Ψ).
All custom components must implement these interfaces.

Design Philosophy:
    - Fixed Interfaces: The core framework defines clear abstract base classes
    - Free Implementation: Users can implement their own algorithms arbitrarily
    - Type Safety: Type annotations improve code reliability

Quintuple Description:
    P (PatternSpace): Perception - Converts raw input into feature vectors
    B (MetaBehavior): Action - Executable skill units
    S (Selector): Evaluation - Computes fitness for each behavior
    D (DecisionStrategy): Decision - Selects behavior based on fitness
    Ψ (LearningMechanism): Learning - Adjusts parameters based on feedback
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, Dict, Any, Optional, Tuple, Union
import numpy as np
import logging
from datetime import datetime


# ============================================================
# Type Variable Definitions
# ============================================================
# Generic type variables to support different input and output types
# ============================================================

T = TypeVar('T')  # Input type (raw data) - can be any type
R = TypeVar('R')  # Result type (behavior output) - can be any type

FeatureVector = np.ndarray
"""
Feature vector type: q = (q1, q2, ..., qs)

This is the output format of the pattern space and serves as the fundamental 
input to all subsequent components. Feature vectors must be:
    - numpy arrays
    - one-dimensional (1D)
    - float64 type
    - Free of NaN or Inf values

Each element represents a feature dimension of the input, for example:
    - Text length
    - Image pixel mean
    - Sensor readings
    - Statistical features
"""

FitnessScore = float
"""
Fitness score: α(Bi) ∈ [0, 1]

Represents how suitable a behavior is in the current context:
    - 0: Completely unsuitable (should not be selected)
    - 0.5: Moderately suitable
    - 1: Perfectly suitable (should be selected)

Fitness scores form the foundation of all decision-making. The selector
computes these scores, and decision strategies use them to make choices.
"""

ParameterDict = Dict[str, float]
"""
Parameter dictionary: Used for selector parameter adjustment

This serves as the interface between the learning mechanism and the selector:
    - Key: Parameter name (e.g., 'w_0_1' represents weight for behavior 0, feature 1)
    - Value: Parameter value (must be float type)

The selector exposes learnable parameters via get_parameters(),
the learning mechanism computes adjustments via compute_adjustment(),
and the selector updates parameters via update_parameters().
"""


# ============================================================
# Exception Definitions
# ============================================================
# Unified exception hierarchy for consistent error handling and debugging
# ============================================================

class MetathinError(Exception):
    """
    Metathin base exception class. All custom exceptions should inherit from this.
    
    This allows users to catch all framework-specific exceptions with a single
    except clause while still having access to specific exception types when needed.
    """
    pass


class PatternExtractionError(MetathinError):
    """
    Pattern extraction error.
    
    Raised when the pattern space cannot extract features from input.
    Possible causes:
        - Incorrect input format
        - Input contains invalid values
        - Feature extraction algorithm failed
    """
    pass


class BehaviorExecutionError(MetathinError):
    """
    Behavior execution error.
    
    Raised when an exception occurs during behavior execution.
    Possible causes:
        - Logic error in behavior implementation
        - Resources unavailable
        - Timeout
        - External dependency failure
    """
    pass


class FitnessComputationError(MetathinError):
    """
    Fitness computation error.
    
    Raised when the selector cannot compute fitness scores.
    Possible causes:
        - Feature vector dimension mismatch
        - Numerical errors during computation
        - Behavior index out of bounds
    """
    pass


class DecisionError(MetathinError):
    """
    Decision error.
    
    Raised when the decision strategy cannot make a selection.
    Possible causes:
        - Empty behavior list
        - Invalid fitness scores
        - Strategy internal error
    """
    pass


class NoBehaviorError(DecisionError):
    """
    No available behavior error.
    
    Raised when no behaviors are suitable for the current context.
    This typically occurs when:
        - No behaviors are registered
        - All behaviors' can_execute() returns False
        - All behaviors' fitness scores are below threshold
    """
    pass


class LearningError(MetathinError):
    """
    Learning error.
    
    Raised when the learning mechanism fails to update parameters.
    Possible causes:
        - Incorrect parameter format
        - Improper learning rate settings
        - Numerical instability (gradient explosion/vanishing)
    """
    pass


class ParameterUpdateError(LearningError):
    """
    Parameter update error.
    
    Raised when updating selector parameters fails.
    This is a specialization of LearningError for the parameter update phase.
    Possible causes:
        - Parameter key doesn't exist
        - Parameter value type mismatch
        - Parameter value out of valid range
    """
    pass


# ============================================================
# Core Interface Definitions
# ============================================================
# Abstract base classes for the quintuple. All custom components must inherit
# from these interfaces to ensure compatibility with the framework.
# ============================================================

class PatternSpace(ABC, Generic[T]):
    """
    Pattern Space P: Window to perceive the environment.
    
    This serves as the agent's "eyes," responsible for converting raw input into
    structured feature vectors. Feature vectors should capture key information
    from the input, providing a foundation for subsequent decision-making.
    
    Design Considerations:
        - Generic parameter T represents the raw input type, which can be any type
        - extract() is the only method that must be implemented
        - Feature vectors must be 1D float64 arrays
        - Different inputs should return feature vectors of the same dimension
        - Features should ideally be normalized to a reasonable range (e.g., [-1,1] or [0,1])
    
    Implementation Examples:
        - Text processing: Extract length, word frequency, sentiment score
        - Image processing: Extract color histograms, edge features
        - Sensor data: Extract statistical features, frequency domain features
        - Time series: Extract trends, periodicity, chaotic features
    
    Example:
        >>> class TextPattern(PatternSpace[str]):
        ...     '''Text pattern space: extracts text length and word count'''
        ...     def extract(self, text: str) -> FeatureVector:
        ...         # Extract two features: character count and word count
        ...         char_count = len(text)
        ...         word_count = len(text.split())
        ...         return np.array([char_count, word_count], dtype=np.float64)
        ...     
        ...     def get_feature_names(self) -> List[str]:
        ...         return ['char_count', 'word_count']
    """
    
    @abstractmethod
    def extract(self, raw_input: T) -> FeatureVector:
        """
        Extract feature vector from raw input.
        
        This is the core method of the pattern space and must be implemented by subclasses.
        Feature extraction can be as simple as statistical calculations or as complex
        as deep learning models.
        
        Args:
            raw_input: Raw input data, type determined by generic parameter T
            
        Returns:
            FeatureVector: Feature vector, must be a numpy array with float64 dtype
            
        Raises:
            PatternExtractionError: When feature extraction fails
            
        Notes:
            - The returned array should be one-dimensional
            - Different inputs should return feature vectors of the same dimension
            - Feature values should ideally be normalized to a reasonable range
            - Avoid returning NaN or Inf values
        """
        pass
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature name list (optional implementation).
        
        Used for debugging, logging, and visualization. If not implemented,
        the framework will use default names.
        Feature names should correspond one-to-one with feature vector dimensions.
        
        Returns:
            List[str]: Feature names list, length should match feature vector dimension
        """
        return []
    
    def get_feature_dimension(self) -> int:
        """
        Get feature dimension (optional implementation).
        
        Default implementation infers dimension by extracting features from a sample.
        If extraction is expensive, subclasses can override this method for efficiency.
        
        Returns:
            int: Feature vector dimension
        """
        try:
            # Use a simple sample to test dimension
            sample = self.extract("sample" if hasattr(self, '__orig_class__') else None)
            return len(sample)
        except:
            return 0
    
    def validate_features(self, features: FeatureVector) -> bool:
        """
        Validate feature vector (optional implementation).
        
        Checks whether the feature vector meets requirements:
            - Is a numpy array
            - Has float64 dtype
            - Is one-dimensional
            - Contains no NaN values
            - Contains no Inf values
        
        Args:
            features: Feature vector to validate
            
        Returns:
            bool: Whether the feature vector is valid
        """
        return (isinstance(features, np.ndarray) and 
                features.dtype == np.float64 and
                features.ndim == 1 and
                not np.any(np.isnan(features)) and
                not np.any(np.isinf(features)))


class MetaBehavior(ABC, Generic[T, R]):
    """
    Meta-Behavior B: System's skill collection.
    
    Each behavior focuses on solving a well-defined subproblem. Behaviors can
    maintain internal state but should be thread-safe.
    
    Design Considerations:
        - The name property must be unique for identification and selection
        - execute() is the core method that implements specific business logic
        - Can maintain internal state (e.g., execution count, cache)
        - Provides hook functions (before/after/on_error) for extensibility
        - Complexity value helps evaluate resource consumption
    
    Implementation Examples:
        - Computation behavior: Performs mathematical calculations
        - Query behavior: Retrieves information from databases
        - Control behavior: Sends commands to external devices
        - Composite behavior: Calls other behaviors
    
    Example:
        >>> class GreetBehavior(MetaBehavior[FeatureVector, str]):
        ...     '''Greeting behavior: generates greeting based on feature vector'''
        ...     
        ...     def __init__(self, name: str = "greet"):
        ...         super().__init__()
        ...         self._name = name
        ...         self.greeted_count = 0  # Can maintain state
        ...     
        ...     @property
        ...     def name(self) -> str:
        ...         return self._name
        ...     
        ...     def execute(self, features: FeatureVector, **kwargs) -> str:
        ...         self.greeted_count += 1
        ...         length = features[0] if len(features) > 0 else 0
        ...         return f"Hello! ({self.greeted_count}th greeting, input length: {length})"
        ...     
        ...     def can_execute(self, features: FeatureVector) -> bool:
        ...         # Can only execute when all feature values are positive
        ...         return np.all(features > 0)
    """
    
    def __init__(self):
        """Initialize behavior base class with basic statistical attributes."""
        self._execution_count = 0          # Number of times executed
        self._last_execution_time = 0.0    # Last execution duration (seconds)
        self._total_execution_time = 0.0   # Total execution duration (seconds)
        self._last_error: Optional[Exception] = None  # Last error encountered
        self._created_at = datetime.now()   # Creation timestamp
        self._logger = logging.getLogger(f"metathin.behavior.{self.__class__.__name__}")
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Behavior name, used for identification and logging.
        
        Must be unique. Meaningful names are recommended.
        Naming conventions:
            - Use camelCase
            - Reflect the behavior's function
            - Avoid special characters
        """
        pass
    
    @abstractmethod
    def execute(self, features: FeatureVector, **kwargs) -> R:
        """
        Execute the behavior and return the result.
        
        This is the core method of the behavior, implementing specific business logic.
        
        Args:
            features: Current feature vector
            **kwargs: Additional context parameters, such as:
                - context: Global context information
                - timeout: Timeout setting
                - callback: Callback function
                
        Returns:
            R: Behavior execution result, type determined by generic parameter R
            
        Raises:
            BehaviorExecutionError: When execution fails
        """
        pass
    
    def can_execute(self, features: FeatureVector) -> bool:
        """
        Determine if execution is possible with the current features (optional override).
        
        Used to filter out unsuitable cases in advance, improving system efficiency.
        For example:
            - Check if feature values are within valid range
            - Check if required resources are available
            - Check if state meets preconditions
        
        Args:
            features: Current feature vector
            
        Returns:
            bool: Whether execution is possible, default returns True
        """
        return True
    
    def get_complexity(self) -> float:
        """
        Get behavior complexity (optional override).
        
        Used to evaluate resource consumption, influencing scheduling decisions.
        Complexity can be based on:
            - Algorithm time complexity
            - Expected execution time
            - Resource usage
            - Historical execution data
        
        Returns:
            float: Complexity value, default returns 1.0
        """
        return 1.0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get behavior statistics.
        
        Returns runtime statistics for monitoring and debugging.
        
        Returns:
            Dict: Statistics including execution count, timing, etc.
        """
        return {
            'name': self.name,
            'execution_count': self._execution_count,
            'last_execution_time': self._last_execution_time,
            'total_execution_time': self._total_execution_time,
            'avg_execution_time': (self._total_execution_time / self._execution_count 
                                   if self._execution_count > 0 else 0),
            'created_at': self._created_at.isoformat(),
            'last_error': str(self._last_error) if self._last_error else None,
        }
    
    def before_execute(self, features: FeatureVector) -> None:
        """
        Pre-execution hook function (optional override).
        
        Can be used for resource preparation, logging, etc.
        Called before execute().
        """
        self._logger.debug(f"Preparing to execute behavior {self.name}")
    
    def after_execute(self, result: R, execution_time: float) -> None:
        """
        Post-execution hook function (optional override).
        
        Can be used for resource cleanup, result logging, etc.
        Called after execute().
        
        Args:
            result: Execution result
            execution_time: Execution duration
        """
        self._logger.debug(f"Behavior {self.name} completed, duration {execution_time:.3f}s")
    
    def on_error(self, error: Exception) -> None:
        """
        Error handling callback (optional override).
        
        Can be used for error recovery, alerting, etc.
        Called when execute() raises an exception.
        
        Args:
            error: Caught exception
        """
        self._last_error = error
        self._logger.error(f"Behavior {self.name} execution failed: {error}")
    
    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._execution_count = 0
        self._last_execution_time = 0.0
        self._total_execution_time = 0.0
        self._last_error = None


class Selector(ABC):
    """
    Selection Mapping S: Evaluates behavior suitability.
    
    Computes fitness score α(Bi) = fi(q) for each behavior.
    The selector serves as the agent's "evaluator," can maintain internal parameters,
    and can be adjusted through the learning mechanism.
    
    Design Considerations:
        - Fitness scores must be in the [0,1] range
        - Can maintain learnable parameters (weights, biases, etc.)
        - Records fitness history for analysis and debugging
        - Supports parameter updates through learning mechanism
    
    Implementation Examples:
        - Linear selector: α = sigmoid(w·x + b)
        - Neural network selector: Uses deep learning models
        - Rule-based selector: Based on predefined rules
        - Ensemble selector: Combines multiple selectors
    
    Example:
        >>> class LinearSelector(Selector):
        ...     '''Linear selector: computes fitness using linear combination of features'''
        ...     
        ...     def __init__(self, n_features: int, n_behaviors: int):
        ...         super().__init__()
        ...         # Initialize weight matrix [n_behaviors, n_features]
        ...         self.weights = np.random.randn(n_behaviors, n_features) * 0.1
        ...         self.bias = np.zeros(n_behaviors)
        ...         self._behavior_indices = {}
        ...     
        ...     def compute_fitness(self, behavior: MetaBehavior, features: FeatureVector) -> float:
        ...         # Get or assign behavior index
        ...         if behavior.name not in self._behavior_indices:
        ...             self._behavior_indices[behavior.name] = len(self._behavior_indices)
        ...         idx = self._behavior_indices[behavior.name]
        ...         
        ...         # Compute linear combination
        ...         z = np.dot(self.weights[idx], features) + self.bias[idx]
        ...         # Map to [0,1] using sigmoid function
        ...         return 1.0 / (1.0 + np.exp(-z))
        ...     
        ...     def get_parameters(self) -> ParameterDict:
        ...         params = {}
        ...         for i in range(len(self.weights)):
        ...             for j in range(self.weights.shape[1]):
        ...                 params[f'w_{i}_{j}'] = float(self.weights[i, j])
        ...             params[f'b_{i}'] = float(self.bias[i])
        ...         return params
    """
    
    def __init__(self):
        """Initialize selector base class."""
        self._logger = logging.getLogger(f"metathin.selector.{self.__class__.__name__}")
        self._fitness_history: Dict[str, List[float]] = {}  # Fitness history per behavior
    
    @abstractmethod
    def compute_fitness(self, 
                       behavior: MetaBehavior, 
                       features: FeatureVector) -> FitnessScore:
        """
        Compute fitness score for a behavior given the current features.
        
        This is the core method of the selector and must be implemented.
        
        Args:
            behavior: Behavior to evaluate
            features: Current feature vector
            
        Returns:
            FitnessScore: Fitness score in the range [0,1]
            
        Raises:
            FitnessComputationError: When computation fails
        """
        pass
    
    def get_parameters(self) -> ParameterDict:
        """
        Get current learnable parameters (optional implementation).
        
        These parameters will be adjusted by the learning mechanism.
        Parameter format must be compatible with update_parameters().
        
        Returns:
            ParameterDict: Parameter dictionary, keys are parameter names, values are parameter values
        """
        return {}
    
    def update_parameters(self, delta: ParameterDict) -> None:
        """
        Update parameters based on adjustments from learning mechanism (optional implementation).
        
        Args:
            delta: Parameter adjustment dictionary, keys must match get_parameters()
            
        Raises:
            ParameterUpdateError: When update fails
        """
        pass
    
    def record_fitness(self, behavior_name: str, fitness: float) -> None:
        """
        Record fitness history for analysis and debugging.
        
        Args:
            behavior_name: Behavior name
            fitness: Fitness value
        """
        if behavior_name not in self._fitness_history:
            self._fitness_history[behavior_name] = []
        self._fitness_history[behavior_name].append(fitness)
        # Limit history length
        if len(self._fitness_history[behavior_name]) > 1000:
            self._fitness_history[behavior_name] = self._fitness_history[behavior_name][-1000:]
    
    def get_fitness_history(self, behavior_name: Optional[str] = None) -> Dict[str, List[float]]:
        """
        Get fitness history.
        
        Args:
            behavior_name: Specific behavior name, returns all if None
            
        Returns:
            Dict: Mapping from behavior name to fitness history
        """
        if behavior_name:
            return {behavior_name: self._fitness_history.get(behavior_name, [])}
        return self._fitness_history.copy()
    
    def reset_history(self) -> None:
        """Reset fitness history."""
        self._fitness_history.clear()


class DecisionStrategy(ABC):
    """
    Decision Space D: Selects optimal behavior based on fitness.
    
    B* = σ(α), where σ is a decision strategy based on all α(Bi) values.
    Different strategies embody different exploration-exploitation trade-offs.
    
    Design Considerations:
        - Input is a list of fitness scores, output is the selected behavior
        - Can embody different decision preferences (greedy, exploratory, round-robin, etc.)
        - Can provide confidence assessment
        - Can maintain internal state (e.g., selection history)
    
    Implementation Examples:
        - Greedy strategy: Always selects the highest fitness
        - ε-greedy: Explores randomly with probability ε
        - Boltzmann strategy: Selects according to probability distribution
        - Round-robin strategy: Cycles through behaviors
        - Hybrid strategy: Switches strategies based on conditions
    
    Example:
        >>> class EpsilonGreedyStrategy(DecisionStrategy):
        ...     '''ε-greedy strategy: explores with probability ε, exploits with probability 1-ε'''
        ...     
        ...     def __init__(self, epsilon: float = 0.1, decay: float = 0.99):
        ...         self.epsilon = epsilon
        ...         self.decay = decay
        ...         self.step = 0
        ...     
        ...     def select(self, behaviors, fitness_scores, features):
        ...         self.step += 1
        ...         
        ...         # Exploration: random selection
        ...         if random.random() < self.epsilon:
        ...             idx = random.randrange(len(behaviors))
        ...         # Exploitation: select highest fitness
        ...         else:
        ...             idx = np.argmax(fitness_scores)
        ...         
        ...         # Decay exploration rate
        ...         self.epsilon *= self.decay
        ...         
        ...         return behaviors[idx]
    """
    
    @abstractmethod
    def select(self, 
               behaviors: List[MetaBehavior], 
               fitness_scores: List[float],
               features: FeatureVector) -> MetaBehavior:
        """
        Select the optimal behavior from the candidate list.
        
        Args:
            behaviors: List of candidate behaviors
            fitness_scores: Corresponding fitness scores
            features: Current feature vector
            
        Returns:
            MetaBehavior: Selected behavior
            
        Raises:
            NoBehaviorError: When no behaviors are available
            DecisionError: When other errors occur during decision
            
        Notes:
            - behaviors and fitness_scores must have the same length
            - Fitness scores should already be in the [0,1] range
        """
        pass
    
    def get_confidence(self, fitness_scores: List[float]) -> float:
        """
        Calculate decision confidence (optional implementation).
        
        Used to assess decision reliability.
        Higher confidence indicates greater certainty in the decision.
        
        Args:
            fitness_scores: Fitness scores of all candidate behaviors
            
        Returns:
            float: Confidence in the range [0,1], 1 indicates very certain
        """
        if len(fitness_scores) < 2:
            return 1.0
        # Default: use difference between highest and second-highest as confidence
        sorted_scores = sorted(fitness_scores, reverse=True)
        return sorted_scores[0] - sorted_scores[1]
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get strategy information (optional implementation).
        
        Returns:
            Dict: Contains strategy name, parameters, etc.
        """
        return {
            'name': self.__class__.__name__,
            'type': 'decision_strategy'
        }


class LearningMechanism(ABC):
    """
    Pattern Regulation Mapping Ψ: Learning capability.
    
    Adjusts selector parameters based on the discrepancy between execution results
    and expectations. Ψ: δ ⟼ γ̂, where δ = E - ε is the residual.
    
    Design Considerations:
        - Input is the difference between expected and actual results
        - Output is the adjustment amount for selector parameters
        - Can be based on various learning paradigms
        - Can maintain internal state (e.g., momentum, gradient history)
    
    Learning Paradigms:
        - Supervised learning: Uses difference between expected and actual
        - Reinforcement learning: Uses reward signals
        - Unsupervised learning: Discovers patterns in data
        - Meta-learning: Learns how to learn
    
    Implementation Examples:
        - Gradient descent: Based on error gradients
        - Reinforcement learning: Based on reward signals
        - Memory-based learning: Based on historical experience
        - Hebbian learning: Based on neuronal correlations
    
    Example:
        >>> class GradientDescentLearning(LearningMechanism):
        ...     '''Gradient descent learning: updates parameters using error gradients'''
        ...     
        ...     def __init__(self, learning_rate: float = 0.01):
        ...         self.lr = learning_rate
        ...     
        ...     def compute_adjustment(self, expected, actual, context):
        ...         # Get current parameters
        ...         params = context.get('parameters', {})
        ...         features = context.get('features', np.array([1.0]))
        ...         
        ...         # Calculate error
        ...         if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        ...             error = expected - actual
        ...         else:
        ...             error = 1.0 if expected == actual else -1.0
        ...         
        ...         # Calculate gradients (simplified version)
        ...         adjustments = {}
        ...         for key, value in params.items():
        ...             if key.startswith('w_'):
        ...                 # Use feature values as gradients
        ...                 feat_idx = int(key.split('_')[1])
        ...                 grad = -2 * error * features[feat_idx % len(features)]
        ...                 adjustments[key] = self.lr * grad
        ...         
        ...         return adjustments if adjustments else None
    """
    
    @abstractmethod
    def compute_adjustment(self, 
                          expected: Any, 
                          actual: Any,
                          context: Dict[str, Any]) -> Optional[ParameterDict]:
        """
        Compute parameter adjustment amount.
        
        This is the core method of the learning mechanism. Based on the discrepancy
        between expected and actual results, it calculates how much the selector
        parameters should be adjusted.
        
        Args:
            expected: Expected result
            actual: Actual result
            context: Context information, including:
                - features: Current feature vector
                - behavior_name: Name of executed behavior
                - parameters: Current selector parameters
                - learning_rate: Learning rate (optional)
                - timestamp: Timestamp
                - reward: Reward value (reinforcement learning scenarios)
                - metadata: Other metadata
                
        Returns:
            Optional[ParameterDict]: Parameter adjustment dictionary, None if no adjustment needed
            Adjustment keys should match those returned by get_parameters()
            
        Raises:
            LearningError: When learning process encounters an error
        """
        pass
    
    def should_learn(self, expected: Any, actual: Any) -> bool:
        """
        Determine whether learning should occur (optional implementation).
        
        Can be used for conditional learning, such as only learning when error is large.
        
        Args:
            expected: Expected result
            actual: Actual result
            
        Returns:
            bool: Whether learning should occur, default returns True
        """
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get learning mechanism statistics (optional implementation).
        
        Returns:
            Dict: Statistics including learning count, average error, etc.
        """
        return {
            'name': self.__class__.__name__,
            'type': 'learning_mechanism'
        }


# ============================================================
# Export all public interfaces
# ============================================================

__all__ = [
    # Type variables
    'T', 'R',
    'FeatureVector',
    'FitnessScore',
    'ParameterDict',
    
    # Exception classes
    'MetathinError',
    'PatternExtractionError',
    'BehaviorExecutionError',
    'FitnessComputationError',
    'DecisionError',
    'NoBehaviorError',
    'LearningError',
    'ParameterUpdateError',
    
    # Core interfaces
    'PatternSpace',
    'MetaBehavior',
    'Selector',
    'DecisionStrategy',
    'LearningMechanism',
]