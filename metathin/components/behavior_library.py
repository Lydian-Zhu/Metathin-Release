"""
Meta-Behavior Components
========================

Provides various executable skill units. All behaviors inherit from the MetaBehavior interface.
Behaviors are the agent's "action" modules, responsible for executing specific tasks.

Design Philosophy:
    - Composition over Inheritance: Create complex behaviors by wrapping existing ones
    - Separation of Concerns: Each behavior focuses on a single responsibility
    - Composability: Behaviors can be nested and combined to build complex workflows
    - Robustness: Comprehensive error handling and retry mechanisms

Behavior Categories:
    - Basic Behaviors: FunctionBehavior, LambdaBehavior
    - Composite Behaviors: CompositeBehavior
    - Control Behaviors: RetryBehavior, TimeoutBehavior, ConditionalBehavior
    - Optimization Behaviors: CachedBehavior

Example:
    >>> # Create basic behavior
    >>> greet = FunctionBehavior("greet", lambda f,**k: f"Hello, length={f[0]}")
    >>> 
    >>> # Add retry mechanism
    >>> robust_greet = RetryBehavior("robust_greet", greet, max_retries=3)
    >>> 
    >>> # Create composite behavior
    >>> pipeline = CompositeBehavior("pipeline", [step1, step2, step3])
    >>> 
    >>> # Conditional behavior
    >>> conditional = ConditionalBehavior("conditional", 
    ...     condition=lambda f: f[0] > 10,
    ...     true_behavior=large_handler,
    ...     false_behavior=small_handler)
"""

import time
import numpy as np
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from functools import wraps
import logging
import threading
import queue
from datetime import datetime

from ..core.interfaces import MetaBehavior, FeatureVector, BehaviorExecutionError


# ============================================================
# Decorators and Helper Functions
# ============================================================

def behavior_logger(func: Callable) -> Callable:
    """
    Behavior logging decorator.
    
    Automatically logs execution time, input, and output information.
    Used for debugging and performance analysis.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        logger = logging.getLogger(f"metathin.behavior.{self.name}")
        start = time.time()
        
        logger.debug(f"⚡ Starting execution: input={args[0] if args else 'None'}")
        
        try:
            result = func(self, *args, **kwargs)
            elapsed = time.time() - start
            logger.debug(f"✅ Execution completed: duration={elapsed:.3f}s, result={result}")
            return result
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"❌ Execution failed: duration={elapsed:.3f}s, error={e}")
            raise
            
    return wrapper


# ============================================================
# Function-Based Behavior
# ============================================================

class FunctionBehavior(MetaBehavior):
    """
    Function-based behavior: Implements behavior using regular Python functions.
    
    The simplest way to implement behaviors, suitable for wrapping existing functions.
    Functions should accept feature vectors and keyword arguments, returning any result.
    
    Characteristics:
        - Simple and intuitive: Directly uses Python functions
        - Flexible: Can encapsulate arbitrary logic
        - Documentable: Can include docstrings for description
    
    Example:
        >>> def process_text(features, **kwargs):
        ...     '''Process text features'''
        ...     length = features[0]
        ...     word_count = features[1] if len(features) > 1 else 0
        ...     return f"Text length: {length}, Word count: {word_count}"
        >>> 
        >>> behavior = FunctionBehavior(
        ...     name="text_processor",
        ...     func=process_text,
        ...     complexity=1.5,
        ...     description="Process text features and return description"
        ... )
        >>> result = behavior.execute(np.array([10, 2]))
        >>> print(result)  # Text length: 10, Word count: 2
    """
    
    def __init__(self, 
                 name: str, 
                 func: Callable,
                 complexity: float = 1.0,
                 description: Optional[str] = None):
        """
        Initialize function-based behavior.
        
        Args:
            name: Behavior name, must be unique
            func: Execution function, receives feature vector and keyword arguments
            complexity: Behavior complexity for resource evaluation
            description: Behavior description for documentation
        """
        super().__init__()
        
        if not name:
            raise ValueError("Behavior name cannot be empty")
        
        if not callable(func):
            raise TypeError(f"func must be callable, got {type(func)}")
        
        self._name = name
        self._func = func
        self._complexity = max(0.1, float(complexity))
        self._description = description or func.__doc__ or f"FunctionBehavior: {name}"
        
        self._logger = logging.getLogger(f"metathin.behavior.{name}")
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def description(self) -> str:
        return self._description
    
    @behavior_logger
    def execute(self, features: FeatureVector, **kwargs) -> Any:
        """
        Execute the function.
        
        Args:
            features: Feature vector
            **kwargs: Additional parameters
            
        Returns:
            Any: Function execution result
        """
        try:
            # Update statistics
            self._execution_count += 1
            start = time.time()
            
            # Pre-execution hook
            self.before_execute(features)
            
            # Execute function
            result = self._func(features, **kwargs)
            
            # Update execution time
            self._last_execution_time = time.time() - start
            self._total_execution_time += self._last_execution_time
            
            # Post-execution hook
            self.after_execute(result, self._last_execution_time)
            
            return result
            
        except Exception as e:
            self._last_execution_time = time.time() - start
            self._total_execution_time += self._last_execution_time
            self.on_error(e)
            raise BehaviorExecutionError(f"Behavior {self.name} execution failed: {e}") from e
    
    def get_complexity(self) -> float:
        return self._complexity


# ============================================================
# Lambda Behavior
# ============================================================

class LambdaBehavior(MetaBehavior):
    """
    Lambda behavior: Creates simple behaviors using lambda expressions.
    
    Suitable for quickly creating simple, one-off behaviors.
    Note: Lambda expressions cannot be serialized, so not suitable for scenarios requiring persistence.
    
    Characteristics:
        - Concise: Define behaviors in one line
        - Quick: Ideal for prototyping
        - Temporary: Not suitable for serialization
    """
    
    def __init__(self, 
                 name: str, 
                 func: Callable,
                 complexity: float = 0.5):
        """
        Initialize lambda behavior.
        
        Args:
            name: Behavior name
            func: Lambda function
            complexity: Behavior complexity
        """
        super().__init__()
        
        if not name:
            raise ValueError("Behavior name cannot be empty")
        
        if not callable(func):
            raise TypeError(f"func must be callable, got {type(func)}")
        
        self._name = name
        self._func = func
        self._complexity = max(0.1, float(complexity))
        self._logger = logging.getLogger(f"metathin.behavior.{name}")
    
    @property
    def name(self) -> str:
        return self._name
    
    @behavior_logger
    def execute(self, features: FeatureVector, **kwargs) -> Any:
        """Execute lambda."""
        try:
            self._execution_count += 1
            start = time.time()
            
            self.before_execute(features)
            result = self._func(features, **kwargs)
            
            self._last_execution_time = time.time() - start
            self._total_execution_time += self._last_execution_time
            
            self.after_execute(result, self._last_execution_time)
            
            return result
            
        except Exception as e:
            self._last_execution_time = time.time() - start
            self._total_execution_time += self._last_execution_time
            self.on_error(e)
            raise BehaviorExecutionError(f"Lambda behavior {self.name} execution failed: {e}") from e


# ============================================================
# Composite Behavior
# ============================================================

class CompositeBehavior(MetaBehavior):
    """
    Composite behavior: Executes multiple sub-behaviors in sequence.
    
    Combines multiple behaviors into a pipeline, where the output of one behavior
    becomes the input to the next. Supports data flow transformation and error handling.
    
    Characteristics:
        - Sequential execution: Strictly executes sub-behaviors in order
        - Data flow: Automatically handles data type conversions between steps
        - Error control: Option to stop or continue on error
        - Observable: Records execution results for each step
    """
    
    def __init__(self, 
                 name: str,
                 behaviors: List[MetaBehavior],
                 stop_on_error: bool = True,
                 description: Optional[str] = None):
        """
        Initialize composite behavior.
        
        Args:
            name: Behavior name
            behaviors: List of sub-behaviors to execute in sequence
            stop_on_error: Whether to stop execution when a step fails
            description: Behavior description
        """
        super().__init__()
        
        if not name:
            raise ValueError("Behavior name cannot be empty")
        
        if not behaviors:
            raise ValueError("Sub-behavior list cannot be empty")
        
        self._name = name
        self._behaviors = behaviors
        self._stop_on_error = stop_on_error
        self._description = description or f"CompositeBehavior with {len(behaviors)} steps"
        self._logger = logging.getLogger(f"metathin.behavior.{name}")
    
    @property
    def name(self) -> str:
        return self._name
    
    @behavior_logger
    def execute(self, features: FeatureVector, **kwargs) -> Any:
        """
        Execute all sub-behaviors in sequence.
        
        Execution flow:
            1. Execute each sub-behavior in order
            2. Transform previous result to next behavior's input
            3. Record execution results for each step
            4. Continue or stop based on stop_on_error setting
        """
        self._execution_count += 1
        current_input = features
        start = time.time()
        step_results = []
        
        try:
            self.before_execute(features)
            
            for i, behavior in enumerate(self._behaviors):
                step_start = time.time()
                
                self._logger.debug(f"  Step {i+1}/{len(self._behaviors)}: {behavior.name}")
                
                try:
                    # Execute sub-behavior
                    result = behavior.execute(current_input, **kwargs)
                    step_results.append({
                        'step': i,
                        'behavior': behavior.name,
                        'result': result,
                        'time': time.time() - step_start
                    })
                    
                    # Prepare input for next step
                    if i < len(self._behaviors) - 1:
                        # Not the last step, need to transform input
                        if isinstance(result, np.ndarray):
                            current_input = result
                        elif isinstance(result, (int, float)):
                            current_input = np.array([float(result)], dtype=np.float64)
                        elif isinstance(result, (list, tuple)):
                            current_input = np.array(result, dtype=np.float64)
                        elif isinstance(result, str):
                            # For strings before last step, convert to length
                            current_input = np.array([float(len(result))], dtype=np.float64)
                        else:
                            # Cannot transform, use original features
                            self._logger.warning(f"  Step {i+1} result cannot be converted to array, using original features")
                            current_input = features
                    else:
                        # Last step, return result directly
                        final_result = result
                        
                except Exception as e:
                    self._logger.error(f"  Step {i+1} failed: {e}")
                    if self._stop_on_error:
                        raise
                    # Continue execution but record error
                    if step_results:
                        step_results[-1]['error'] = str(e)
                    current_input = features  # Reset input
                    final_result = None  # Set default result
            
            # Update execution time
            self._last_execution_time = time.time() - start
            self._total_execution_time += self._last_execution_time
            
            # Record step results in metadata
            self._last_step_results = step_results
            
            self.after_execute(final_result, self._last_execution_time)
            
            return final_result
            
        except Exception as e:
            self._last_execution_time = time.time() - start
            self._total_execution_time += self._last_execution_time
            self.on_error(e)
            raise BehaviorExecutionError(f"Composite behavior {self.name} failed: {e}") from e
    
    def add_behavior(self, behavior: MetaBehavior) -> 'CompositeBehavior':
        """Add sub-behavior to the end."""
        self._behaviors.append(behavior)
        return self
    
    def insert_behavior(self, index: int, behavior: MetaBehavior) -> 'CompositeBehavior':
        """Insert sub-behavior at specified position."""
        self._behaviors.insert(index, behavior)
        return self
    
    def remove_behavior(self, behavior_name: str) -> bool:
        """Remove sub-behavior by name."""
        for i, behavior in enumerate(self._behaviors):
            if behavior.name == behavior_name:
                self._behaviors.pop(i)
                return True
        return False
    
    def get_step_results(self) -> List[Dict]:
        """Get step results from most recent execution."""
        return getattr(self, '_last_step_results', [])
    
    def get_complexity(self) -> float:
        """Complexity is sum of sub-behavior complexities."""
        return sum(b.get_complexity() for b in self._behaviors)


# ============================================================
# Retry Behavior
# ============================================================

class RetryBehavior(MetaBehavior):
    """
    Retry behavior: Adds retry mechanism to another behavior.
    
    Automatically retries on failure to improve system robustness.
    Supports exponential backoff and maximum retry limits.
    
    Characteristics:
        - Auto-retry: Automatically retries on failure
        - Exponential backoff: Retry delay increases progressively
        - Selective retry: Can specify which exceptions to retry
        - Configurable: Max retries, initial delay, etc.
    """
    
    def __init__(self,
                 name: str,
                 behavior: MetaBehavior,
                 max_retries: int = 3,
                 delay: float = 0.1,
                 backoff_factor: float = 1.0,
                 retry_on_exceptions: Optional[List[type]] = None):
        """
        Initialize retry behavior.
        
        Args:
            name: Behavior name
            behavior: Behavior to wrap
            max_retries: Maximum number of retry attempts
            delay: Initial retry delay (seconds)
            backoff_factor: Backoff multiplier for each retry
            retry_on_exceptions: List of exception types to retry, None means retry all
        """
        super().__init__()
        
        self._name = name
        self._behavior = behavior
        self._max_retries = max_retries
        self._delay = delay
        self._backoff_factor = backoff_factor
        self._retry_on_exceptions = retry_on_exceptions
        self._logger = logging.getLogger(f"metathin.behavior.{name}")
    
    @property
    def name(self) -> str:
        return self._name
    
    @behavior_logger
    def execute(self, features: FeatureVector, **kwargs) -> Any:
        """
        Execute with retry logic.
        
        Execution flow:
            1. Attempt to execute original behavior
            2. If failed and under max retries, wait and retry
            3. Delay increases exponentially
            4. Raise exception after all retries fail
        """
        self._execution_count += 1
        start = time.time()
        last_error = None
        current_delay = self._delay
        
        self._logger.debug(f"Starting execution, max retries: {self._max_retries}")
        
        for attempt in range(self._max_retries + 1):
            try:
                self.before_execute(features)
                result = self._behavior.execute(features, **kwargs)
                
                if attempt > 0:
                    self._logger.info(f"Attempt {attempt} succeeded")
                
                self._last_execution_time = time.time() - start
                self._total_execution_time += self._last_execution_time
                
                self.after_execute(result, self._last_execution_time)
                
                return result
                
            except Exception as e:
                last_error = e
                
                # Check if this exception should be retried
                if self._retry_on_exceptions is not None:
                    if not any(isinstance(e, exc_type) for exc_type in self._retry_on_exceptions):
                        self._logger.debug(f"Exception {type(e).__name__} not in retry list, stopping")
                        break
                
                if attempt < self._max_retries:
                    wait_time = current_delay
                    self._logger.warning(f"Attempt {attempt + 1}/{self._max_retries + 1} failed: {e}, waiting {wait_time:.2f}s before retry")
                    
                    time.sleep(wait_time)
                    current_delay *= self._backoff_factor
                else:
                    self._logger.error(f"All {self._max_retries + 1} attempts failed")
        
        self._last_execution_time = time.time() - start
        self._total_execution_time += self._last_execution_time
        self.on_error(last_error)
        
        raise BehaviorExecutionError(f"Failed after {self._max_retries} retries: {last_error}") from last_error
    
    def get_complexity(self) -> float:
        """Complexity is original behavior complexity multiplied by retry count."""
        return self._behavior.get_complexity() * (self._max_retries + 1)


# ============================================================
# Timeout Behavior
# ============================================================

class TimeoutBehavior(MetaBehavior):
    """
    Timeout behavior: Adds timeout control to another behavior.
    
    Terminates if execution exceeds specified time, preventing blocking.
    Implemented using threading, be aware of thread safety.
    
    Characteristics:
        - Timeout control: Prevents long-running operations from blocking
        - Thread-safe: Uses queues for thread communication
        - Default value: Can specify return value on timeout
        - Non-interrupting: Doesn't interrupt the original thread on timeout
    """
    
    def __init__(self,
                 name: str,
                 behavior: MetaBehavior,
                 timeout: float = 5.0,
                 timeout_result: Any = None):
        """
        Initialize timeout behavior.
        
        Args:
            name: Behavior name
            behavior: Behavior to wrap
            timeout: Timeout in seconds
            timeout_result: Default return value on timeout
        """
        super().__init__()
        
        self._name = name
        self._behavior = behavior
        self._timeout = timeout
        self._timeout_result = timeout_result
        self._logger = logging.getLogger(f"metathin.behavior.{name}")
    
    @property
    def name(self) -> str:
        return self._name
    
    @behavior_logger
    def execute(self, features: FeatureVector, **kwargs) -> Any:
        """
        Execute with timeout control.
        
        Uses queue and thread for timeout detection.
        """
        self._execution_count += 1
        start = time.time()
        
        # Use queues to receive results and errors
        result_queue = queue.Queue()
        error_queue = queue.Queue()
        
        def target():
            """Thread target function."""
            try:
                self.before_execute(features)
                result = self._behavior.execute(features, **kwargs)
                result_queue.put(result)
            except Exception as e:
                error_queue.put(e)
        
        # Create and start thread
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        
        # Wait for thread completion or timeout
        thread.join(timeout=self._timeout)
        
        elapsed = time.time() - start
        self._last_execution_time = elapsed
        self._total_execution_time += elapsed
        
        # Check results
        if thread.is_alive():
            # Thread still running, timeout occurred
            self._logger.warning(f"Behavior execution timed out ({self._timeout}s)")
            
            if self._timeout_result is not None:
                self.after_execute(self._timeout_result, elapsed)
                return self._timeout_result
            else:
                raise TimeoutError(f"Behavior {self._name} execution timed out ({self._timeout}s)")
        
        # Check for errors
        if not error_queue.empty():
            error = error_queue.get()
            self.on_error(error)
            raise BehaviorExecutionError(f"Behavior {self._name} execution failed: {error}") from error
        
        # Get result
        result = result_queue.get()
        self.after_execute(result, elapsed)
        
        return result


# ============================================================
# Conditional Behavior
# ============================================================

class ConditionalBehavior(MetaBehavior):
    """
    Conditional behavior: Selects different sub-behaviors based on conditions.
    
    Implements decision branching logic, dynamically choosing execution path
    based on feature vectors.
    
    Characteristics:
        - Branch selection: Chooses different paths based on conditions
        - Extensible: Can handle only true branch if false branch is omitted
        - Observable: Records branch selection for each execution
        - Statistical: Provides branch execution statistics
    """
    
    def __init__(self,
                 name: str,
                 condition: Callable[[FeatureVector], bool],
                 true_behavior: MetaBehavior,
                 false_behavior: Optional[MetaBehavior] = None):
        """
        Initialize conditional behavior.
        
        Args:
            name: Behavior name
            condition: Condition function, receives feature vector and returns boolean
            true_behavior: Behavior to execute when condition is true
            false_behavior: Behavior to execute when condition is false (optional)
        """
        super().__init__()
        
        self._name = name
        self._condition = condition
        self._true_behavior = true_behavior
        self._false_behavior = false_behavior
        self._logger = logging.getLogger(f"metathin.behavior.{name}")
        
        # Track execution history
        self._execution_history: List[Dict] = []
    
    @property
    def name(self) -> str:
        return self._name
    
    @behavior_logger
    def execute(self, features: FeatureVector, **kwargs) -> Any:
        """
        Execute based on condition.
        """
        self._execution_count += 1
        start = time.time()
        
        self.before_execute(features)
        
        try:
            # Evaluate condition
            condition_result = self._condition(features)
            
            self._logger.debug(f"Condition result: {condition_result}")
            
            # Record execution
            self._execution_history.append({
                'timestamp': time.time(),
                'features': features.tolist() if hasattr(features, 'tolist') else features,
                'condition': condition_result,
                'chosen': 'true' if condition_result else ('false' if self._false_behavior else 'none')
            })
            
            # Limit history length
            if len(self._execution_history) > 100:
                self._execution_history = self._execution_history[-100:]
            
            # Execute based on condition
            if condition_result:
                result = self._true_behavior.execute(features, **kwargs)
            elif self._false_behavior:
                result = self._false_behavior.execute(features, **kwargs)
            else:
                result = None
            
            self._last_execution_time = time.time() - start
            self._total_execution_time += self._last_execution_time
            
            self.after_execute(result, self._last_execution_time)
            
            return result
            
        except Exception as e:
            self._last_execution_time = time.time() - start
            self._total_execution_time += self._last_execution_time
            self.on_error(e)
            raise BehaviorExecutionError(f"Conditional behavior {self.name} failed: {e}") from e
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Get execution statistics.
        
        Returns:
            Dict: Statistics including true/false branch counts
        """
        true_count = sum(1 for h in self._execution_history if h['condition'])
        false_count = len(self._execution_history) - true_count
        
        return {
            'total_executions': len(self._execution_history),
            'true_branch': true_count,
            'false_branch': false_count,
            'true_ratio': true_count / max(1, len(self._execution_history))
        }


# ============================================================
# Cached Behavior
# ============================================================

class CachedBehavior(MetaBehavior):
    """
    Cached behavior: Adds result caching to another behavior.
    
    Caches results for identical inputs to improve efficiency.
    Supports cache size limits and time-to-live (TTL) expiration.
    
    Characteristics:
        - Performance optimization: Avoids redundant computations
        - LRU strategy: Automatically evicts oldest cache entries
        - TTL support: Can set cache expiration time
        - Statistical: Provides cache hit rate statistics
    """
    
    def __init__(self,
                 name: str,
                 behavior: MetaBehavior,
                 cache_size: int = 100,
                 ttl: Optional[float] = None):
        """
        Initialize cached behavior.
        
        Args:
            name: Behavior name
            behavior: Behavior to wrap
            cache_size: Maximum number of cache entries
            ttl: Cache time-to-live in seconds, None means never expire
        """
        super().__init__()
        
        self._name = name
        self._behavior = behavior
        self._cache_size = cache_size
        self._ttl = ttl
        self._logger = logging.getLogger(f"metathin.behavior.{name}")
        
        # Cache structure: {key: (timestamp, result)}
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def _make_key(self, features: FeatureVector, kwargs: Dict) -> str:
        """Generate cache key."""
        feat_str = ','.join(f"{x:.6f}" for x in features)
        kwargs_str = ','.join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return f"{feat_str}|{kwargs_str}"
    
    def _clean_expired(self):
        """Clean expired cache entries."""
        if self._ttl is None:
            return
        
        now = time.time()
        expired_keys = [
            key for key, (timestamp, _) in self._cache.items()
            if now - timestamp > self._ttl
        ]
        
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            self._logger.debug(f"Cleaned {len(expired_keys)} expired cache entries")
    
    def _enforce_size_limit(self):
        """Enforce cache size limit."""
        if len(self._cache) <= self._cache_size:
            return
        
        # Sort by timestamp, delete oldest
        sorted_items = sorted(self._cache.items(), key=lambda x: x[1][0])
        to_remove = len(self._cache) - self._cache_size
        
        for key, _ in sorted_items[:to_remove]:
            del self._cache[key]
        
        self._logger.debug(f"Removed {to_remove} oldest cache entries")
    
    @property
    def name(self) -> str:
        return self._name
    
    @behavior_logger
    def execute(self, features: FeatureVector, **kwargs) -> Any:
        """
        Execute with caching.
        
        Execution flow:
            1. Generate cache key
            2. Check cache hit
            3. If hit, return cached result
            4. If miss, execute original behavior and cache result
        """
        self._execution_count += 1
        start = time.time()
        
        # Generate cache key
        cache_key = self._make_key(features, kwargs)
        
        # Clean expired entries
        self._clean_expired()
        
        # Try cache first
        if cache_key in self._cache:
            timestamp, result = self._cache[cache_key]
            self._cache_hits += 1
            self._last_execution_time = time.time() - start
            self._total_execution_time += self._last_execution_time
            
            self._logger.debug(f"Cache hit: {cache_key}")
            return result
        
        self._cache_misses += 1
        self._logger.debug(f"Cache miss: {cache_key}")
        
        # Execute original behavior
        try:
            self.before_execute(features)
            result = self._behavior.execute(features, **kwargs)
            
            # Store in cache
            self._cache[cache_key] = (time.time(), result)
            self._enforce_size_limit()
            
            self._last_execution_time = time.time() - start
            self._total_execution_time += self._last_execution_time
            
            self.after_execute(result, self._last_execution_time)
            
            return result
            
        except Exception as e:
            self._last_execution_time = time.time() - start
            self._total_execution_time += self._last_execution_time
            self.on_error(e)
            raise
    
    def clear_cache(self):
        """Clear all cached results."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._cache_hits + self._cache_misses
        return {
            'size': len(self._cache),
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'hit_rate': self._cache_hits / max(1, total),
            'capacity': self._cache_size,
            'ttl': self._ttl
        }