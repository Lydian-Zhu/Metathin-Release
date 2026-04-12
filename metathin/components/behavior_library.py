"""
Meta-Behavior Components | 元行为组件
=======================================

Provides various executable skill units. All behaviors inherit from the MetaBehavior interface.
Behaviors are the agent's "action" modules, responsible for executing specific tasks.

提供各种可执行的技能单元。所有行为都继承自 MetaBehavior 接口。
行为是代理的"行动"模块，负责执行具体任务。

Component Types | 组件类型:
    - FunctionBehavior: Function-based behavior | 基于函数的行为
    - LambdaBehavior: Lambda-based behavior | 基于 Lambda 的行为
    - CompositeBehavior: Sequential composite behavior | 顺序组合行为
    - RetryBehavior: Auto-retry behavior | 自动重试行为
    - TimeoutBehavior: Timeout-controlled behavior | 超时控制行为
    - ConditionalBehavior: Conditional branching behavior | 条件分支行为
    - CachedBehavior: Result-caching behavior | 结果缓存行为

Design Philosophy | 设计理念:
    - Composition over Inheritance: Create complex behaviors by wrapping existing ones
      组合优于继承：通过包装现有行为创建复杂行为
    - Separation of Concerns: Each behavior focuses on a single responsibility
      关注点分离：每个行为专注于单一职责
    - Composability: Behaviors can be nested and combined
      可组合性：行为可以嵌套和组合
    - Robustness: Comprehensive error handling and retry mechanisms
      健壮性：全面的错误处理和重试机制
"""

import time
import numpy as np
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from functools import wraps
import logging
import threading
import queue
from datetime import datetime

# ============================================================
# Fixed Imports for Refactored Core | 重构后核心的修复导入
# ============================================================
from ..core.b_behavior import MetaBehavior
from ..core.types import FeatureVector
from ..core.exceptions import BehaviorExecutionError


# ============================================================
# Decorators and Helper Functions | 装饰器和辅助函数
# ============================================================

def behavior_logger(func: Callable) -> Callable:
    """
    Behavior logging decorator.
    
    行为日志装饰器。
    
    Automatically logs execution time, input, and output information.
    Used for debugging and performance analysis.
    
    自动记录执行时间、输入和输出信息。
    用于调试和性能分析。
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
# 1. Function-Based Behavior | 基于函数的行为
# ============================================================

class FunctionBehavior(MetaBehavior):
    """
    Function-based behavior: Implements behavior using regular Python functions.
    
    基于函数的行为：使用常规 Python 函数实现行为。
    
    The simplest way to implement behaviors, suitable for wrapping existing functions.
    Functions should accept feature vectors and keyword arguments, returning any result.
    
    实现行为的最简单方式，适合包装现有函数。
    函数应接受特征向量和关键字参数，返回任意结果。
    
    Characteristics | 特性:
        - Simple and intuitive: Directly uses Python functions | 简单直观：直接使用 Python 函数
        - Flexible: Can encapsulate arbitrary logic | 灵活：可以封装任意逻辑
        - Documentable: Can include docstrings for description | 可文档化：可以包含文档字符串
    
    Example | 示例:
        >>> def process_text(features, **kwargs):
        ...     '''Process text features | 处理文本特征'''
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
        
        初始化基于函数的行为。
        
        Args | 参数:
            name: Behavior name, must be unique | 行为名称，必须唯一
            func: Execution function, receives feature vector and keyword arguments
                 执行函数，接收特征向量和关键字参数
            complexity: Behavior complexity for resource evaluation | 行为复杂度，用于资源评估
            description: Behavior description for documentation | 行为描述，用于文档
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
        
        执行函数。
        
        Args | 参数:
            features: Feature vector | 特征向量
            **kwargs: Additional parameters | 额外参数
            
        Returns | 返回:
            Any: Function execution result | 函数执行结果
        """
        try:
            # Update statistics | 更新统计信息
            self._execution_count += 1
            start = time.time()
            
            # Pre-execution hook | 执行前钩子
            self.before_execute(features)
            
            # Execute function | 执行函数
            result = self._func(features, **kwargs)
            
            # Update execution time | 更新执行时间
            self._last_execution_time = time.time() - start
            self._total_execution_time += self._last_execution_time
            
            # Post-execution hook | 执行后钩子
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
# 2. Lambda Behavior | Lambda 行为
# ============================================================

class LambdaBehavior(MetaBehavior):
    """
    Lambda behavior: Creates simple behaviors using lambda expressions.
    
    Lambda 行为：使用 lambda 表达式创建简单行为。
    
    Suitable for quickly creating simple, one-off behaviors.
    Note: Lambda expressions cannot be serialized, so not suitable for scenarios requiring persistence.
    
    适合快速创建简单的、一次性的行为。
    注意：Lambda 表达式不能被序列化，因此不适合需要持久化的场景。
    
    Characteristics | 特性:
        - Concise: Define behaviors in one line | 简洁：一行定义行为
        - Quick: Ideal for prototyping | 快速：适合原型开发
        - Temporary: Not suitable for serialization | 临时：不适合序列化
    """
    
    def __init__(self, 
                 name: str, 
                 func: Callable,
                 complexity: float = 0.5):
        """
        Initialize lambda behavior.
        
        初始化 lambda 行为。
        
        Args | 参数:
            name: Behavior name | 行为名称
            func: Lambda function | Lambda 函数
            complexity: Behavior complexity | 行为复杂度
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
        """Execute lambda | 执行 lambda"""
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
# 3. Composite Behavior | 组合行为
# ============================================================

class CompositeBehavior(MetaBehavior):
    """
    Composite behavior: Executes multiple sub-behaviors in sequence.
    
    组合行为：按顺序执行多个子行为。
    
    Combines multiple behaviors into a pipeline, where the output of one behavior
    becomes the input to the next. Supports data flow transformation and error handling.
    
    将多个行为组合成一个流水线，其中一个行为的输出成为下一个行为的输入。
    支持数据流转换和错误处理。
    
    Characteristics | 特性:
        - Sequential execution: Strictly executes sub-behaviors in order | 顺序执行：严格按顺序执行子行为
        - Data flow: Automatically handles data type conversions between steps | 数据流：自动处理步骤间的数据类型转换
        - Error control: Option to stop or continue on error | 错误控制：可选择在错误时停止或继续
        - Observable: Records execution results for each step | 可观测：记录每个步骤的执行结果
    """
    
    def __init__(self, 
                 name: str,
                 behaviors: List[MetaBehavior],
                 stop_on_error: bool = True,
                 description: Optional[str] = None):
        """
        Initialize composite behavior.
        
        初始化组合行为。
        
        Args | 参数:
            name: Behavior name | 行为名称
            behaviors: List of sub-behaviors to execute in sequence | 按顺序执行的子行为列表
            stop_on_error: Whether to stop execution when a step fails | 步骤失败时是否停止执行
            description: Behavior description | 行为描述
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
        
        按顺序执行所有子行为。
        
        Execution flow | 执行流程:
            1. Execute each sub-behavior in order | 按顺序执行每个子行为
            2. Transform previous result to next behavior's input | 将前一个结果转换为下一个行为的输入
            3. Record execution results for each step | 记录每个步骤的执行结果
            4. Continue or stop based on stop_on_error setting | 根据 stop_on_error 设置决定继续或停止
        """
        self._execution_count += 1
        current_input = features
        start = time.time()
        step_results = []
        final_result = None
        
        try:
            self.before_execute(features)
            
            for i, behavior in enumerate(self._behaviors):
                step_start = time.time()
                
                self._logger.debug(f"  Step {i+1}/{len(self._behaviors)}: {behavior.name}")
                
                try:
                    # Execute sub-behavior | 执行子行为
                    result = behavior.execute(current_input, **kwargs)
                    step_results.append({
                        'step': i,
                        'behavior': behavior.name,
                        'result': result,
                        'time': time.time() - step_start
                    })
                    
                    # Prepare input for next step | 准备下一步的输入
                    if i < len(self._behaviors) - 1:
                        # Not the last step, need to transform input | 不是最后一步，需要转换输入
                        if isinstance(result, np.ndarray):
                            current_input = result
                        elif isinstance(result, (int, float)):
                            current_input = np.array([float(result)], dtype=np.float64)
                        elif isinstance(result, (list, tuple)):
                            current_input = np.array(result, dtype=np.float64)
                        elif isinstance(result, str):
                            # For strings before last step, convert to length | 最后一步之前的字符串，转换为长度
                            current_input = np.array([float(len(result))], dtype=np.float64)
                        else:
                            # Cannot transform, use original features | 无法转换，使用原始特征
                            self._logger.warning(f"  Step {i+1} result cannot be converted to array, using original features")
                            current_input = features
                    else:
                        # Last step, return result directly | 最后一步，直接返回结果
                        final_result = result
                        
                except Exception as e:
                    self._logger.error(f"  Step {i+1} failed: {e}")
                    if self._stop_on_error:
                        raise
                    # Continue execution but record error | 继续执行但记录错误
                    if step_results:
                        step_results[-1]['error'] = str(e)
                    current_input = features  # Reset input | 重置输入
                    final_result = None  # Set default result | 设置默认结果
            
            # Update execution time | 更新执行时间
            self._last_execution_time = time.time() - start
            self._total_execution_time += self._last_execution_time
            
            # Record step results in metadata | 在元数据中记录步骤结果
            self._last_step_results = step_results
            
            self.after_execute(final_result, self._last_execution_time)
            
            return final_result
            
        except Exception as e:
            self._last_execution_time = time.time() - start
            self._total_execution_time += self._last_execution_time
            self.on_error(e)
            raise BehaviorExecutionError(f"Composite behavior {self.name} failed: {e}") from e
    
    def add_behavior(self, behavior: MetaBehavior) -> 'CompositeBehavior':
        """Add sub-behavior to the end | 在末尾添加子行为"""
        self._behaviors.append(behavior)
        return self
    
    def insert_behavior(self, index: int, behavior: MetaBehavior) -> 'CompositeBehavior':
        """Insert sub-behavior at specified position | 在指定位置插入子行为"""
        self._behaviors.insert(index, behavior)
        return self
    
    def remove_behavior(self, behavior_name: str) -> bool:
        """Remove sub-behavior by name | 按名称移除子行为"""
        for i, behavior in enumerate(self._behaviors):
            if behavior.name == behavior_name:
                self._behaviors.pop(i)
                return True
        return False
    
    def get_step_results(self) -> List[Dict]:
        """Get step results from most recent execution | 获取最近执行的步骤结果"""
        return getattr(self, '_last_step_results', [])
    
    def get_complexity(self) -> float:
        """Complexity is sum of sub-behavior complexities | 复杂度是子行为复杂度之和"""
        return sum(b.get_complexity() for b in self._behaviors)


# ============================================================
# 4. Retry Behavior | 重试行为
# ============================================================

class RetryBehavior(MetaBehavior):
    """
    Retry behavior: Adds retry mechanism to another behavior.
    
    重试行为：为另一个行为添加重试机制。
    
    Automatically retries on failure to improve system robustness.
    Supports exponential backoff and maximum retry limits.
    
    在失败时自动重试，提高系统健壮性。
    支持指数退避和最大重试次数限制。
    
    Characteristics | 特性:
        - Auto-retry: Automatically retries on failure | 自动重试：失败时自动重试
        - Exponential backoff: Retry delay increases progressively | 指数退避：重试延迟逐渐增加
        - Selective retry: Can specify which exceptions to retry | 选择性重试：可指定哪些异常需要重试
        - Configurable: Max retries, initial delay, etc. | 可配置：最大重试次数、初始延迟等
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
        
        初始化重试行为。
        
        Args | 参数:
            name: Behavior name | 行为名称
            behavior: Behavior to wrap | 要包装的行为
            max_retries: Maximum number of retry attempts | 最大重试次数
            delay: Initial retry delay (seconds) | 初始重试延迟（秒）
            backoff_factor: Backoff multiplier for each retry | 每次重试的退避乘数
            retry_on_exceptions: List of exception types to retry, None means retry all
                                 需要重试的异常类型列表，None 表示重试所有
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
        
        使用重试逻辑执行。
        
        Execution flow | 执行流程:
            1. Attempt to execute original behavior | 尝试执行原始行为
            2. If failed and under max retries, wait and retry | 如果失败且未超过最大重试次数，等待后重试
            3. Delay increases exponentially | 延迟指数增长
            4. Raise exception after all retries fail | 所有重试失败后抛出异常
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
                
                # Check if this exception should be retried | 检查此异常是否应该重试
                if self._retry_on_exceptions is not None:
                    # 获取原始异常（如果是 BehaviorExecutionError 包装的）
                    original_error = e
                    if hasattr(e, '__cause__') and e.__cause__ is not None:
                        original_error = e.__cause__
                    if not any(isinstance(original_error, exc_type) for exc_type in self._retry_on_exceptions):
                        self._logger.debug(f"Exception {type(original_error).__name__} not in retry list, stopping")
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
        """Complexity is original behavior complexity multiplied by retry count | 复杂度是原始行为复杂度乘以重试次数"""
        return self._behavior.get_complexity() * (self._max_retries + 1)


# ============================================================
# 5. Timeout Behavior | 超时行为
# ============================================================

class TimeoutBehavior(MetaBehavior):
    """
    Timeout behavior: Adds timeout control to another behavior.
    
    超时行为：为另一个行为添加超时控制。
    
    Terminates if execution exceeds specified time, preventing blocking.
    Implemented using threading, be aware of thread safety.
    
    如果执行超过指定时间则终止，防止阻塞。
    使用线程实现，注意线程安全。
    
    Characteristics | 特性:
        - Timeout control: Prevents long-running operations from blocking | 超时控制：防止长时间运行的操作阻塞
        - Thread-safe: Uses queues for thread communication | 线程安全：使用队列进行线程通信
        - Default value: Can specify return value on timeout | 默认值：可指定超时时的返回值
        - Non-interrupting: Doesn't interrupt the original thread on timeout | 非中断：超时时不中断原始线程
    """
    
    def __init__(self,
                 name: str,
                 behavior: MetaBehavior,
                 timeout: float = 5.0,
                 timeout_result: Any = None):
        """
        Initialize timeout behavior.
        
        初始化超时行为。
        
        Args | 参数:
            name: Behavior name | 行为名称
            behavior: Behavior to wrap | 要包装的行为
            timeout: Timeout in seconds | 超时时间（秒）
            timeout_result: Default return value on timeout | 超时时的默认返回值
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
        
        使用超时控制执行。
        
        Uses queue and thread for timeout detection.
        
        使用队列和线程进行超时检测。
        """
        self._execution_count += 1
        start = time.time()
        
        # Use queues to receive results and errors | 使用队列接收结果和错误
        result_queue = queue.Queue()
        error_queue = queue.Queue()
        
        def target():
            """Thread target function | 线程目标函数"""
            try:
                self.before_execute(features)
                result = self._behavior.execute(features, **kwargs)
                result_queue.put(result)
            except Exception as e:
                error_queue.put(e)
        
        # Create and start thread | 创建并启动线程
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        
        # Wait for thread completion or timeout | 等待线程完成或超时
        thread.join(timeout=self._timeout)
        
        elapsed = time.time() - start
        self._last_execution_time = elapsed
        self._total_execution_time += elapsed
        
        # Check results | 检查结果
        if thread.is_alive():
            # Thread still running, timeout occurred | 线程仍在运行，发生超时
            self._logger.warning(f"Behavior execution timed out ({self._timeout}s)")
            
            if self._timeout_result is not None:
                self.after_execute(self._timeout_result, elapsed)
                return self._timeout_result
            else:
                raise TimeoutError(f"Behavior {self._name} execution timed out ({self._timeout}s)")
        
        # Check for errors | 检查错误
        if not error_queue.empty():
            error = error_queue.get()
            self.on_error(error)
            raise BehaviorExecutionError(f"Behavior {self._name} execution failed: {error}") from error
        
        # Get result | 获取结果
        result = result_queue.get()
        self.after_execute(result, elapsed)
        
        return result


# ============================================================
# 6. Conditional Behavior | 条件行为
# ============================================================

class ConditionalBehavior(MetaBehavior):
    """
    Conditional behavior: Selects different sub-behaviors based on conditions.
    
    条件行为：根据条件选择不同的子行为。
    
    Implements decision branching logic, dynamically choosing execution path
    based on feature vectors.
    
    实现决策分支逻辑，根据特征向量动态选择执行路径。
    
    Characteristics | 特性:
        - Branch selection: Chooses different paths based on conditions | 分支选择：根据条件选择不同路径
        - Extensible: Can handle only true branch if false branch is omitted | 可扩展：如果省略 false 分支，可只处理 true 分支
        - Observable: Records branch selection for each execution | 可观测：记录每次执行的分支选择
        - Statistical: Provides branch execution statistics | 统计：提供分支执行统计
    """
    
    def __init__(self,
                 name: str,
                 condition: Callable[[FeatureVector], bool],
                 true_behavior: MetaBehavior,
                 false_behavior: Optional[MetaBehavior] = None):
        """
        Initialize conditional behavior.
        
        初始化条件行为。
        
        Args | 参数:
            name: Behavior name | 行为名称
            condition: Condition function, receives feature vector and returns boolean
                       条件函数，接收特征向量并返回布尔值
            true_behavior: Behavior to execute when condition is true | 条件为真时执行的行为
            false_behavior: Behavior to execute when condition is false (optional) | 条件为假时执行的行为（可选）
        """
        super().__init__()
        
        self._name = name
        self._condition = condition
        self._true_behavior = true_behavior
        self._false_behavior = false_behavior
        self._logger = logging.getLogger(f"metathin.behavior.{name}")
        
        # Track execution history | 跟踪执行历史
        self._execution_history: List[Dict] = []
    
    @property
    def name(self) -> str:
        return self._name
    
    @behavior_logger
    def execute(self, features: FeatureVector, **kwargs) -> Any:
        """
        Execute based on condition.
        
        根据条件执行。
        """
        self._execution_count += 1
        start = time.time()
        
        self.before_execute(features)
        
        try:
            # Evaluate condition | 评估条件
            condition_result = self._condition(features)
            
            self._logger.debug(f"Condition result: {condition_result}")
            
            # Record execution | 记录执行
            self._execution_history.append({
                'timestamp': time.time(),
                'features': features.tolist() if hasattr(features, 'tolist') else features,
                'condition': condition_result,
                'chosen': 'true' if condition_result else ('false' if self._false_behavior else 'none')
            })
            
            # Limit history length | 限制历史长度
            if len(self._execution_history) > 100:
                self._execution_history = self._execution_history[-100:]
            
            # Execute based on condition | 根据条件执行
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
        
        获取执行统计信息。
        
        Returns | 返回:
            Dict: Statistics including true/false branch counts | 包含 true/false 分支计数的统计信息
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
# 7. Cached Behavior | 缓存行为
# ============================================================

class CachedBehavior(MetaBehavior):
    """
    Cached behavior: Adds result caching to another behavior.
    
    缓存行为：为另一个行为添加结果缓存。
    
    Caches results for identical inputs to improve efficiency.
    Supports cache size limits and time-to-live (TTL) expiration.
    
    缓存相同输入的结果以提高效率。
    支持缓存大小限制和生存时间（TTL）过期。
    
    Characteristics | 特性:
        - Performance optimization: Avoids redundant computations | 性能优化：避免冗余计算
        - LRU strategy: Automatically evicts oldest cache entries | LRU 策略：自动淘汰最旧的缓存条目
        - TTL support: Can set cache expiration time | TTL 支持：可设置缓存过期时间
        - Statistical: Provides cache hit rate statistics | 统计：提供缓存命中率统计
    """
    
    def __init__(self,
                 name: str,
                 behavior: MetaBehavior,
                 cache_size: int = 100,
                 ttl: Optional[float] = None):
        """
        Initialize cached behavior.
        
        初始化缓存行为。
        
        Args | 参数:
            name: Behavior name | 行为名称
            behavior: Behavior to wrap | 要包装的行为
            cache_size: Maximum number of cache entries | 最大缓存条目数
            ttl: Cache time-to-live in seconds, None means never expire | 缓存生存时间（秒），None 表示永不过期
        """
        super().__init__()
        
        self._name = name
        self._behavior = behavior
        self._cache_size = cache_size
        self._ttl = ttl
        self._logger = logging.getLogger(f"metathin.behavior.{name}")
        
        # Cache structure: {key: (timestamp, result)} | 缓存结构：{键: (时间戳, 结果)}
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def _make_key(self, features: FeatureVector, kwargs: Dict) -> str:
        """Generate cache key | 生成缓存键"""
        feat_str = ','.join(f"{x:.6f}" for x in features)
        kwargs_str = ','.join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return f"{feat_str}|{kwargs_str}"
    
    def _clean_expired(self):
        """Clean expired cache entries | 清理过期的缓存条目"""
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
        """Enforce cache size limit | 强制执行缓存大小限制"""
        if len(self._cache) <= self._cache_size:
            return
        
        # Sort by timestamp, delete oldest | 按时间戳排序，删除最旧的
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
        
        使用缓存执行。
        
        Execution flow | 执行流程:
            1. Generate cache key | 生成缓存键
            2. Check cache hit | 检查缓存命中
            3. If hit, return cached result | 如果命中，返回缓存结果
            4. If miss, execute original behavior and cache result | 如果未命中，执行原始行为并缓存结果
        """
        self._execution_count += 1
        start = time.time()
        
        # Generate cache key | 生成缓存键
        cache_key = self._make_key(features, kwargs)
        
        # Clean expired entries | 清理过期条目
        self._clean_expired()
        
        # Try cache first | 先尝试缓存
        if cache_key in self._cache:
            timestamp, result = self._cache[cache_key]
            self._cache_hits += 1
            self._last_execution_time = time.time() - start
            self._total_execution_time += self._last_execution_time
            
            self._logger.debug(f"Cache hit: {cache_key}")
            return result
        
        self._cache_misses += 1
        self._logger.debug(f"Cache miss: {cache_key}")
        
        # Execute original behavior | 执行原始行为
        try:
            self.before_execute(features)
            result = self._behavior.execute(features, **kwargs)
            
            # Store in cache | 存储到缓存
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
        """Clear all cached results | 清空所有缓存结果"""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        self._logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        获取缓存统计信息。
        
        Returns | 返回:
            Dict: Cache statistics | 缓存统计信息
        """
        total = self._cache_hits + self._cache_misses
        return {
            'size': len(self._cache),
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'hit_rate': self._cache_hits / max(1, total),
            'capacity': self._cache_size,
            'ttl': self._ttl
        }


# ============================================================
# Export Interface | 导出接口
# ============================================================

__all__ = [
    'FunctionBehavior',
    'LambdaBehavior',
    'CompositeBehavior',
    'RetryBehavior',
    'TimeoutBehavior',
    'ConditionalBehavior',
    'CachedBehavior',
]