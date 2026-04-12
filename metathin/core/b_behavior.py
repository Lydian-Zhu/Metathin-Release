"""
MetaBehavior Interface (B) - Action Layer | 元行为接口 (B) - 行动层
====================================================================

Defines the MetaBehavior interface - the agent's executable skill units.
Each behavior focuses on solving a well-defined subproblem.

定义元行为接口 - 代理的可执行技能单元。
每个行为专注于解决一个明确定义的子问题。

Role in Quintuple | 五元组中的角色:
    B (MetaBehavior): Action - Executable skill units
    B (MetaBehavior): 行动 - 可执行的技能单元

Design Philosophy | 设计理念:
    - Focused: Each behavior solves one subproblem | 聚焦：每个行为解决一个子问题
    - Observable: Provides execution statistics | 可观测：提供执行统计
    - Extensible: Hook methods for before/after execution | 可扩展：执行前后的钩子方法
    - Safe: Comprehensive error handling | 安全：全面的错误处理
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Generic
import logging
from datetime import datetime

from .types import FeatureVector, R
from .exceptions import BehaviorExecutionError


# ============================================================
# MetaBehavior Interface | 元行为接口
# ============================================================

class MetaBehavior(ABC, Generic[R]):
    """
    Meta-Behavior B: System's skill collection.
    
    元行为 B：系统的技能集合。
    
    Each behavior focuses on solving a well-defined subproblem. Behaviors can
    maintain internal state but should be thread-safe.
    
    每个行为专注于解决一个明确定义的子问题。行为可以维护内部状态，
    但应该是线程安全的。
    
    Implementation Requirements | 实现要求:
        - name property MUST be implemented (must be unique) | 必须实现 name 属性（必须唯一）
        - execute() method MUST be implemented | 必须实现 execute() 方法
        - can_execute(), get_complexity() are optional | can_execute()、get_complexity() 是可选的
        - Hook methods (before/after/on_error) can be overridden | 可覆盖钩子方法
    
    Type Parameters | 类型参数:
        R: Result type of behavior execution | 行为执行的结果类型
    
    Example | 示例:
        >>> class GreetBehavior(MetaBehavior[str]):
        ...     '''Greeting behavior | 问候行为'''
        ...     
        ...     def __init__(self):
        ...         super().__init__()
        ...         self._name = "greet"
        ...         self.greeted_count = 0
        ...     
        ...     @property
        ...     def name(self) -> str:
        ...         return self._name
        ...     
        ...     def execute(self, features: FeatureVector, **kwargs) -> str:
        ...         self.greeted_count += 1
        ...         length = features[0] if len(features) > 0 else 0
        ...         return f"Hello! (greeting #{self.greeted_count}, length={length})"
        ...     
        ...     def can_execute(self, features: FeatureVector) -> bool:
        ...         return np.all(features >= 0)  # Only for non-negative features
    """
    
    def __init__(self):
        """
        Initialize behavior base class.
        
        初始化行为基类。
        
        Sets up basic statistical attributes and logger.
        
        设置基本统计属性和日志记录器。
        """
        self._execution_count: int = 0
        """Number of times executed | 执行次数"""
        
        self._last_execution_time: float = 0.0
        """Last execution duration (seconds) | 上次执行时长（秒）"""
        
        self._total_execution_time: float = 0.0
        """Total execution duration (seconds) | 总执行时长（秒）"""
        
        self._last_error: Optional[Exception] = None
        """Last error encountered | 上次遇到的错误"""
        
        self._created_at: datetime = datetime.now()
        """Creation timestamp | 创建时间戳"""
        
        self._logger: logging.Logger = logging.getLogger(
            f"metathin.behavior.{self.__class__.__name__}"
        )
        """Logger instance for this behavior | 此行为的日志记录器实例"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Behavior name, used for identification and logging.
        
        行为名称，用于识别和日志记录。
        
        Must be unique across all registered behaviors. Meaningful names are recommended.
        
        在所有注册的行为中必须唯一。建议使用有意义的名称。
        
        Naming conventions | 命名规范:
            - Use snake_case | 使用蛇形命名
            - Reflect the behavior's function | 反映行为的功能
            - Avoid special characters | 避免特殊字符
        
        Returns | 返回:
            str: Unique behavior name | 唯一的行为名称
        """
        pass
    
    @abstractmethod
    def execute(self, features: FeatureVector, **kwargs) -> R:
        """
        Execute the behavior and return the result.
        
        执行行为并返回结果。
        
        This is the core method of the behavior, implementing specific business logic.
        
        这是行为的核心方法，实现具体的业务逻辑。
        
        Args | 参数:
            features: Current feature vector | 当前特征向量
            **kwargs: Additional context parameters | 额外的上下文参数
                - context: Global context information | 全局上下文信息
                - timeout: Timeout setting | 超时设置
                - callback: Callback function | 回调函数
            
        Returns | 返回:
            R: Behavior execution result | 行为执行结果
            
        Raises | 抛出:
            BehaviorExecutionError: When execution fails | 执行失败时
        """
        pass
    
    def can_execute(self, features: FeatureVector) -> bool:
        """
        Determine if execution is possible with the current features (optional override).
        
        判断当前特征是否可执行（可选覆盖）。
        
        Used to filter out unsuitable cases in advance, improving system efficiency.
        
        用于提前过滤不适用的情况，提高系统效率。
        
        Examples | 示例:
            - Check if feature values are within valid range | 检查特征值是否在有效范围内
            - Check if required resources are available | 检查所需资源是否可用
            - Check if state meets preconditions | 检查状态是否满足前置条件
        
        Args | 参数:
            features: Current feature vector | 当前特征向量
            
        Returns | 返回:
            bool: Whether execution is possible, default returns True
                 是否可执行，默认返回 True
        """
        return True
    
    def get_complexity(self) -> float:
        """
        Get behavior complexity (optional override).
        
        获取行为复杂度（可选覆盖）。
        
        Used to evaluate resource consumption, influencing scheduling decisions.
        
        用于评估资源消耗，影响调度决策。
        
        Complexity factors | 复杂度因素:
            - Algorithm time complexity | 算法时间复杂度
            - Expected execution time | 预期执行时间
            - Resource usage | 资源使用情况
            - Historical execution data | 历史执行数据
        
        Returns | 返回:
            float: Complexity value, default 1.0 | 复杂度值，默认为 1.0
        """
        return 1.0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get behavior statistics.
        
        获取行为统计信息。
        
        Returns runtime statistics for monitoring and debugging.
        
        返回运行时统计信息，用于监控和调试。
        
        Returns | 返回:
            Dict: Statistics including execution count, timing, etc.
                 包含执行次数、时间等信息的统计字典
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
        
        执行前钩子函数（可选覆盖）。
        
        Can be used for resource preparation, logging, etc.
        Called before execute().
        
        可用于资源准备、日志记录等。
        在 execute() 之前调用。
        
        Args | 参数:
            features: Current feature vector | 当前特征向量
        """
        self._logger.debug(f"Preparing to execute behavior {self.name}")
    
    def after_execute(self, result: R, execution_time: float) -> None:
        """
        Post-execution hook function (optional override).
        
        执行后钩子函数（可选覆盖）。
        
        Can be used for resource cleanup, result logging, etc.
        Called after execute().
        
        可用于资源清理、结果记录等。
        在 execute() 之后调用。
        
        Args | 参数:
            result: Execution result | 执行结果
            execution_time: Execution duration in seconds | 执行时长（秒）
        """
        self._logger.debug(f"Behavior {self.name} completed, duration {execution_time:.3f}s")
    
    def on_error(self, error: Exception) -> None:
        """
        Error handling callback (optional override).
        
        错误处理回调（可选覆盖）。
        
        Can be used for error recovery, alerting, etc.
        Called when execute() raises an exception.
        
        可用于错误恢复、告警等。
        当 execute() 抛出异常时调用。
        
        Args | 参数:
            error: Caught exception | 捕获的异常
        """
        self._last_error = error
        self._logger.error(f"Behavior {self.name} execution failed: {error}")
    
    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._execution_count = 0
        self._last_execution_time = 0.0
        self._total_execution_time = 0.0
        self._last_error = None


# ============================================================
# Export Interface | 导出接口
# ============================================================

__all__ = [
    'MetaBehavior',           # Meta-behavior interface | 元行为接口
    'BehaviorExecutionError', # Behavior execution exception | 行为执行异常
]