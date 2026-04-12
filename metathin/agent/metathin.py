"""
Metathin Agent - Main Facade Class | Metathin 代理 - 主门面类
===============================================================

The main entry point for the Metathin framework. This facade class assembles
all components and services, providing a simple interface for users.

Metathin 框架的主入口点。这个门面类组装所有组件和服务，
为用户提供简单的接口。

Design Philosophy | 设计理念:
    - Facade pattern: Simple interface hiding complexity | 门面模式：简单接口隐藏复杂性
    - Dependency injection: Components are injected, not hardcoded
      依赖注入：组件是注入的，不是硬编码的
    - Lazy initialization: Services are initialized only when needed
      延迟初始化：服务仅在需要时初始化
    - Backward compatible: Maintains compatibility with original API
      向后兼容：保持与原始 API 的兼容性

Example | 示例:
    >>> from metathin.core import SimplePatternSpace
    >>> from metathin.components import FunctionBehavior, MaxFitnessStrategy
    >>> 
    >>> # Create components | 创建组件
    >>> pattern = SimplePatternSpace(lambda x: [len(str(x))])
    >>> greet = FunctionBehavior("greet", lambda f,**k: "Hello!")
    >>> 
    >>> # Create agent | 创建代理
    >>> agent = Metathin(pattern_space=pattern)
    >>> agent.register_behavior(greet)
    >>> 
    >>> # Use agent | 使用代理
    >>> result = agent.think("world")
    >>> print(result)  # "Hello!"
"""

import logging
import uuid
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from ..core.types import FeatureVector
from ..core.p_pattern import PatternSpace
from ..core.b_behavior import MetaBehavior
from ..core.s_selector import Selector
from ..core.d_decision import DecisionStrategy
from ..core.psi_learning import LearningMechanism
from ..core.exceptions import MetathinError, NoBehaviorError

from ..engine import ThinkingPipeline, PipelineResult, ThinkingContext, HookManager
from ..services import MemoryManager, HistoryTracker, MetricsCollector, ThoughtRecord
from ..config import MetathinConfig, load_config

# Import default components (lazy) | 导入默认组件（延迟）
from ..components.pattern_space import SimplePatternSpace
from ..components.selector import SimpleSelector
from ..components.decision import MaxFitnessStrategy
from ..components.learning import GradientLearning


# ============================================================
# Metathin Agent Facade | Metathin 代理门面
# ============================================================

class Metathin:
    """
    Metathin Agent - Main facade class.
    
    Metathin 代理 - 主门面类。
    
    This class provides a simple interface for creating and using cognitive agents.
    It assembles the five components (P, B, S, D, Ψ) and optional services.
    
    这个类为创建和使用认知代理提供了简单的接口。
    它组装五个组件 (P, B, S, D, Ψ) 和可选的服务。
    
    Attributes | 属性:
        name: Agent name for identification | 代理名称，用于标识
        P: Pattern space (perception) | 模式空间（感知）
        B: List of behaviors (action) | 行为列表（行动）
        S: Selector (evaluation) | 选择器（评估）
        D: Decision strategy (decision) | 决策策略（决策）
        Ψ: Learning mechanism (learning) | 学习机制（学习）
    """
    
    def __init__(
        self,
        pattern_space: Optional[PatternSpace] = None,
        selector: Optional[Selector] = None,
        decision_strategy: Optional[DecisionStrategy] = None,
        learning_mechanism: Optional[LearningMechanism] = None,
        config: Optional[Union[MetathinConfig, Dict[str, Any]]] = None,
        name: Optional[str] = None,
        # Backward compatibility | 向后兼容
        memory_backend=None,
    ):
        """
        Initialize a Metathin agent.
        
        初始化 Metathin 代理。
        
        Args | 参数:
            pattern_space: Pattern space for feature extraction (P)
                           用于特征提取的模式空间 (P)
            selector: Selector for fitness computation (S)
                      用于适应度计算的选择器 (S)
            decision_strategy: Decision strategy for behavior selection (D)
                               用于行为选择的决策策略 (D)
            learning_mechanism: Learning mechanism for parameter updates (Ψ)
                                用于参数更新的学习机制 (Ψ)
            config: Configuration object or dictionary | 配置对象或字典
            name: Agent name (auto-generated if None) | 代理名称（为 None 时自动生成）
            memory_backend: Legacy parameter for backward compatibility
                            遗留参数，用于向后兼容
        """
        # ============================================================
        # Configuration | 配置
        # ============================================================
        if config is None:
            self._config = MetathinConfig()
        elif isinstance(config, dict):
            self._config = MetathinConfig.from_dict(config)
        else:
            self._config = config
        
        # Override name if provided | 如果提供了名称则覆盖
        if name:
            self._config = MetathinConfig(
                pipeline=self._config.pipeline,
                memory=self._config.memory,
                observability=self._config.observability,
                agent_name=name,
                extra=self._config.extra,
            )
        
        self.name = self._config.agent_name
        
        # ============================================================
        # Components (P, B, S, D, Ψ) | 组件
        # ============================================================
        
        # P - Pattern Space | 模式空间
        if pattern_space is None:
            # Default pattern space (extracts single feature: length) | 默认模式空间
            pattern_space = SimplePatternSpace(lambda x: [float(len(str(x)))])
        self.P: PatternSpace = pattern_space
        
        # B - Behaviors | 行为列表
        self.B: List[MetaBehavior] = []
        
        # S - Selector | 选择器
        if selector is None:
            # Default selector (simple linear) | 默认选择器（简单线性）
            selector = SimpleSelector()
        self.S: Selector = selector
        
        # D - Decision Strategy | 决策策略
        if decision_strategy is None:
            decision_strategy = MaxFitnessStrategy()
        self.D: DecisionStrategy = decision_strategy
        
        # Ψ - Learning Mechanism | 学习机制
        self.Ψ: Optional[LearningMechanism] = learning_mechanism
        if learning_mechanism is None and self._config.pipeline.enable_learning:
            self.Ψ = GradientLearning(learning_rate=self._config.pipeline.learning_rate)
        
        # ============================================================
        # Services (Lazy Initialization) | 服务（延迟初始化）
        # ============================================================
        self._memory: Optional[MemoryManager] = None
        self._history: Optional[HistoryTracker] = None
        self._metrics: Optional[MetricsCollector] = None
        self._hook_manager: Optional[HookManager] = None
        
        # Legacy memory backend support | 遗留记忆后端支持
        if memory_backend is not None and self._config.memory.enabled:
            from ..core.memory_backend import MemoryBackend
            if isinstance(memory_backend, MemoryBackend):
                self._memory = MemoryManager(
                    backend=memory_backend,
                    cache_size=self._config.memory.cache_size,
                    default_ttl=self._config.memory.default_ttl,
                )
        
        # ============================================================
        # Pipeline | 流水线
        # ============================================================
        self._pipeline = ThinkingPipeline({
            'min_fitness_threshold': self._config.pipeline.min_fitness_threshold,
            'enable_learning': self._config.pipeline.enable_learning,
        })
        
        # ============================================================
        # Statistics and State | 统计和状态
        # ============================================================
        self._thinking_count: int = 0
        self._error_count: int = 0
        self._is_thinking: bool = False
        
        # Logger | 日志记录器
        self._logger = self._setup_logger()
        
        self._logger.info(f"✅ Metathin '{self.name}' initialized successfully")
        self._log_component_info()
    
    # ============================================================
    # Initialization Helpers | 初始化辅助方法
    # ============================================================
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for the agent."""
        logger = logging.getLogger(f"metathin.agent.{self.name}")
        
        # Avoid duplicate handlers | 避免重复处理器
        if logger.handlers:
            return logger
        
        # Set log level from config | 从配置设置日志级别
        log_level = self._config.observability.get_log_level_int()
        logger.setLevel(log_level)
        
        # Console handler | 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(console_handler)
        
        # File handler (if configured) | 文件处理器（如果配置了）
        if self._config.observability.log_file:
            try:
                file_handler = logging.FileHandler(
                    self._config.observability.log_file,
                    encoding='utf-8'
                )
                file_handler.setFormatter(logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                ))
                logger.addHandler(file_handler)
            except Exception as e:
                print(f"Cannot create log file: {e}")
        
        return logger
    
    def _log_component_info(self) -> None:
        """Log component information for debugging."""
        self._logger.debug(f"  P={self.P.__class__.__name__}")
        self._logger.debug(f"  S={self.S.__class__.__name__}")
        self._logger.debug(f"  D={self.D.__class__.__name__}")
        self._logger.debug(f"  Ψ={self.Ψ.__class__.__name__ if self.Ψ else 'None'}")
        self._logger.debug(f"  Behaviors: {len(self.B)} registered")
    
    def _ensure_services(self) -> None:
        """Ensure services are initialized (lazy initialization)."""
        # Memory service | 记忆服务
        if self._memory is None and self._config.memory.enabled:
            from ..core.memory_backend import JSONMemoryBackend, SQLiteMemoryBackend, InMemoryBackend
            
            backend_path = self._config.memory.get_backend_path(self.name)
            
            if self._config.memory.backend_type == 'json':
                backend = JSONMemoryBackend(backend_path)
            elif self._config.memory.backend_type == 'sqlite':
                backend = SQLiteMemoryBackend(backend_path)
            else:
                backend = InMemoryBackend()
            
            self._memory = MemoryManager(
                backend=backend,
                cache_size=self._config.memory.cache_size,
                enable_cache=True,
                default_ttl=self._config.memory.default_ttl,
                cleanup_interval=self._config.memory.cleanup_interval,
            )
        
        # History service | 历史服务
        # 注意：即使 keep_history 为 False，也可能需要创建（用于测试）
        if self._history is None:
            if self._config.observability.keep_history:
                from ..services import HistoryTracker
                self._history = HistoryTracker(
                    max_size=self._config.observability.max_history_size,
                    keep_successful=self._config.observability.keep_successful,
                    keep_failed=self._config.observability.keep_failed,
                )
                self._logger.debug("HistoryTracker initialized")
        
        # Metrics service | 指标服务
        if self._metrics is None and self._config.observability.enable_metrics:
            from ..services import MetricsCollector
            self._metrics = MetricsCollector(
                window_size=self._config.observability.metrics_window_size,
                enable_time_series=self._config.observability.enable_time_series,
                max_time_series_length=self._config.observability.max_time_series_length,
            )
    
    # ============================================================
    # Core API | 核心 API
    # ============================================================
    
    def think(self, raw_input: Any, expected: Any = None, **context) -> Any:
        """
        Execute a complete thinking cycle.
        
        执行完整的思考周期。
        
        Flow | 流程:
            1. Perceive (P): Extract features | 提取特征
            2. Hypothesize (S): Compute fitness | 计算适应度
            3. Decide (D): Select behavior | 选择行为
            4. Execute (B): Run behavior | 执行行为
            5. Learn (Ψ): Update parameters | 更新参数
        
        Args | 参数:
            raw_input: Raw input data to process | 要处理的原始输入数据
            expected: Expected result (for learning) | 期望结果（用于学习）
            **context: Additional context parameters | 额外的上下文参数
            
        Returns | 返回:
            Any: Result of behavior execution | 行为执行结果
            
        Raises | 抛出:
            NoBehaviorError: When no behaviors are registered | 没有注册行为时
            MetathinError: When other errors occur and raise_on_error is True
                           发生其他错误且 raise_on_error 为 True 时
        """
        # Prevent reentrancy | 防止重入
        if self._is_thinking:
            raise MetathinError("Agent is already thinking, cannot re-enter")
        
        self._is_thinking = True
        self._thinking_count += 1
        
        # Ensure services are initialized | 确保服务已初始化
        self._ensure_services()
        
        try:
            # Run pipeline | 运行流水线
            result = self._pipeline.run(
                raw_input=raw_input,
                pattern_space=self.P,
                behaviors=self.B,
                selector=self.S,
                decision_strategy=self.D,
                learning_mechanism=self.Ψ,
                expected=expected,
                context_params=context,
            )
            
            # Record to services | 记录到服务
            if result.success:
                self._logger.debug(f"✅ Thinking completed: output={result.result}")
            else:
                self._error_count += 1
                self._logger.error(f"❌ Thinking failed: {result.error}")
            
            # Record to history | 记录到历史
            if self._history:
                self._history.record_from_context(result.context)
            
            # Record to metrics | 记录到指标
            if self._metrics:
                self._metrics.record_from_context(result.context)
            
            # Handle error | 处理错误
            if not result.success and self._config.pipeline.raise_on_error:
                raise result.error or MetathinError("Unknown error")
            
            return result.result
            
        finally:
            self._is_thinking = False
    
    def __call__(self, raw_input: Any, **kwargs) -> Any:
        """Make instance callable, directly invoking think method."""
        return self.think(raw_input, **kwargs)
    
    # ============================================================
    # Behavior Management | 行为管理
    # ============================================================
    
    def register_behavior(self, behavior: MetaBehavior) -> 'Metathin':
        """
        Register a single behavior.
        
        注册单个行为。
        
        Args | 参数:
            behavior: Behavior instance to register | 要注册的行为实例
            
        Returns | 返回:
            self: Supports method chaining | 支持方法链
            
        Raises | 抛出:
            TypeError: When behavior type is incorrect | 行为类型错误时
            ValueError: When behavior name already exists | 行为名称已存在时
        """
        if not isinstance(behavior, MetaBehavior):
            raise TypeError(f"Behavior must inherit from MetaBehavior, got {type(behavior)}")
        
        if not behavior.name or not isinstance(behavior.name, str):
            raise ValueError("Behavior must have a valid name")
        
        if any(b.name == behavior.name for b in self.B):
            raise ValueError(f"Behavior name '{behavior.name}' already exists")
        
        self.B.append(behavior)
        self._logger.info(f"✅ Registered behavior: {behavior.name}")
        
        return self
    
    def register_behaviors(self, behaviors: List[MetaBehavior]) -> 'Metathin':
        """
        Batch register behaviors.
        
        批量注册行为。
        
        Args | 参数:
            behaviors: List of behaviors to register | 要注册的行为列表
            
        Returns | 返回:
            self: Supports method chaining | 支持方法链
        """
        for behavior in behaviors:
            self.register_behavior(behavior)
        
        self._logger.info(f"✅ Batch registered {len(behaviors)} behaviors")
        return self
    
    def unregister_behavior(self, behavior_name: str) -> bool:
        """
        Unregister a behavior by name.
        
        按名称注销行为。
        
        Args | 参数:
            behavior_name: Name of the behavior to unregister | 要注销的行为名称
            
        Returns | 返回:
            bool: True if successful, False if not found | 成功返回 True，未找到返回 False
        """
        for i, behavior in enumerate(self.B):
            if behavior.name == behavior_name:
                self.B.pop(i)
                self._logger.info(f"🗑️ Unregistered behavior: {behavior_name}")
                return True
        
        self._logger.warning(f"Behavior not found: {behavior_name}")
        return False
    
    def get_behavior(self, behavior_name: str) -> Optional[MetaBehavior]:
        """
        Get a behavior by name.
        
        根据名称获取行为。
        
        Args | 参数:
            behavior_name: Name of the behavior | 行为名称
            
        Returns | 返回:
            Optional[MetaBehavior]: The behavior, or None if not found
                                     行为实例，未找到时返回 None
        """
        for behavior in self.B:
            if behavior.name == behavior_name:
                return behavior
        return None
    
    def list_behaviors(self) -> List[Dict[str, Any]]:
        """
        List all registered behaviors with their stats.
        
        列出所有已注册的行为及其统计信息。
        
        Returns | 返回:
            List[Dict]: List of behavior information | 行为信息列表
        """
        return [b.get_stats() for b in self.B]
    
    # ============================================================
    # Memory API | 记忆 API
    # ============================================================
    
    def remember(self, key: str, value: Any, permanent: bool = True) -> bool:
        """
        Store information in memory.
        
        在记忆中存储信息。
        
        Args | 参数:
            key: Memory key | 记忆键
            value: Memory value | 记忆值
            permanent: Whether to persist to backend | 是否持久化到后端
            
        Returns | 返回:
            bool: True if successful | 成功返回 True
        """
        self._ensure_services()
        if self._memory is None:
            self._logger.warning("Memory not enabled")
            return False
        
        return self._memory.remember(key, value, permanent)
    
    def recall(self, key: str, default: Any = None) -> Any:
        """
        Retrieve information from memory.
        
        从记忆中检索信息。
        
        Args | 参数:
            key: Memory key | 记忆键
            default: Default value if not found | 未找到时的默认值
            
        Returns | 返回:
            Any: Stored value or default | 存储的值或默认值
        """
        self._ensure_services()
        if self._memory is None:
            self._logger.warning("Memory not enabled")
            return default
        
        return self._memory.recall(key, default)
    
    def forget(self, key: str, permanent: bool = True) -> bool:
        """
        Delete information from memory.
        
        从记忆中删除信息。
        
        Args | 参数:
            key: Memory key | 记忆键
            permanent: Whether to delete from backend | 是否从后端删除
            
        Returns | 返回:
            bool: True if successful | 成功返回 True
        """
        self._ensure_services()
        if self._memory is None:
            self._logger.warning("Memory not enabled")
            return False
        
        return self._memory.forget(key, permanent)
    
    def clear_memory(self, permanent: bool = True) -> bool:
        """
        Clear all memory.
        
        清空所有记忆。
        
        Args | 参数:
            permanent: Whether to clear backend as well | 是否也清空后端
            
        Returns | 返回:
            bool: True if successful | 成功返回 True
        """
        self._ensure_services()
        if self._memory is None:
            self._logger.warning("Memory not enabled")
            return False
        
        return self._memory.clear(permanent)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics.
        
        获取记忆统计信息。
        
        Returns | 返回:
            Dict: Memory statistics | 记忆统计信息
        """
        self._ensure_services()
        if self._memory is None:
            return {'enabled': False}
        
        return self._memory.get_stats()
    
    # ============================================================
    # History API | 历史 API
    # ============================================================
    
    def get_history(
        self,
        limit: Optional[int] = None,
        success_only: bool = False,
        failure_only: bool = False,
    ) -> List[ThoughtRecord]:
        """
        Get thought history.
        
        获取思考历史。
        
        Args | 参数:
            limit: Maximum number of records | 最大记录数
            success_only: Only successful thoughts | 仅成功的思考
            failure_only: Only failed thoughts | 仅失败的思考
            
        Returns | 返回:
            List[ThoughtRecord]: Thought history | 思考历史
        """
        self._ensure_services()
        if self._history is None:
            return []
        
        if success_only:
            records = self._history.get_successful(limit)
        elif failure_only:
            records = self._history.get_failed(limit)
        else:
            records = self._history.get_recent(limit) if limit else list(self._history)
        
        return records
    
    def get_last_thought(self) -> Optional[ThoughtRecord]:
        """
        Get the last thought record.
        
        获取最后一次思考记录。
        
        Returns | 返回:
            Optional[ThoughtRecord]: Last thought, or None | 最后一次思考，或 None
        """
        self._ensure_services()
        if self._history is None:
            return None
        
        return self._history.get_last()
    
    def clear_history(self) -> None:
        """Clear thought history."""
        self._ensure_services()
        if self._history:
            self._history.clear()
            self._logger.info("History cleared")
    
    def export_history(self, filepath: str, format: str = 'json') -> bool:
        """
        Export thought history to file.
        
        导出思考历史到文件。
        
        Args | 参数:
            filepath: Output file path | 输出文件路径
            format: Export format ('json' or 'pickle') | 导出格式
            
        Returns | 返回:
            bool: True if successful | 成功返回 True
        """
        self._ensure_services()
        if self._history is None:
            self._logger.warning("History not enabled")
            return False
        
        if format == 'json':
            return self._history.export_json(filepath)
        elif format == 'pickle':
            return self._history.export_pickle(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    # ============================================================
    # Statistics API | 统计 API
    # ============================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get agent statistics.
        
        获取代理统计信息。
        
        Returns | 返回:
            Dict: Statistics including counts, rates, etc.
                  包含计数、比率等的统计字典
        """
        self._ensure_services()
        
        stats = {
            'name': self.name,
            'total_thoughts': self._thinking_count,
            'error_count': self._error_count,
            'is_thinking': self._is_thinking,
            'behaviors_count': len(self.B),
            'behaviors': self.list_behaviors(),
        }
        
        # Add history stats | 添加历史统计
        if self._history:
            stats['history_stats'] = self._history.get_stats()
        
        # Add metrics | 添加指标
        if self._metrics:
            stats['metrics'] = self._metrics.get_summary()
        
        # Add memory stats | 添加记忆统计
        if self._memory:
            stats['memory'] = self._memory.get_stats()
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._thinking_count = 0
        self._error_count = 0
        
        if self._metrics:
            self._metrics.reset()
        
        self._logger.info("Statistics reset")
    
    def reset(self) -> None:
        """Completely reset the agent (clear history, reset stats)."""
        self.reset_stats()
        self.clear_history()
        
        if self._memory:
            self._memory.clear()
        
        self._logger.info("Agent fully reset")
    
    # ============================================================
    # Serialization | 序列化
    # ============================================================
    
    def save(self, path: Union[str, Path]) -> bool:
        """
        Save agent state to file.
        
        保存代理状态到文件。
        
        Note: Only configuration and statistics are saved.
        Components and behaviors must be recreated when loading.
        
        注意：仅保存配置和统计信息。
        加载时需要重新创建组件和行为。
        
        Args | 参数:
            path: Save path | 保存路径
            
        Returns | 返回:
            bool: True if successful | 成功返回 True
        """
        import pickle
        
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'config': self._config.to_dict(),
                'stats': {
                    'thinking_count': self._thinking_count,
                    'error_count': self._error_count,
                },
                'behaviors': [{'name': b.name, 'type': type(b).__name__} for b in self.B],
            }
            
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            
            self._logger.info(f"💾 Saved to: {path}")
            return True
            
        except Exception as e:
            self._logger.error(f"Save failed: {e}")
            return False
    
    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        pattern_space: PatternSpace,
        behaviors: Optional[List[MetaBehavior]] = None,
        selector: Optional[Selector] = None,
        decision_strategy: Optional[DecisionStrategy] = None,
        learning_mechanism: Optional[LearningMechanism] = None,
    ) -> 'Metathin':
        """
        Load agent state from file.
        
        从文件加载代理状态。
        
        Args | 参数:
            path: File path | 文件路径
            pattern_space: Pattern space (required) | 模式空间（必需）
            behaviors: List of behaviors (required) | 行为列表（必需）
            selector: Selector (optional) | 选择器（可选）
            decision_strategy: Decision strategy (optional) | 决策策略（可选）
            learning_mechanism: Learning mechanism (optional) | 学习机制（可选）
            
        Returns | 返回:
            Metathin: Loaded agent instance | 加载的代理实例
            
        Raises | 抛出:
            FileNotFoundError: If file doesn't exist | 文件不存在
            ValueError: If required parameters missing | 缺少必需参数
        """
        import pickle
        
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        if behaviors is None:
            raise ValueError("behaviors parameter must be provided")
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        # Create agent | 创建代理
        agent = cls(
            pattern_space=pattern_space,
            selector=selector,
            decision_strategy=decision_strategy,
            learning_mechanism=learning_mechanism,
            config=data.get('config'),
            name=data.get('config', {}).get('agent_name'),
        )
        
        # Register behaviors | 注册行为
        for behavior in behaviors:
            agent.register_behavior(behavior)
        
        # Restore stats | 恢复统计
        stats = data.get('stats', {})
        agent._thinking_count = stats.get('thinking_count', 0)
        agent._error_count = stats.get('error_count', 0)
        
        agent._logger.info(f"📂 Loaded from: {path}")
        return agent
    
    # ============================================================
    # Magic Methods | 魔术方法
    # ============================================================
    
    def __repr__(self) -> str:
        return (
            f"Metathin(name='{self.name}', "
            f"behaviors={len(self.B)}, "
            f"thoughts={self._thinking_count}, "
            f"memory={'✓' if self._memory else '✗'})"
        )
    
    def __str__(self) -> str:
        return self.__repr__()
    
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
# Export Interface | 导出接口
# ============================================================

__all__ = [
    'Metathin',  # Main agent facade | 主代理门面
]