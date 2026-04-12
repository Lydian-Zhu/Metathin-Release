"""
Metathin Builder - Fluent Construction API | Metathin 构建器 - 流畅构建 API
===========================================================================

Provides a fluent builder pattern for constructing Metathin agents with
explicit dependency injection.

提供流畅的构建器模式，用于通过显式依赖注入构建 Metathin 代理。

Design Philosophy | 设计理念:
    - Fluent interface: Method chaining for readability | 流畅接口：方法链提高可读性
    - Explicit: All dependencies are clearly specified | 显式：所有依赖都清晰指定
    - Flexible: Components can be set in any order | 灵活：组件可以按任意顺序设置
    - Validated: Builder validates before building | 验证：构建前进行验证

Example | 示例:
    >>> from metathin.agent import MetathinBuilder
    >>> from metathin.components import FunctionBehavior, MaxFitnessStrategy
    >>> 
    >>> agent = (MetathinBuilder()
    ...     .with_name("MyAgent")
    ...     .with_pattern_space(MyPattern())
    ...     .with_behavior(greet_behavior)
    ...     .with_behavior(echo_behavior)
    ...     .with_decision_strategy(MaxFitnessStrategy())
    ...     .with_config(MetathinConfig.create_production())
    ...     .build()
    ... )
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from ..core.p_pattern import PatternSpace
from ..core.b_behavior import MetaBehavior
from ..core.s_selector import Selector
from ..core.d_decision import DecisionStrategy
from ..core.psi_learning import LearningMechanism
from ..config import MetathinConfig, PipelineConfig, MemoryConfig, ObservabilityConfig

from .metathin import Metathin


# ============================================================
# Metathin Builder | Metathin 构建器
# ============================================================

class MetathinBuilder:
    """
    Fluent builder for Metathin agents.
    
    Metathin 代理的流畅构建器。
    
    This builder provides a clear and readable way to construct Metathin agents
    with explicit dependency injection.
    
    这个构建器提供了一种清晰可读的方式来构建具有显式依赖注入的 Metathin 代理。
    
    Example | 示例:
        >>> builder = MetathinBuilder()
        >>> builder.with_name("MyAgent")
        >>> builder.with_pattern_space(MyPattern())
        >>> builder.with_behavior(greet_behavior)
        >>> agent = builder.build()
    """
    
    def __init__(self):
        """Initialize builder with default values."""
        self._name: Optional[str] = None
        self._pattern_space: Optional[PatternSpace] = None
        self._behaviors: List[MetaBehavior] = []
        self._selector: Optional[Selector] = None
        self._decision_strategy: Optional[DecisionStrategy] = None
        self._learning_mechanism: Optional[LearningMechanism] = None
        self._config: Optional[MetathinConfig] = None
        
        # Override flags | 覆盖标志
        self._config_overrides: Dict[str, Any] = {}
    
    # ============================================================
    # Component Setters | 组件设置器
    # ============================================================
    
    def with_name(self, name: str) -> 'MetathinBuilder':
        """
        Set the agent name.
        
        设置代理名称。
        
        Args | 参数:
            name: Agent name | 代理名称
            
        Returns | 返回:
            self: For method chaining | 用于方法链
        """
        self._name = name
        return self
    
    def with_pattern_space(self, pattern_space: PatternSpace) -> 'MetathinBuilder':
        """
        Set the pattern space (P).
        
        设置模式空间 (P)。
        
        Args | 参数:
            pattern_space: Pattern space instance | 模式空间实例
            
        Returns | 返回:
            self: For method chaining | 用于方法链
        """
        self._pattern_space = pattern_space
        return self
    
    def with_behavior(self, behavior: MetaBehavior) -> 'MetathinBuilder':
        """
        Add a single behavior (B).
        
        添加单个行为 (B)。
        
        Args | 参数:
            behavior: Behavior instance | 行为实例
            
        Returns | 返回:
            self: For method chaining | 用于方法链
        """
        self._behaviors.append(behavior)
        return self
    
    def with_behaviors(self, behaviors: List[MetaBehavior]) -> 'MetathinBuilder':
        """
        Add multiple behaviors (B).
        
        添加多个行为 (B)。
        
        Args | 参数:
            behaviors: List of behavior instances | 行为实例列表
            
        Returns | 返回:
            self: For method chaining | 用于方法链
        """
        self._behaviors.extend(behaviors)
        return self
    
    def with_selector(self, selector: Selector) -> 'MetathinBuilder':
        """
        Set the selector (S).
        
        设置选择器 (S)。
        
        Args | 参数:
            selector: Selector instance | 选择器实例
            
        Returns | 返回:
            self: For method chaining | 用于方法链
        """
        self._selector = selector
        return self
    
    def with_decision_strategy(self, strategy: DecisionStrategy) -> 'MetathinBuilder':
        """
        Set the decision strategy (D).
        
        设置决策策略 (D)。
        
        Args | 参数:
            strategy: Decision strategy instance | 决策策略实例
            
        Returns | 返回:
            self: For method chaining | 用于方法链
        """
        self._decision_strategy = strategy
        return self
    
    def with_learning_mechanism(self, mechanism: LearningMechanism) -> 'MetathinBuilder':
        """
        Set the learning mechanism (Ψ).
        
        设置学习机制 (Ψ)。
        
        Args | 参数:
            mechanism: Learning mechanism instance | 学习机制实例
            
        Returns | 返回:
            self: For method chaining | 用于方法链
        """
        self._learning_mechanism = mechanism
        return self
    
    # ============================================================
    # Configuration Setters | 配置设置器
    # ============================================================
    
    def with_config(self, config: MetathinConfig) -> 'MetathinBuilder':
        """
        Set the complete configuration.
        
        设置完整配置。
        
        Args | 参数:
            config: Configuration instance | 配置实例
            
        Returns | 返回:
            self: For method chaining | 用于方法链
        """
        self._config = config
        return self
    
    def with_pipeline_config(self, pipeline_config: PipelineConfig) -> 'MetathinBuilder':
        """
        Set pipeline configuration.
        
        设置流水线配置。
        
        Args | 参数:
            pipeline_config: Pipeline configuration | 流水线配置
            
        Returns | 返回:
            self: For method chaining | 用于方法链
        """
        self._config_overrides['pipeline'] = pipeline_config
        return self
    
    def with_memory_config(self, memory_config: MemoryConfig) -> 'MetathinBuilder':
        """
        Set memory configuration.
        
        设置记忆配置。
        
        Args | 参数:
            memory_config: Memory configuration | 记忆配置
            
        Returns | 返回:
            self: For method chaining | 用于方法链
        """
        self._config_overrides['memory'] = memory_config
        return self
    
    def with_observability_config(self, obs_config: ObservabilityConfig) -> 'MetathinBuilder':
        """
        Set observability configuration.
        
        设置可观测性配置。
        
        Args | 参数:
            obs_config: Observability configuration | 可观测性配置
            
        Returns | 返回:
            self: For method chaining | 用于方法链
        """
        self._config_overrides['observability'] = obs_config
        return self
    
    def enable_memory(self, enabled: bool = True) -> 'MetathinBuilder':
        """
        Enable or disable memory.
        
        启用或禁用记忆。
        
        Args | 参数:
            enabled: Whether to enable memory | 是否启用记忆
            
        Returns | 返回:
            self: For method chaining | 用于方法链
        """
        self._config_overrides.setdefault('memory', MemoryConfig())
        # Create new config with updated enabled flag | 创建带有更新 enabled 标志的新配置
        old = self._config_overrides['memory']
        self._config_overrides['memory'] = MemoryConfig(
            enabled=enabled,
            backend_type=old.backend_type,
            backend_path=old.backend_path,
            cache_size=old.cache_size,
            default_ttl=old.default_ttl,
            auto_save=old.auto_save,
            cleanup_interval=old.cleanup_interval,
        )
        return self
    
    def enable_history(self, enabled: bool = True, max_size: Optional[int] = 1000) -> 'MetathinBuilder':
        """
        Enable or disable history tracking.
        
        启用或禁用历史追踪。
        
        Args | 参数:
            enabled: Whether to enable history | 是否启用历史
            max_size: Maximum history size | 最大历史大小
            
        Returns | 返回:
            self: For method chaining | 用于方法链
        """
        self._config_overrides.setdefault('observability', ObservabilityConfig())
        old = self._config_overrides['observability']
        self._config_overrides['observability'] = ObservabilityConfig(
            keep_history=enabled,
            max_history_size=max_size if enabled else None,
            keep_successful=old.keep_successful,
            keep_failed=old.keep_failed,
            enable_metrics=old.enable_metrics,
            metrics_window_size=old.metrics_window_size,
            enable_time_series=old.enable_time_series,
            max_time_series_length=old.max_time_series_length,
            log_level=old.log_level,
            log_file=old.log_file,
        )
        return self
    
    def enable_metrics(self, enabled: bool = True) -> 'MetathinBuilder':
        """
        Enable or disable metrics collection.
        
        启用或禁用指标收集。
        
        Args | 参数:
            enabled: Whether to enable metrics | 是否启用指标
            
        Returns | 返回:
            self: For method chaining | 用于方法链
        """
        self._config_overrides.setdefault('observability', ObservabilityConfig())
        old = self._config_overrides['observability']
        self._config_overrides['observability'] = ObservabilityConfig(
            keep_history=old.keep_history,
            max_history_size=old.max_history_size,
            keep_successful=old.keep_successful,
            keep_failed=old.keep_failed,
            enable_metrics=enabled,
            metrics_window_size=old.metrics_window_size,
            enable_time_series=old.enable_time_series,
            max_time_series_length=old.max_time_series_length,
            log_level=old.log_level,
            log_file=old.log_file,
        )
        return self
    
    def with_log_level(self, level: str) -> 'MetathinBuilder':
        """
        Set log level.
        
        设置日志级别。
        
        Args | 参数:
            level: Log level ('DEBUG', 'INFO', 'WARNING', 'ERROR') | 日志级别
            
        Returns | 返回:
            self: For method chaining | 用于方法链
        """
        self._config_overrides.setdefault('observability', ObservabilityConfig())
        old = self._config_overrides['observability']
        self._config_overrides['observability'] = ObservabilityConfig(
            keep_history=old.keep_history,
            max_history_size=old.max_history_size,
            keep_successful=old.keep_successful,
            keep_failed=old.keep_failed,
            enable_metrics=old.enable_metrics,
            metrics_window_size=old.metrics_window_size,
            enable_time_series=old.enable_time_series,
            max_time_series_length=old.max_time_series_length,
            log_level=level,
            log_file=old.log_file,
        )
        return self
    
    def with_min_fitness_threshold(self, threshold: float) -> 'MetathinBuilder':
        """
        Set minimum fitness threshold.
        
        设置最小适应度阈值。
        
        Args | 参数:
            threshold: Threshold value (0-1) | 阈值 (0-1)
            
        Returns | 返回:
            self: For method chaining | 用于方法链
        """
        self._config_overrides.setdefault('pipeline', PipelineConfig())
        old = self._config_overrides['pipeline']
        self._config_overrides['pipeline'] = PipelineConfig(
            min_fitness_threshold=threshold,
            enable_learning=old.enable_learning,
            learning_rate=old.learning_rate,
            max_retries=old.max_retries,
            raise_on_error=old.raise_on_error,
        )
        return self
    
    def with_learning_rate(self, learning_rate: float) -> 'MetathinBuilder':
        """
        Set learning rate.
        
        设置学习率。
        
        Args | 参数:
            learning_rate: Learning rate (>0) | 学习率 (>0)
            
        Returns | 返回:
            self: For method chaining | 用于方法链
        """
        self._config_overrides.setdefault('pipeline', PipelineConfig())
        old = self._config_overrides['pipeline']
        self._config_overrides['pipeline'] = PipelineConfig(
            min_fitness_threshold=old.min_fitness_threshold,
            enable_learning=old.enable_learning,
            learning_rate=learning_rate,
            max_retries=old.max_retries,
            raise_on_error=old.raise_on_error,
        )
        return self
    
    def enable_learning(self, enabled: bool = True) -> 'MetathinBuilder':
        """
        Enable or disable learning.
        
        启用或禁用学习。
        
        Args | 参数:
            enabled: Whether to enable learning | 是否启用学习
            
        Returns | 返回:
            self: For method chaining | 用于方法链
        """
        self._config_overrides.setdefault('pipeline', PipelineConfig())
        old = self._config_overrides['pipeline']
        self._config_overrides['pipeline'] = PipelineConfig(
            min_fitness_threshold=old.min_fitness_threshold,
            enable_learning=enabled,
            learning_rate=old.learning_rate,
            max_retries=old.max_retries,
            raise_on_error=old.raise_on_error,
        )
        return self
    
    # ============================================================
    # Build Method | 构建方法
    # ============================================================
    
    def build(self) -> Metathin:
        """
        Build the Metathin agent.
        
        构建 Metathin 代理。
        
        Returns | 返回:
            Metathin: Configured agent instance | 配置好的代理实例
            
        Raises | 抛出:
            ValueError: If required components are missing | 缺少必需组件时
        """
        # Validate required components | 验证必需组件
        if self._pattern_space is None:
            raise ValueError("Pattern space (P) is required. Call .with_pattern_space()")
        
        # Build configuration | 构建配置
        config = self._build_config()
        
        # Create agent | 创建代理
        agent = Metathin(
            pattern_space=self._pattern_space,
            selector=self._selector,
            decision_strategy=self._decision_strategy,
            learning_mechanism=self._learning_mechanism,
            config=config,
            name=self._name,
        )
        
        # Register behaviors | 注册行为
        if self._behaviors:
            agent.register_behaviors(self._behaviors)
        
        return agent
    
    def _build_config(self) -> MetathinConfig:
        """
        Build the final configuration.
        
        构建最终配置。
        
        Returns | 返回:
            MetathinConfig: Final configuration | 最终配置
        """
        # Start with provided config or default | 从提供的配置或默认值开始
        if self._config is not None:
            base_config = self._config
        else:
            base_config = MetathinConfig()
        
        # Apply overrides | 应用覆盖
        if not self._config_overrides:
            # Override name if provided | 如果提供了名称则覆盖
            if self._name:
                return MetathinConfig(
                    pipeline=base_config.pipeline,
                    memory=base_config.memory,
                    observability=base_config.observability,
                    agent_name=self._name,
                    extra=base_config.extra,
                )
            return base_config
        
        # Apply pipeline overrides | 应用流水线覆盖
        pipeline = base_config.pipeline
        if 'pipeline' in self._config_overrides:
            pipeline = self._config_overrides['pipeline']
        
        # Apply memory overrides | 应用记忆覆盖
        memory = base_config.memory
        if 'memory' in self._config_overrides:
            memory = self._config_overrides['memory']
        
        # Apply observability overrides | 应用可观测性覆盖
        observability = base_config.observability
        if 'observability' in self._config_overrides:
            observability = self._config_overrides['observability']
        
        # Create final config | 创建最终配置
        return MetathinConfig(
            pipeline=pipeline,
            memory=memory,
            observability=observability,
            agent_name=self._name or base_config.agent_name,
            extra=base_config.extra,
        )
    
    # ============================================================
    # Convenience Factory Methods | 便捷工厂方法
    # ============================================================
    
    @classmethod
    def create_default(cls) -> 'MetathinBuilder':
        """
        Create a builder with default configuration.
        
        创建带有默认配置的构建器。
        
        Returns | 返回:
            MetathinBuilder: Builder instance | 构建器实例
        """
        return cls()
    
    @classmethod
    def create_minimal(cls) -> 'MetathinBuilder':
        """
        Create a builder with minimal configuration (no memory, no history).
        
        创建带有最小配置的构建器（无记忆、无历史）。
        
        Returns | 返回:
            MetathinBuilder: Builder instance | 构建器实例
        """
        builder = cls()
        builder.enable_memory(False)
        builder.enable_history(False)
        builder.enable_metrics(False)
        return builder
    
    @classmethod
    def create_production(cls, name: str = "Metathin") -> 'MetathinBuilder':
        """
        Create a builder with production configuration.
        
        创建带有生产配置的构建器。
        """
        builder = cls()
        builder.with_name(name)
        # 确保 memory 配置被正确设置
        from metathin.config import MemoryConfig
        builder._config_overrides['memory'] = MemoryConfig(
            enabled=True,
            backend_type='sqlite',
            backend_path=f"{name}_memory.db",
            cache_size=5000,
        )
        builder.with_config(MetathinConfig.create_production(name))
        return builder