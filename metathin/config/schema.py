"""
Configuration Schema - Data Structures | 配置模式 - 数据结构
=============================================================

Defines the configuration data structures for the Metathin framework.
Separates configuration into logical domains for better organization.

定义 Metathin 框架的配置数据结构。
将配置按逻辑域分离，以便更好地组织。

Configuration Domains | 配置域:
    - PipelineConfig: Thinking pipeline behavior | 思考流水线行为
    - MemoryConfig: Memory service settings | 记忆服务设置
    - ObservabilityConfig: History and metrics settings | 历史和指标设置
    - MetathinConfig: Combined configuration | 组合配置

Design Philosophy | 设计理念:
    - Domain separation: Each aspect has its own config | 域分离：每个方面有自己的配置
    - Sensible defaults: All fields have default values | 合理的默认值：所有字段都有默认值
    - Validation: Configuration can be validated after creation | 验证：配置创建后可以验证
    - Immutable: Configs are dataclasses (frozen) | 不可变：配置是数据类（冻结）
"""

from dataclasses import dataclass, field
from typing import Optional, Literal, Dict, Any
from pathlib import Path


# ============================================================
# Pipeline Configuration | 流水线配置
# ============================================================

@dataclass(frozen=True)
class PipelineConfig:
    """
    Configuration for the thinking pipeline.
    
    思考流水线配置。
    
    Controls the behavior of the cognitive cycle (P → B → S → D → Ψ).
    
    控制认知循环 (P → B → S → D → Ψ) 的行为。
    
    Attributes | 属性:
        min_fitness_threshold: Minimum fitness to consider a behavior (0-1)
                               考虑行为的最低适应度 (0-1)
        enable_learning: Whether learning is enabled | 是否启用学习
        learning_rate: Default learning rate for learning mechanisms
                       学习机制的默认学习率
        max_retries: Maximum retries for failed behaviors | 失败行为的最大重试次数
        raise_on_error: Whether to raise exceptions on errors | 错误时是否抛出异常
    """
    
    min_fitness_threshold: float = 0.0
    """Minimum fitness to consider a behavior (0-1) | 考虑行为的最低适应度 (0-1)"""
    
    enable_learning: bool = True
    """Whether learning is enabled | 是否启用学习"""
    
    learning_rate: float = 0.01
    """Default learning rate for learning mechanisms | 学习机制的默认学习率"""
    
    max_retries: int = 3
    """Maximum retries for failed behaviors | 失败行为的最大重试次数"""
    
    raise_on_error: bool = True
    """Whether to raise exceptions on errors | 错误时是否抛出异常"""
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not 0 <= self.min_fitness_threshold <= 1:
            raise ValueError(f"min_fitness_threshold must be in [0,1], got {self.min_fitness_threshold}")
        
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")


# ============================================================
# Memory Configuration | 记忆配置
# ============================================================

@dataclass(frozen=True)
class MemoryConfig:
    """
    Configuration for the memory service.
    
    记忆服务配置。
    
    Controls the two-tier caching memory system.
    
    控制二级缓存记忆系统。
    
    Attributes | 属性:
        enabled: Whether memory is enabled | 是否启用记忆
        backend_type: Backend type ('json', 'sqlite', 'memory')
                      后端类型 ('json', 'sqlite', 'memory')
        backend_path: Path for persistent backends | 持久化后端的路径
        cache_size: Maximum cache size (None = unlimited) | 最大缓存大小（None = 无限制）
        default_ttl: Default time-to-live in seconds (None = never expire)
                     默认生存时间（秒），None 表示永不过期
        auto_save: Whether to auto-save changes | 是否自动保存更改
        cleanup_interval: Cleanup interval for expired items (seconds)
                          过期项清理间隔（秒）
    """
    
    enabled: bool = True
    """Whether memory is enabled | 是否启用记忆"""
    
    backend_type: Literal['json', 'sqlite', 'memory'] = 'json'
    """Backend type: 'json', 'sqlite', or 'memory' | 后端类型：'json'、'sqlite' 或 'memory'"""
    
    backend_path: Optional[str] = None
    """Path for persistent backends (JSON/SQLite) | 持久化后端的路径（JSON/SQLite）"""
    
    cache_size: Optional[int] = 1000
    """Maximum cache size, None for unlimited | 最大缓存大小，None 表示无限制"""
    
    default_ttl: Optional[float] = None
    """Default time-to-live in seconds, None = never expire | 默认生存时间（秒），None 表示永不过期"""
    
    auto_save: bool = True
    """Whether to auto-save changes to backend | 是否自动保存更改到后端"""
    
    cleanup_interval: float = 60.0
    """Cleanup interval for expired items (seconds) | 过期项清理间隔（秒）"""
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.backend_type not in ('json', 'sqlite', 'memory'):
            raise ValueError(f"backend_type must be 'json', 'sqlite', or 'memory', got {self.backend_type}")
        
        if self.cache_size is not None and self.cache_size <= 0:
            raise ValueError(f"cache_size must be > 0 or None, got {self.cache_size}")
        
        if self.default_ttl is not None and self.default_ttl <= 0:
            raise ValueError(f"default_ttl must be > 0 or None, got {self.default_ttl}")
        
        if self.cleanup_interval <= 0:
            raise ValueError(f"cleanup_interval must be > 0, got {self.cleanup_interval}")
    
    def get_backend_path(self, default_name: str = "metathin_memory") -> str:
        """
        Get the backend path with default if not specified.
        
        获取后端路径，如果未指定则使用默认值。
        
        Args | 参数:
            default_name: Default filename without extension | 默认文件名（不含扩展名）
            
        Returns | 返回:
            str: Full path with appropriate extension | 带适当扩展名的完整路径
        """
        if self.backend_path:
            return self.backend_path
        
        ext = '.json' if self.backend_type == 'json' else '.db'
        return f"{default_name}{ext}"


# ============================================================
# Observability Configuration | 可观测性配置
# ============================================================

@dataclass(frozen=True)
class ObservabilityConfig:
    """
    Configuration for observability services (history and metrics).
    
    可观测性服务配置（历史和指标）。
    
    Controls history tracking and metrics collection.
    
    控制历史追踪和指标收集。
    
    Attributes | 属性:
        keep_history: Whether to keep thought history | 是否保留思考历史
        max_history_size: Maximum number of thoughts to keep (None = unlimited)
                          最大保留思考数（None = 无限制）
        keep_successful: Whether to keep successful thoughts | 是否保留成功的思考
        keep_failed: Whether to keep failed thoughts | 是否保留失败的思考
        enable_metrics: Whether to collect metrics | 是否收集指标
        metrics_window_size: Rolling window size for metrics | 指标的滚动窗口大小
        enable_time_series: Whether to store time series data | 是否存储时间序列数据
        max_time_series_length: Maximum time series length | 时间序列最大长度
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
                   日志级别
        log_file: Optional log file path | 可选的日志文件路径
    """
    
    # History settings | 历史设置
    keep_history: bool = True
    """Whether to keep thought history | 是否保留思考历史"""
    
    max_history_size: Optional[int] = 1000
    """Maximum number of thoughts to keep (None = unlimited) | 最大保留思考数（None = 无限制）"""
    
    keep_successful: bool = True
    """Whether to keep successful thoughts | 是否保留成功的思考"""
    
    keep_failed: bool = True
    """Whether to keep failed thoughts | 是否保留失败的思考"""
    
    # Metrics settings | 指标设置
    enable_metrics: bool = False
    """Whether to collect metrics | 是否收集指标"""
    
    metrics_window_size: int = 100
    """Rolling window size for metrics | 指标的滚动窗口大小"""
    
    enable_time_series: bool = True
    """Whether to store time series data | 是否存储时间序列数据"""
    
    max_time_series_length: int = 10000
    """Maximum time series length | 时间序列最大长度"""
    
    # Logging settings | 日志设置
    log_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR'] = 'INFO'
    """Logging level | 日志级别"""
    
    log_file: Optional[str] = None
    """Optional log file path | 可选的日志文件路径"""
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_history_size is not None and self.max_history_size <= 0:
            raise ValueError(f"max_history_size must be > 0 or None, got {self.max_history_size}")
        
        if self.metrics_window_size <= 0:
            raise ValueError(f"metrics_window_size must be > 0, got {self.metrics_window_size}")
        
        if self.max_time_series_length <= 0:
            raise ValueError(f"max_time_series_length must be > 0, got {self.max_time_series_length}")
        
        valid_levels = ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        if self.log_level not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}, got {self.log_level}")
    
    def get_log_level_int(self) -> int:
        """
        Get log level as integer for logging module.
        
        获取日志级别作为 logging 模块的整数。
        
        Returns | 返回:
            int: Logging level constant | 日志级别常量
        """
        import logging
        return getattr(logging, self.log_level, logging.INFO)


# ============================================================
# Combined Configuration | 组合配置
# ============================================================

@dataclass(frozen=True)
class MetathinConfig:
    """
    Complete Metathin configuration.
    
    完整的 Metathin 配置。
    
    Combines all configuration domains into a single object.
    
    将所有配置域组合成一个对象。
    
    Attributes | 属性:
        pipeline: Pipeline configuration | 流水线配置
        memory: Memory configuration | 记忆配置
        observability: Observability configuration | 可观测性配置
        agent_name: Name of the agent instance | 代理实例名称
        extra: Additional user-defined configuration | 额外的用户定义配置
    """
    
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    """Pipeline configuration | 流水线配置"""
    
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    """Memory configuration | 记忆配置"""
    
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    """Observability configuration | 可观测性配置"""
    
    agent_name: str = "Metathin"
    """Name of the agent instance | 代理实例名称"""
    
    extra: Dict[str, Any] = field(default_factory=dict)
    """Additional user-defined configuration | 额外的用户定义配置"""
    
    @classmethod
    def create_default(cls, agent_name: str = "Metathin") -> 'MetathinConfig':
        """
        Create a default configuration.
        
        创建默认配置。
        
        Args | 参数:
            agent_name: Name of the agent instance | 代理实例名称
            
        Returns | 返回:
            MetathinConfig: Default configuration | 默认配置
        """
        return cls(agent_name=agent_name)
    
    @classmethod
    def create_minimal(cls, agent_name: str = "Metathin") -> 'MetathinConfig':
        """
        Create a minimal configuration (no memory, no history).
        
        创建最小配置（无记忆、无历史）。
        
        Args | 参数:
            agent_name: Name of the agent instance | 代理实例名称
            
        Returns | 返回:
            MetathinConfig: Minimal configuration | 最小配置
        """
        return cls(
            pipeline=PipelineConfig(enable_learning=False),
            memory=MemoryConfig(enabled=False),
            observability=ObservabilityConfig(keep_history=False, enable_metrics=False),
            agent_name=agent_name,
        )
    
    @classmethod
    def create_production(cls, agent_name: str = "Metathin") -> 'MetathinConfig':
        """
        Create a production configuration (SQLite memory, full observability).
        
        创建生产配置（SQLite 记忆、完整可观测性）。
        
        Args | 参数:
            agent_name: Name of the agent instance | 代理实例名称
            
        Returns | 返回:
            MetathinConfig: Production configuration | 生产配置
        """
        return cls(
            pipeline=PipelineConfig(enable_learning=True, raise_on_error=False),
            memory=MemoryConfig(
                enabled=True,
                backend_type='sqlite',
                backend_path=f"{agent_name}_memory.db",
                cache_size=5000,
            ),
            observability=ObservabilityConfig(
                keep_history=True,
                max_history_size=10000,
                enable_metrics=True,
                log_level='WARNING',
            ),
            agent_name=agent_name,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        将配置转换为字典。
        
        Returns | 返回:
            Dict: Dictionary representation | 字典表示
        """
        return {
            'pipeline': {
                'min_fitness_threshold': self.pipeline.min_fitness_threshold,
                'enable_learning': self.pipeline.enable_learning,
                'learning_rate': self.pipeline.learning_rate,
                'max_retries': self.pipeline.max_retries,
                'raise_on_error': self.pipeline.raise_on_error,
            },
            'memory': {
                'enabled': self.memory.enabled,
                'backend_type': self.memory.backend_type,
                'backend_path': self.memory.backend_path,
                'cache_size': self.memory.cache_size,
                'default_ttl': self.memory.default_ttl,
                'auto_save': self.memory.auto_save,
                'cleanup_interval': self.memory.cleanup_interval,
            },
            'observability': {
                'keep_history': self.observability.keep_history,
                'max_history_size': self.observability.max_history_size,
                'keep_successful': self.observability.keep_successful,
                'keep_failed': self.observability.keep_failed,
                'enable_metrics': self.observability.enable_metrics,
                'metrics_window_size': self.observability.metrics_window_size,
                'enable_time_series': self.observability.enable_time_series,
                'max_time_series_length': self.observability.max_time_series_length,
                'log_level': self.observability.log_level,
                'log_file': self.observability.log_file,
            },
            'agent_name': self.agent_name,
            'extra': self.extra.copy(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetathinConfig':
        """
        Create configuration from dictionary.
        
        从字典创建配置。
        
        Args | 参数:
            data: Dictionary representation | 字典表示
            
        Returns | 返回:
            MetathinConfig: Configuration instance | 配置实例
        """
        pipeline_data = data.get('pipeline', {})
        memory_data = data.get('memory', {})
        observability_data = data.get('observability', {})
        
        return cls(
            pipeline=PipelineConfig(
                min_fitness_threshold=pipeline_data.get('min_fitness_threshold', 0.0),
                enable_learning=pipeline_data.get('enable_learning', True),
                learning_rate=pipeline_data.get('learning_rate', 0.01),
                max_retries=pipeline_data.get('max_retries', 3),
                raise_on_error=pipeline_data.get('raise_on_error', True),
            ),
            memory=MemoryConfig(
                enabled=memory_data.get('enabled', True),
                backend_type=memory_data.get('backend_type', 'json'),
                backend_path=memory_data.get('backend_path'),
                cache_size=memory_data.get('cache_size', 1000),
                default_ttl=memory_data.get('default_ttl'),
                auto_save=memory_data.get('auto_save', True),
                cleanup_interval=memory_data.get('cleanup_interval', 60.0),
            ),
            observability=ObservabilityConfig(
                keep_history=observability_data.get('keep_history', True),
                max_history_size=observability_data.get('max_history_size', 1000),
                keep_successful=observability_data.get('keep_successful', True),
                keep_failed=observability_data.get('keep_failed', True),
                enable_metrics=observability_data.get('enable_metrics', False),
                metrics_window_size=observability_data.get('metrics_window_size', 100),
                enable_time_series=observability_data.get('enable_time_series', True),
                max_time_series_length=observability_data.get('max_time_series_length', 10000),
                log_level=observability_data.get('log_level', 'INFO'),
                log_file=observability_data.get('log_file'),
            ),
            agent_name=data.get('agent_name', 'Metathin'),
            extra=data.get('extra', {}),
        )


# ============================================================
# Export Interface | 导出接口
# ============================================================

__all__ = [
    'PipelineConfig',       # Pipeline configuration | 流水线配置
    'MemoryConfig',         # Memory configuration | 记忆配置
    'ObservabilityConfig',  # Observability configuration | 可观测性配置
    'MetathinConfig',       # Complete configuration | 完整配置
]