"""
Configuration Loader - Load from Various Sources | 配置加载器 - 从多种来源加载
===============================================================================

Provides utilities for loading configuration from different sources:
- Default values | 默认值
- JSON/YAML files | JSON/YAML 文件
- Environment variables | 环境变量
- Dictionaries | 字典

提供从不同来源加载配置的工具：
- 默认值
- JSON/YAML 文件
- 环境变量
- 字典

Design Philosophy | 设计理念:
    - Layered: Defaults < File < Environment < Explicit override
      分层：默认值 < 文件 < 环境变量 < 显式覆盖
    - Flexible: Support multiple file formats | 灵活：支持多种文件格式
    - Validated: Configuration is validated after loading | 验证：加载后验证配置
"""

import json
import os
from typing import Dict, Any, Optional, Union
from pathlib import Path
import logging

from .schema import MetathinConfig, PipelineConfig, MemoryConfig, ObservabilityConfig


# ============================================================
# Environment Variable Mapping | 环境变量映射
# ============================================================

ENV_MAPPING = {
    # Pipeline settings | 流水线设置
    'METATHIN_MIN_FITNESS': ('pipeline', 'min_fitness_threshold', float),
    'METATHIN_ENABLE_LEARNING': ('pipeline', 'enable_learning', lambda x: x.lower() == 'true'),
    'METATHIN_LEARNING_RATE': ('pipeline', 'learning_rate', float),
    'METATHIN_MAX_RETRIES': ('pipeline', 'max_retries', int),
    'METATHIN_RAISE_ON_ERROR': ('pipeline', 'raise_on_error', lambda x: x.lower() == 'true'),
    
    # Memory settings | 记忆设置
    'METATHIN_MEMORY_ENABLED': ('memory', 'enabled', lambda x: x.lower() == 'true'),
    'METATHIN_MEMORY_BACKEND': ('memory', 'backend_type', str),
    'METATHIN_MEMORY_PATH': ('memory', 'backend_path', str),
    'METATHIN_MEMORY_CACHE_SIZE': ('memory', 'cache_size', lambda x: int(x) if x != 'none' else None),
    'METATHIN_MEMORY_TTL': ('memory', 'default_ttl', lambda x: float(x) if x != 'none' else None),
    'METATHIN_MEMORY_AUTO_SAVE': ('memory', 'auto_save', lambda x: x.lower() == 'true'),
    
    # Observability settings | 可观测性设置
    'METATHIN_KEEP_HISTORY': ('observability', 'keep_history', lambda x: x.lower() == 'true'),
    'METATHIN_MAX_HISTORY': ('observability', 'max_history_size', lambda x: int(x) if x != 'none' else None),
    'METATHIN_ENABLE_METRICS': ('observability', 'enable_metrics', lambda x: x.lower() == 'true'),
    'METATHIN_LOG_LEVEL': ('observability', 'log_level', str),
    'METATHIN_LOG_FILE': ('observability', 'log_file', str),
    
    # Agent settings | 代理设置
    'METATHIN_AGENT_NAME': ('agent_name', None, str),
}


# ============================================================
# Configuration Loader | 配置加载器
# ============================================================

class ConfigLoader:
    """
    Configuration loader for Metathin.
    
    Metathin 配置加载器。
    
    Loads configuration from multiple sources with proper layering.
    
    从多个来源加载配置，具有正确的分层。
    
    Load Order (later overrides earlier) | 加载顺序（后加载覆盖先加载）:
        1. Default values | 默认值
        2. File (JSON/YAML) | 文件
        3. Environment variables | 环境变量
        4. Explicit overrides | 显式覆盖
    
    Example | 示例:
        >>> loader = ConfigLoader()
        >>> 
        >>> # Load from file | 从文件加载
        >>> config = loader.load_file("config.json")
        >>> 
        >>> # Load from environment | 从环境变量加载
        >>> config = loader.load_env()
        >>> 
        >>> # Load from all sources | 从所有来源加载
        >>> config = loader.load(
        ...     file_path="config.json",
        ...     load_env=True,
        ...     overrides={'pipeline': {'enable_learning': False}}
        ... )
    """
    
    def __init__(self):
        """Initialize configuration loader."""
        self._logger = logging.getLogger("metathin.config.ConfigLoader")
    
    def load_default(self) -> MetathinConfig:
        """
        Load default configuration.
        
        加载默认配置。
        
        Returns | 返回:
            MetathinConfig: Default configuration | 默认配置
        """
        return MetathinConfig()
    
    def load_file(self, file_path: Union[str, Path]) -> MetathinConfig:
        """
        Load configuration from a file.
        
        从文件加载配置。
        
        Supports JSON and YAML formats (auto-detected by extension).
        
        支持 JSON 和 YAML 格式（根据扩展名自动检测）。
        
        Args | 参数:
            file_path: Path to configuration file | 配置文件路径
            
        Returns | 返回:
            MetathinConfig: Loaded configuration | 加载的配置
            
        Raises | 抛出:
            FileNotFoundError: If file doesn't exist | 文件不存在
            ValueError: If file format is invalid | 文件格式无效
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        # Read file based on extension | 根据扩展名读取文件
        content = path.read_text(encoding='utf-8')
        
        if path.suffix.lower() in ('.yaml', '.yml'):
            try:
                import yaml
                data = yaml.safe_load(content)
            except ImportError:
                raise ImportError("PyYAML is required for YAML config files. Install with: pip install pyyaml")
        elif path.suffix.lower() == '.json':
            data = json.loads(content)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}. Use .json, .yaml, or .yml")
        
        if data is None:
            data = {}
        
        self._logger.info(f"Loaded configuration from {file_path}")
        return MetathinConfig.from_dict(data)
    
    def load_env(self, prefix: str = "METATHIN_") -> MetathinConfig:
        """
        Load configuration from environment variables.
        
        从环境变量加载配置。
        """
        # Start with default
        config = MetathinConfig()
        
        # Helper to update config
        def update_config(current_config, updates):
            pipeline = current_config.pipeline
            memory = current_config.memory
            observability = current_config.observability
            agent_name = current_config.agent_name
            
            for key, value in updates.items():
                if key == 'min_fitness_threshold':
                    pipeline = PipelineConfig(
                        min_fitness_threshold=value,
                        enable_learning=pipeline.enable_learning,
                        learning_rate=pipeline.learning_rate,
                        max_retries=pipeline.max_retries,
                        raise_on_error=pipeline.raise_on_error
                    )
                elif key == 'enable_learning':
                    pipeline = PipelineConfig(
                        min_fitness_threshold=pipeline.min_fitness_threshold,
                        enable_learning=value,
                        learning_rate=pipeline.learning_rate,
                        max_retries=pipeline.max_retries,
                        raise_on_error=pipeline.raise_on_error
                    )
                elif key == 'learning_rate':
                    pipeline = PipelineConfig(
                        min_fitness_threshold=pipeline.min_fitness_threshold,
                        enable_learning=pipeline.enable_learning,
                        learning_rate=value,
                        max_retries=pipeline.max_retries,
                        raise_on_error=pipeline.raise_on_error
                    )
                elif key == 'max_retries':
                    pipeline = PipelineConfig(
                        min_fitness_threshold=pipeline.min_fitness_threshold,
                        enable_learning=pipeline.enable_learning,
                        learning_rate=pipeline.learning_rate,
                        max_retries=value,
                        raise_on_error=pipeline.raise_on_error
                    )
                elif key == 'raise_on_error':
                    pipeline = PipelineConfig(
                        min_fitness_threshold=pipeline.min_fitness_threshold,
                        enable_learning=pipeline.enable_learning,
                        learning_rate=pipeline.learning_rate,
                        max_retries=pipeline.max_retries,
                        raise_on_error=value
                    )
                elif key == 'memory_enabled':
                    memory = MemoryConfig(
                        enabled=value,
                        backend_type=memory.backend_type,
                        backend_path=memory.backend_path,
                        cache_size=memory.cache_size,
                        default_ttl=memory.default_ttl,
                        auto_save=memory.auto_save,
                        cleanup_interval=memory.cleanup_interval
                    )
                elif key == 'memory_backend':
                    memory = MemoryConfig(
                        enabled=memory.enabled,
                        backend_type=value,
                        backend_path=memory.backend_path,
                        cache_size=memory.cache_size,
                        default_ttl=memory.default_ttl,
                        auto_save=memory.auto_save,
                        cleanup_interval=memory.cleanup_interval
                    )
                elif key == 'keep_history':
                    observability = ObservabilityConfig(
                        keep_history=value,
                        max_history_size=observability.max_history_size,
                        keep_successful=observability.keep_successful,
                        keep_failed=observability.keep_failed,
                        enable_metrics=observability.enable_metrics,
                        metrics_window_size=observability.metrics_window_size,
                        enable_time_series=observability.enable_time_series,
                        max_time_series_length=observability.max_time_series_length,
                        log_level=observability.log_level,
                        log_file=observability.log_file
                    )
                elif key == 'log_level':
                    observability = ObservabilityConfig(
                        keep_history=observability.keep_history,
                        max_history_size=observability.max_history_size,
                        keep_successful=observability.keep_successful,
                        keep_failed=observability.keep_failed,
                        enable_metrics=observability.enable_metrics,
                        metrics_window_size=observability.metrics_window_size,
                        enable_time_series=observability.enable_time_series,
                        max_time_series_length=observability.max_time_series_length,
                        log_level=value,
                        log_file=observability.log_file
                    )
                elif key == 'agent_name':
                    agent_name = value
            
            return MetathinConfig(
                pipeline=pipeline,
                memory=memory,
                observability=observability,
                agent_name=agent_name,
                extra=current_config.extra
            )
        
        # Read environment variables
        updates = {}
        
        min_fitness = os.environ.get(f"{prefix}MIN_FITNESS")
        if min_fitness is not None:
            try:
                updates['min_fitness_threshold'] = float(min_fitness)
            except ValueError:
                pass
        
        enable_learning = os.environ.get(f"{prefix}ENABLE_LEARNING")
        if enable_learning is not None:
            updates['enable_learning'] = enable_learning.lower() == 'true'
        
        learning_rate = os.environ.get(f"{prefix}LEARNING_RATE")
        if learning_rate is not None:
            try:
                updates['learning_rate'] = float(learning_rate)
            except ValueError:
                pass
        
        memory_enabled = os.environ.get(f"{prefix}MEMORY_ENABLED")
        if memory_enabled is not None:
            updates['memory_enabled'] = memory_enabled.lower() == 'true'
        
        memory_backend = os.environ.get(f"{prefix}MEMORY_BACKEND")
        if memory_backend is not None:
            updates['memory_backend'] = memory_backend
        
        keep_history = os.environ.get(f"{prefix}KEEP_HISTORY")
        if keep_history is not None:
            updates['keep_history'] = keep_history.lower() == 'true'
        
        log_level = os.environ.get(f"{prefix}LOG_LEVEL")
        if log_level is not None:
            updates['log_level'] = log_level
        
        agent_name = os.environ.get(f"{prefix}AGENT_NAME")
        if agent_name is not None:
            updates['agent_name'] = agent_name
        
        # Apply updates
        if updates:
            config = update_config(config, updates)
        
        self._logger.info("Loaded configuration from environment variables")
        return config
    def load_dict(self, config_dict: Dict[str, Any]) -> MetathinConfig:
        """
        Load configuration from a dictionary.
        
        从字典加载配置。
        
        Args | 参数:
            config_dict: Configuration dictionary | 配置字典
            
        Returns | 返回:
            MetathinConfig: Loaded configuration | 加载的配置
        """
        # Ensure agent_name is a string, not a dict
        if isinstance(config_dict.get('agent_name'), dict):
            # Extract the actual value from the dict
            agent_name_value = config_dict['agent_name']
            if isinstance(agent_name_value, dict) and None in agent_name_value:
                config_dict['agent_name'] = agent_name_value[None]
            elif isinstance(agent_name_value, dict) and len(agent_name_value) > 0:
                config_dict['agent_name'] = list(agent_name_value.values())[0]
            else:
                config_dict['agent_name'] = str(agent_name_value)
        return MetathinConfig.from_dict(config_dict)
    
    def load(
        self,
        file_path: Optional[Union[str, Path]] = None,
        load_env: bool = True,
        env_prefix: str = "METATHIN_",
        overrides: Optional[Dict[str, Any]] = None,
    ) -> MetathinConfig:
        """
        Load configuration from multiple sources.
        
        从多个来源加载配置。
        
        Args | 参数:
            file_path: Optional configuration file path | 可选的配置文件路径
            load_env: Whether to load from environment | 是否从环境变量加载
            env_prefix: Environment variable prefix | 环境变量前缀
            overrides: Explicit overrides dictionary | 显式覆盖字典
            
        Returns | 返回:
            MetathinConfig: Final configuration | 最终配置
        """
        # Start with defaults | 从默认值开始
        config = self.load_default()
        
        # Load from file | 从文件加载
        if file_path:
            try:
                config = self.load_file(file_path)
            except Exception as e:
                self._logger.warning(f"Failed to load config from {file_path}: {e}")
        
        # Load from environment | 从环境变量加载
        if load_env:
            env_config = self.load_env(env_prefix)
            # Merge: file config overrides default, env overrides file
            merged_dict = config.to_dict()
            env_dict = env_config.to_dict()
            self._merge_dict(merged_dict, env_dict)
            config = MetathinConfig.from_dict(merged_dict)
        
        # Apply overrides | 应用覆盖
        if overrides:
            config_dict = config.to_dict()
            self._merge_dict(config_dict, overrides)
            config = MetathinConfig.from_dict(config_dict)
        
        self._logger.info(f"Configuration loaded: agent_name={config.agent_name}")
        return config
    
    def _merge_dict(self, target: Dict, source: Dict) -> None:
        """
        Recursively merge source dictionary into target.
        
        递归地将源字典合并到目标字典。
        
        Args | 参数:
            target: Target dictionary (modified in place) | 目标字典（原地修改）
            source: Source dictionary | 源字典
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._merge_dict(target[key], value)
            else:
                target[key] = value
    
    def save_file(self, config: MetathinConfig, file_path: Union[str, Path]) -> bool:
        """
        Save configuration to a file.
        
        保存配置到文件。
        
        Args | 参数:
            config: Configuration to save | 要保存的配置
            file_path: Output file path | 输出文件路径
            
        Returns | 返回:
            bool: True if successful | 成功返回 True
        """
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            data = config.to_dict()
            
            if path.suffix.lower() in ('.yaml', '.yml'):
                try:
                    import yaml
                    content = yaml.dump(data, default_flow_style=False, allow_unicode=True)
                except ImportError:
                    self._logger.error("PyYAML is required for YAML output")
                    return False
            elif path.suffix.lower() == '.json':
                content = json.dumps(data, indent=2, ensure_ascii=False)
            else:
                # Default to JSON | 默认使用 JSON
                content = json.dumps(data, indent=2, ensure_ascii=False)
            
            path.write_text(content, encoding='utf-8')
            self._logger.info(f"Saved configuration to {file_path}")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to save configuration: {e}")
            return False


# ============================================================
# Convenience Functions | 便捷函数
# ============================================================

def load_config(
    file_path: Optional[Union[str, Path]] = None,
    load_env: bool = True,
    overrides: Optional[Dict[str, Any]] = None,
) -> MetathinConfig:
    """
    Convenience function to load configuration.
    
    加载配置的便捷函数。
    
    Args | 参数:
        file_path: Optional configuration file path | 可选的配置文件路径
        load_env: Whether to load from environment | 是否从环境变量加载
        overrides: Explicit overrides | 显式覆盖
        
    Returns | 返回:
        MetathinConfig: Loaded configuration | 加载的配置
    """
    loader = ConfigLoader()
    return loader.load(file_path=file_path, load_env=load_env, overrides=overrides)


def save_config(config: MetathinConfig, file_path: Union[str, Path]) -> bool:
    """
    Convenience function to save configuration.
    
    保存配置的便捷函数。
    
    Args | 参数:
        config: Configuration to save | 要保存的配置
        file_path: Output file path | 输出文件路径
        
    Returns | 返回:
        bool: True if successful | 成功返回 True
    """
    loader = ConfigLoader()
    return loader.save_file(config, file_path)


# ============================================================
# Export Interface | 导出接口
# ============================================================

__all__ = [
    'ConfigLoader',    # Configuration loader class | 配置加载器类
    'load_config',     # Convenience load function | 便捷加载函数
    'save_config',     # Convenience save function | 便捷保存函数
]