"""
Unit tests for configuration schema.
配置模式单元测试。
"""

import pytest
from metathin.config import (
    PipelineConfig, MemoryConfig, ObservabilityConfig, MetathinConfig
)


class TestPipelineConfig:
    """Test PipelineConfig."""
    
    def test_default_values(self):
        """PipelineConfig should have sensible defaults."""
        config = PipelineConfig()
        
        assert config.min_fitness_threshold == 0.0
        assert config.enable_learning is True
        assert config.learning_rate == 0.01
        assert config.max_retries == 3
        assert config.raise_on_error is True
    
    def test_custom_values(self):
        """PipelineConfig should accept custom values."""
        config = PipelineConfig(
            min_fitness_threshold=0.5,
            enable_learning=False,
            learning_rate=0.001,
            max_retries=5,
            raise_on_error=False
        )
        
        assert config.min_fitness_threshold == 0.5
        assert config.enable_learning is False
        assert config.learning_rate == 0.001
        assert config.max_retries == 5
        assert config.raise_on_error is False
    
    def test_validation_min_fitness_threshold(self):
        """Should validate min_fitness_threshold range."""
        with pytest.raises(ValueError, match="must be in"):
            PipelineConfig(min_fitness_threshold=-0.1)
        
        with pytest.raises(ValueError, match="must be in"):
            PipelineConfig(min_fitness_threshold=1.1)
        
        # Valid values
        PipelineConfig(min_fitness_threshold=0.0)
        PipelineConfig(min_fitness_threshold=0.5)
        PipelineConfig(min_fitness_threshold=1.0)
    
    def test_validation_learning_rate(self):
        """Should validate learning_rate > 0."""
        with pytest.raises(ValueError, match="must be > 0"):
            PipelineConfig(learning_rate=0)
        
        with pytest.raises(ValueError, match="must be > 0"):
            PipelineConfig(learning_rate=-0.1)
        
        # Valid values
        PipelineConfig(learning_rate=0.001)
        PipelineConfig(learning_rate=0.1)
        PipelineConfig(learning_rate=1.0)
    
    def test_validation_max_retries(self):
        """Should validate max_retries >= 0."""
        with pytest.raises(ValueError, match="must be >= 0"):
            PipelineConfig(max_retries=-1)
        
        # Valid values
        PipelineConfig(max_retries=0)
        PipelineConfig(max_retries=3)


class TestMemoryConfig:
    """Test MemoryConfig."""
    
    def test_default_values(self):
        """MemoryConfig should have sensible defaults."""
        config = MemoryConfig()
        
        assert config.enabled is True
        assert config.backend_type == 'json'
        assert config.backend_path is None
        assert config.cache_size == 1000
        assert config.default_ttl is None
        assert config.auto_save is True
        assert config.cleanup_interval == 60.0
    
    def test_custom_values(self):
        """MemoryConfig should accept custom values."""
        config = MemoryConfig(
            enabled=False,
            backend_type='sqlite',
            backend_path='/custom/path.db',
            cache_size=500,
            default_ttl=3600,
            auto_save=False,
            cleanup_interval=120.0
        )
        
        assert config.enabled is False
        assert config.backend_type == 'sqlite'
        assert config.backend_path == '/custom/path.db'
        assert config.cache_size == 500
        assert config.default_ttl == 3600
        assert config.auto_save is False
        assert config.cleanup_interval == 120.0
    
    def test_validation_backend_type(self):
        """Should validate backend_type."""
        with pytest.raises(ValueError, match="must be 'json', 'sqlite', or 'memory'"):
            MemoryConfig(backend_type='invalid')
        
        # Valid values
        MemoryConfig(backend_type='json')
        MemoryConfig(backend_type='sqlite')
        MemoryConfig(backend_type='memory')
    
    def test_validation_cache_size(self):
        """Should validate cache_size > 0 or None."""
        with pytest.raises(ValueError, match="must be > 0"):
            MemoryConfig(cache_size=0)
        
        with pytest.raises(ValueError, match="must be > 0"):
            MemoryConfig(cache_size=-100)
        
        # Valid values
        MemoryConfig(cache_size=100)
        MemoryConfig(cache_size=None)
    
    def test_validation_default_ttl(self):
        """Should validate default_ttl > 0 or None."""
        with pytest.raises(ValueError, match="must be > 0"):
            MemoryConfig(default_ttl=0)
        
        with pytest.raises(ValueError, match="must be > 0"):
            MemoryConfig(default_ttl=-10)
        
        # Valid values
        MemoryConfig(default_ttl=60)
        MemoryConfig(default_ttl=None)
    
    def test_validation_cleanup_interval(self):
        """Should validate cleanup_interval > 0."""
        with pytest.raises(ValueError, match="must be > 0"):
            MemoryConfig(cleanup_interval=0)
        
        with pytest.raises(ValueError, match="must be > 0"):
            MemoryConfig(cleanup_interval=-10)
    
    def test_get_backend_path(self):
        """get_backend_path() should return appropriate path."""
        config = MemoryConfig(backend_type='json')
        path = config.get_backend_path("my_agent")
        assert path == "my_agent.json"
        
        config = MemoryConfig(backend_type='sqlite')
        path = config.get_backend_path("my_agent")
        assert path == "my_agent.db"
        
        config = MemoryConfig(backend_path="/custom/path.json")
        path = config.get_backend_path("ignored")
        assert path == "/custom/path.json"


class TestObservabilityConfig:
    """Test ObservabilityConfig."""
    
    def test_default_values(self):
        """ObservabilityConfig should have sensible defaults."""
        config = ObservabilityConfig()
        
        assert config.keep_history is True
        assert config.max_history_size == 1000
        assert config.keep_successful is True
        assert config.keep_failed is True
        assert config.enable_metrics is False
        assert config.metrics_window_size == 100
        assert config.enable_time_series is True
        assert config.max_time_series_length == 10000
        assert config.log_level == 'INFO'
        assert config.log_file is None
    
    def test_custom_values(self):
        """ObservabilityConfig should accept custom values."""
        config = ObservabilityConfig(
            keep_history=False,
            max_history_size=500,
            keep_successful=False,
            keep_failed=False,
            enable_metrics=True,
            metrics_window_size=200,
            enable_time_series=False,
            max_time_series_length=5000,
            log_level='DEBUG',
            log_file='/var/log/metathin.log'
        )
        
        assert config.keep_history is False
        assert config.max_history_size == 500
        assert config.keep_successful is False
        assert config.keep_failed is False
        assert config.enable_metrics is True
        assert config.metrics_window_size == 200
        assert config.enable_time_series is False
        assert config.max_time_series_length == 5000
        assert config.log_level == 'DEBUG'
        assert config.log_file == '/var/log/metathin.log'
    
    def test_validation_max_history_size(self):
        """Should validate max_history_size > 0 or None."""
        with pytest.raises(ValueError, match="must be > 0"):
            ObservabilityConfig(max_history_size=0)
        
        with pytest.raises(ValueError, match="must be > 0"):
            ObservabilityConfig(max_history_size=-100)
        
        # Valid values
        ObservabilityConfig(max_history_size=100)
        ObservabilityConfig(max_history_size=None)
    
    def test_validation_metrics_window_size(self):
        """Should validate metrics_window_size > 0."""
        with pytest.raises(ValueError, match="must be > 0"):
            ObservabilityConfig(metrics_window_size=0)
        
        with pytest.raises(ValueError, match="must be > 0"):
            ObservabilityConfig(metrics_window_size=-10)
    
    def test_validation_max_time_series_length(self):
        """Should validate max_time_series_length > 0."""
        with pytest.raises(ValueError, match="must be > 0"):
            ObservabilityConfig(max_time_series_length=0)
        
        with pytest.raises(ValueError, match="must be > 0"):
            ObservabilityConfig(max_time_series_length=-10)
    
    def test_validation_log_level(self):
        """Should validate log_level."""
        with pytest.raises(ValueError, match="must be one of"):
            ObservabilityConfig(log_level='INVALID')
        
        # Valid values
        ObservabilityConfig(log_level='DEBUG')
        ObservabilityConfig(log_level='INFO')
        ObservabilityConfig(log_level='WARNING')
        ObservabilityConfig(log_level='ERROR')
    
    def test_get_log_level_int(self):
        """get_log_level_int() should return correct integer."""
        import logging
        
        config = ObservabilityConfig(log_level='DEBUG')
        assert config.get_log_level_int() == logging.DEBUG
        
        config = ObservabilityConfig(log_level='INFO')
        assert config.get_log_level_int() == logging.INFO
        
        config = ObservabilityConfig(log_level='WARNING')
        assert config.get_log_level_int() == logging.WARNING
        
        config = ObservabilityConfig(log_level='ERROR')
        assert config.get_log_level_int() == logging.ERROR


class TestMetathinConfig:
    """Test MetathinConfig."""
    
    def test_default_values(self):
        """MetathinConfig should have sensible defaults."""
        config = MetathinConfig()
        
        assert isinstance(config.pipeline, PipelineConfig)
        assert isinstance(config.memory, MemoryConfig)
        assert isinstance(config.observability, ObservabilityConfig)
        assert config.agent_name == "Metathin"
        assert config.extra == {}
    
    def test_custom_values(self):
        """MetathinConfig should accept custom values."""
        pipeline = PipelineConfig(min_fitness_threshold=0.7)
        memory = MemoryConfig(enabled=False)
        observability = ObservabilityConfig(keep_history=False)
        
        config = MetathinConfig(
            pipeline=pipeline,
            memory=memory,
            observability=observability,
            agent_name="CustomAgent",
            extra={"custom_key": "custom_value"}
        )
        
        assert config.pipeline.min_fitness_threshold == 0.7
        assert config.memory.enabled is False
        assert config.observability.keep_history is False
        assert config.agent_name == "CustomAgent"
        assert config.extra == {"custom_key": "custom_value"}
    
    def test_create_default(self):
        """create_default() should return default config."""
        config = MetathinConfig.create_default("TestAgent")
        
        assert config.agent_name == "TestAgent"
        assert config.pipeline.min_fitness_threshold == 0.0
        assert config.memory.enabled is True
    
    def test_create_minimal(self):
        """create_minimal() should return minimal config."""
        config = MetathinConfig.create_minimal("MinimalAgent")
        
        assert config.agent_name == "MinimalAgent"
        assert config.pipeline.enable_learning is False
        assert config.memory.enabled is False
        assert config.observability.keep_history is False
        assert config.observability.enable_metrics is False
    
    def test_create_production(self):
        """create_production() should return production config."""
        config = MetathinConfig.create_production("ProdAgent")
        
        assert config.agent_name == "ProdAgent"
        assert config.pipeline.enable_learning is True
        assert config.pipeline.raise_on_error is False
        assert config.memory.enabled is True
        assert config.memory.backend_type == 'sqlite'
        assert config.memory.cache_size == 5000
        assert config.observability.keep_history is True
        assert config.observability.max_history_size == 10000
        assert config.observability.enable_metrics is True
        assert config.observability.log_level == 'WARNING'
    
    def test_to_dict(self):
        """to_dict() should convert to dictionary."""
        config = MetathinConfig.create_default("TestAgent")
        d = config.to_dict()
        
        assert 'pipeline' in d
        assert 'memory' in d
        assert 'observability' in d
        assert 'agent_name' in d
        assert d['agent_name'] == "TestAgent"
        assert 'extra' in d
    
    def test_from_dict(self):
        """from_dict() should create config from dictionary."""
        data = {
            'pipeline': {'min_fitness_threshold': 0.8},
            'memory': {'enabled': False},
            'observability': {'keep_history': False},
            'agent_name': 'FromDictAgent',
            'extra': {'test': True}
        }
        
        config = MetathinConfig.from_dict(data)
        
        assert config.pipeline.min_fitness_threshold == 0.8
        assert config.memory.enabled is False
        assert config.observability.keep_history is False
        assert config.agent_name == "FromDictAgent"
        assert config.extra == {'test': True}
    
    def test_round_trip_serialization(self):
        """Config should survive to_dict/from_dict round trip."""
        original = MetathinConfig.create_production("RoundTripAgent")
        
        d = original.to_dict()
        reconstructed = MetathinConfig.from_dict(d)
        
        assert reconstructed.pipeline.min_fitness_threshold == original.pipeline.min_fitness_threshold
        assert reconstructed.memory.backend_type == original.memory.backend_type
        assert reconstructed.observability.log_level == original.observability.log_level
        assert reconstructed.agent_name == original.agent_name