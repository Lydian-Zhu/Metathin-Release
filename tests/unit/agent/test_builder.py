"""
Unit tests for MetathinBuilder.
Metathin 构建器单元测试。
"""

import pytest
from metathin.agent import MetathinBuilder, Metathin
from metathin.components import SimplePatternSpace, FunctionBehavior, SimpleSelector
from metathin.config import MetathinConfig, PipelineConfig, MemoryConfig, ObservabilityConfig


class TestMetathinBuilder:
    """Test MetathinBuilder."""
    
    def test_builder_initialization(self):
        """Builder should initialize empty."""
        builder = MetathinBuilder()
        
        assert builder._name is None
        assert builder._pattern_space is None
        assert builder._behaviors == []
        assert builder._selector is None
        assert builder._decision_strategy is None
        assert builder._learning_mechanism is None
    
    def test_with_name(self):
        """with_name() should set agent name."""
        builder = MetathinBuilder()
        builder.with_name("TestAgent")
        
        assert builder._name == "TestAgent"
    
    def test_with_pattern_space(self):
        """with_pattern_space() should set pattern space."""
        pattern = SimplePatternSpace(lambda x: [0])
        builder = MetathinBuilder()
        builder.with_pattern_space(pattern)
        
        assert builder._pattern_space is pattern
    
    def test_with_behavior(self):
        """with_behavior() should add a behavior."""
        behavior = FunctionBehavior("b1", lambda f,**k: "r1")
        builder = MetathinBuilder()
        builder.with_behavior(behavior)
        
        assert len(builder._behaviors) == 1
        assert builder._behaviors[0] is behavior
    
    def test_with_behaviors(self):
        """with_behaviors() should add multiple behaviors."""
        behaviors = [
            FunctionBehavior("b1", lambda f,**k: "r1"),
            FunctionBehavior("b2", lambda f,**k: "r2"),
        ]
        builder = MetathinBuilder()
        builder.with_behaviors(behaviors)
        
        assert len(builder._behaviors) == 2
    
    def test_with_selector(self):
        """with_selector() should set selector."""
        selector = SimpleSelector()
        builder = MetathinBuilder()
        builder.with_selector(selector)
        
        assert builder._selector is selector
    
    def test_with_decision_strategy(self):
        """with_decision_strategy() should set decision strategy."""
        from metathin.components import MaxFitnessStrategy
        strategy = MaxFitnessStrategy()
        builder = MetathinBuilder()
        builder.with_decision_strategy(strategy)
        
        assert builder._decision_strategy is strategy
    
    def test_with_learning_mechanism(self):
        """with_learning_mechanism() should set learning mechanism."""
        from metathin.components import GradientLearning
        learner = GradientLearning()
        builder = MetathinBuilder()
        builder.with_learning_mechanism(learner)
        
        assert builder._learning_mechanism is learner
    
    def test_with_config(self):
        """with_config() should set configuration."""
        config = MetathinConfig.create_production("ProdAgent")
        builder = MetathinBuilder()
        builder.with_config(config)
        
        assert builder._config is config
    
    def test_with_pipeline_config(self):
        """with_pipeline_config() should override pipeline config."""
        pipeline = PipelineConfig(min_fitness_threshold=0.8)
        builder = MetathinBuilder()
        builder.with_pipeline_config(pipeline)
        
        assert 'pipeline' in builder._config_overrides
        assert builder._config_overrides['pipeline'].min_fitness_threshold == 0.8
    
    def test_with_memory_config(self):
        """with_memory_config() should override memory config."""
        memory = MemoryConfig(enabled=False)
        builder = MetathinBuilder()
        builder.with_memory_config(memory)
        
        assert 'memory' in builder._config_overrides
        assert builder._config_overrides['memory'].enabled is False
    
    def test_with_observability_config(self):
        """with_observability_config() should override observability config."""
        obs = ObservabilityConfig(keep_history=False)
        builder = MetathinBuilder()
        builder.with_observability_config(obs)
        
        assert 'observability' in builder._config_overrides
        assert builder._config_overrides['observability'].keep_history is False
    
    def test_enable_memory(self):
        """enable_memory() should set memory enabled flag."""
        builder = MetathinBuilder()
        builder.enable_memory(False)
        
        assert builder._config_overrides['memory'].enabled is False
    
    def test_enable_history(self):
        """enable_history() should set history enabled flag."""
        builder = MetathinBuilder()
        builder.enable_history(True, max_size=500)
        
        obs = builder._config_overrides['observability']
        assert obs.keep_history is True
        assert obs.max_history_size == 500
    
    def test_enable_metrics(self):
        """enable_metrics() should set metrics enabled flag."""
        builder = MetathinBuilder()
        builder.enable_metrics(True)
        
        obs = builder._config_overrides['observability']
        assert obs.enable_metrics is True
    
    def test_with_log_level(self):
        """with_log_level() should set log level."""
        builder = MetathinBuilder()
        builder.with_log_level('DEBUG')
        
        obs = builder._config_overrides['observability']
        assert obs.log_level == 'DEBUG'
    
    def test_with_min_fitness_threshold(self):
        """with_min_fitness_threshold() should set threshold."""
        builder = MetathinBuilder()
        builder.with_min_fitness_threshold(0.75)
        
        pipeline = builder._config_overrides['pipeline']
        assert pipeline.min_fitness_threshold == 0.75
    
    def test_with_learning_rate(self):
        """with_learning_rate() should set learning rate."""
        builder = MetathinBuilder()
        builder.with_learning_rate(0.001)
        
        pipeline = builder._config_overrides['pipeline']
        assert pipeline.learning_rate == 0.001
    
    def test_enable_learning(self):
        """enable_learning() should set learning enabled flag."""
        builder = MetathinBuilder()
        builder.enable_learning(False)
        
        pipeline = builder._config_overrides['pipeline']
        assert pipeline.enable_learning is False
    
    def test_build_minimal_agent(self):
        """Should build agent with minimal configuration."""
        pattern = SimplePatternSpace(lambda x: [0])
        behavior = FunctionBehavior("test", lambda f,**k: "result")
        
        agent = (MetathinBuilder()
            .with_pattern_space(pattern)
            .with_behavior(behavior)
            .build())
        
        assert isinstance(agent, Metathin)
        assert len(agent.B) == 1
        assert agent.B[0].name == "test"
    
    def test_build_without_pattern_space_raises(self):
        """Should raise error when pattern space not provided."""
        builder = MetathinBuilder()
        
        with pytest.raises(ValueError, match="Pattern space.*required"):
            builder.build()
    
    def test_build_with_full_configuration(self):
        """Should build agent with full configuration."""
        pattern = SimplePatternSpace(lambda x: [0])
        behavior = FunctionBehavior("test", lambda f,**k: "result")
        selector = SimpleSelector()
        
        agent = (MetathinBuilder()
            .with_name("FullAgent")
            .with_pattern_space(pattern)
            .with_behavior(behavior)
            .with_selector(selector)
            .with_min_fitness_threshold(0.8)
            .enable_memory(False)
            .enable_history(False)
            .build())
        
        assert agent.name == "FullAgent"
        assert agent.S is selector
        assert agent._config.pipeline.min_fitness_threshold == 0.8
        assert agent._config.memory.enabled is False
        assert agent._config.observability.keep_history is False
    
    def test_build_with_multiple_behaviors(self):
        """Should build agent with multiple behaviors."""
        pattern = SimplePatternSpace(lambda x: [0])
        behaviors = [
            FunctionBehavior("b1", lambda f,**k: "r1"),
            FunctionBehavior("b2", lambda f,**k: "r2"),
            FunctionBehavior("b3", lambda f,**k: "r3"),
        ]
        
        agent = (MetathinBuilder()
            .with_pattern_space(pattern)
            .with_behaviors(behaviors)
            .build())
        
        assert len(agent.B) == 3
    
    def test_create_default_builder(self):
        """create_default() should return builder with defaults."""
        builder = MetathinBuilder.create_default()
        
        assert builder._config_overrides == {}
    
    def test_create_minimal_builder(self):
        """create_minimal() should return builder with minimal settings."""
        builder = MetathinBuilder.create_minimal()
        
        # Should have overrides for memory and history
        assert builder._config_overrides['memory'].enabled is False
        assert builder._config_overrides['observability'].keep_history is False
        assert builder._config_overrides['observability'].enable_metrics is False
    
    def test_create_production_builder(self):
        """create_production() should return builder with production settings."""
        builder = MetathinBuilder.create_production("ProdAgent")
        
        assert builder._name == "ProdAgent"
        assert builder._config_overrides['memory'].enabled is True
        assert builder._config_overrides['memory'].backend_type == 'sqlite'
    
    def test_builder_method_chaining(self):
        """All builder methods should return self for chaining."""
        builder = MetathinBuilder()
        
        result = (builder
            .with_name("ChainTest")
            .with_pattern_space(SimplePatternSpace(lambda x: [0]))
            .with_behavior(FunctionBehavior("b1", lambda f,**k: "r1"))
            .enable_memory(False))
        
        assert result is builder