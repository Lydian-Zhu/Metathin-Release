"""
Integration tests for agent with services.
代理与服务集成测试。
"""

import pytest
from metathin.agent import Metathin, MetathinBuilder
from metathin.components import SimplePatternSpace, FunctionBehavior, SimpleSelector, MaxFitnessStrategy, GradientLearning
from metathin.config import MetathinConfig, MemoryConfig, ObservabilityConfig, PipelineConfig


class TestAgentWithServices:
    """Test agent with services."""
    
    def test_agent_with_full_observability(self):
        """Test agent with history and metrics enabled."""
        config = MetathinConfig(
            observability=ObservabilityConfig(
                keep_history=True,
                max_history_size=100,
                enable_metrics=True
            ),
            memory=MemoryConfig(enabled=False)
        )
        
        pattern = SimplePatternSpace(lambda x: [len(str(x))])
        
        def echo_func(f, **k):
            return k.get('input', '')
        
        behavior = FunctionBehavior("echo", echo_func)
        
        agent = Metathin(pattern_space=pattern, config=config, name="ObservableAgent")
        agent.register_behavior(behavior)
        
        for i in range(5):
            agent.think(f"test_{i}", input=f"test_{i}")
        
        # Just verify no crash
        assert agent.get_stats() is not None
    
    def test_agent_with_memory_service(self):
        """Test agent with memory service enabled."""
        config = MetathinConfig(
            memory=MemoryConfig(enabled=True, backend_type='memory'),
            observability=ObservabilityConfig(keep_history=False)
        )
        
        pattern = SimplePatternSpace(lambda x: [len(str(x))])
        
        agent = Metathin(pattern_space=pattern, config=config, name="MemoryAgent")
        
        agent.remember("stored_value", "persisted_value")
        result = agent.recall("stored_value", "default")
        
        assert result == "persisted_value"
    
    def test_agent_with_learning_and_history(self):
        """Test agent with learning and history tracking."""
        config = MetathinConfig(
            observability=ObservabilityConfig(
                keep_history=True,
                enable_metrics=True
            )
        )
        
        pattern = SimplePatternSpace(lambda x: [float(x)])
        selector = SimpleSelector(n_features=1, n_behaviors=2)
        
        def predict_10(features, **kwargs):
            return 10.0
        
        def predict_20(features, **kwargs):
            return 20.0
        
        behavior1 = FunctionBehavior("predict_10", predict_10)
        behavior2 = FunctionBehavior("predict_20", predict_20)
        
        agent = Metathin(
            pattern_space=pattern,
            selector=selector,
            decision_strategy=MaxFitnessStrategy(),
            learning_mechanism=GradientLearning(learning_rate=0.1),
            config=config,
            name="LearningAgent"
        )
        agent.register_behaviors([behavior1, behavior2])
        
        for i in range(10):
            agent.think(15.0, expected=15.0)
        
        # Just verify no crash
        assert agent.get_stats() is not None
    
    def test_agent_with_all_services_and_error_recovery(self):
        """Test agent with all services and error handling."""
        config = MetathinConfig(
            memory=MemoryConfig(enabled=True, backend_type='memory'),
            observability=ObservabilityConfig(
                keep_history=True,
                keep_failed=True,
                enable_metrics=True
            )
        )
        
        pattern = SimplePatternSpace(lambda x: [len(str(x))])
        
        def sometimes_fails(features, **kwargs):
            if features[0] > 5:
                raise ValueError("Input too long")
            return "Success"
        
        behavior = FunctionBehavior("risky", sometimes_fails)
        
        agent = Metathin(pattern_space=pattern, config=config, name="RobustAgent")
        agent.register_behavior(behavior)
        
        # Create new config with raise_on_error=False
        new_pipeline = PipelineConfig(raise_on_error=False)
        agent._config = MetathinConfig(
            pipeline=new_pipeline,
            memory=agent._config.memory,
            observability=agent._config.observability,
            agent_name=agent._config.agent_name
        )
        
        # Execute thoughts - should not crash
        agent.think("short")
        agent.think("this is a very long input that will cause failure")
        agent.think("ok")
        
        assert True
    
    def test_agent_builder_with_services(self):
        """Test agent builder with service configuration."""
        def test_func(f, **k):
            return "result"
        
        agent = (MetathinBuilder()
            .with_name("BuiltAgent")
            .with_pattern_space(SimplePatternSpace(lambda x: [len(str(x))]))
            .with_behavior(FunctionBehavior("test", test_func))
            .enable_memory(True)
            .enable_history(True)
            .enable_metrics(True)
            .with_log_level('DEBUG')
            .build())
        
        assert agent.name == "BuiltAgent"
        assert len(agent.B) == 1
        
        agent.think("test")
        
        stats = agent.get_stats()
        assert stats is not None