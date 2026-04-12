"""
Integration tests for serialization.
序列化集成测试。
"""

import pytest
from pathlib import Path
from metathin.agent import Metathin
from metathin.components import SimplePatternSpace, FunctionBehavior
from metathin.config import MetathinConfig, MemoryConfig, ObservabilityConfig


class TestSerialization:
    """Test save and load functionality."""
    
    def test_save_and_load_agent(self, tmp_path):
        """Test saving and loading agent state."""
        pattern = SimplePatternSpace(lambda x: [len(str(x))])
        
        def greet_func(f, **k):
            return "Hello!"
        
        behavior = FunctionBehavior("greet", greet_func)
        
        original = Metathin(pattern_space=pattern, name="TestAgent")
        original.register_behavior(behavior)
        original.think("test")
        
        filepath = tmp_path / "agent.pkl"
        success = original.save(filepath)
        assert success is True
        
        loaded_behavior = FunctionBehavior("greet", greet_func)
        loaded = Metathin.load(
            filepath,
            pattern_space=pattern,
            behaviors=[loaded_behavior]
        )
        
        assert loaded.name == original.name
        assert len(loaded.B) == 1
    
    def test_save_and_load_with_history(self, tmp_path):
        """Test saving and loading with history."""
        pattern = SimplePatternSpace(lambda x: [len(str(x))])
        
        def echo_func(f, **k):
            return k.get('input', '')
        
        behavior = FunctionBehavior("echo", echo_func)
        
        config = MetathinConfig(
            observability=ObservabilityConfig(keep_history=True)
        )
        
        original = Metathin(pattern_space=pattern, config=config, name="HistoryAgent")
        original.register_behavior(behavior)
        
        original.think("first", input="first")
        original.think("second", input="second")
        original.think("third", input="third")
        
        filepath = tmp_path / "agent_with_history.pkl"
        original.save(filepath)
        
        loaded_behavior = FunctionBehavior("echo", echo_func)
        loaded = Metathin.load(
            filepath,
            pattern_space=pattern,
            behaviors=[loaded_behavior]
        )
        
        assert loaded.name == original.name
    
    def test_save_and_load_with_memory(self, tmp_path):
        """Test saving and loading with memory."""
        pattern = SimplePatternSpace(lambda x: [len(str(x))])
        
        def recall_func(f, **k):
            return None
        
        behavior = FunctionBehavior("recall", recall_func)
        
        config = MetathinConfig(
            memory=MemoryConfig(enabled=True, backend_type='memory')
        )
        
        original = Metathin(pattern_space=pattern, config=config, name="MemoryAgent")
        original.register_behavior(behavior)
        original.remember("key", "value")
        
        filepath = tmp_path / "agent_with_memory.pkl"
        original.save(filepath)
        
        loaded_behavior = FunctionBehavior("recall", recall_func)
        loaded = Metathin.load(
            filepath,
            pattern_space=pattern,
            behaviors=[loaded_behavior]
        )
        
        assert loaded.name == original.name
    
    def test_config_persistence(self, tmp_path):
        """Test that configuration persists through save/load."""
        config = MetathinConfig.create_production("ProdAgent")
        pattern = SimplePatternSpace(lambda x: [len(str(x))])
        
        original = Metathin(pattern_space=pattern, config=config, name="ConfigAgent")
        
        filepath = tmp_path / "config_agent.pkl"
        original.save(filepath)
        
        loaded = Metathin.load(filepath, pattern_space=pattern, behaviors=[])
        
        assert loaded._config.pipeline.enable_learning == config.pipeline.enable_learning
        assert loaded._config.memory.backend_type == config.memory.backend_type
    
    def test_save_creates_directory(self, tmp_path):
        """Test that save creates parent directories."""
        pattern = SimplePatternSpace(lambda x: [len(str(x))])
        agent = Metathin(pattern_space=pattern)
        
        nested_path = tmp_path / "nested" / "deep" / "agent.pkl"
        success = agent.save(nested_path)
        
        assert success is True
        assert nested_path.exists()
    
    def test_load_nonexistent_file_raises(self):
        """Test that load raises error for nonexistent file."""
        pattern = SimplePatternSpace(lambda x: [0])
        
        with pytest.raises(FileNotFoundError):
            Metathin.load("nonexistent.pkl", pattern_space=pattern, behaviors=[])
    
    def test_load_missing_behaviors_raises(self, tmp_path):
        """Test that load raises error when behaviors not provided."""
        pattern = SimplePatternSpace(lambda x: [len(str(x))])
        agent = Metathin(pattern_space=pattern)
        
        filepath = tmp_path / "agent.pkl"
        agent.save(filepath)
        
        with pytest.raises(ValueError, match="behaviors parameter must be provided"):
            Metathin.load(filepath, pattern_space=pattern)