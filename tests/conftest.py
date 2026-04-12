"""
Pytest configuration and shared fixtures for Metathin tests.
Metathin 测试的 Pytest 配置和共享 Fixtures。
"""

import pytest
import numpy as np
from typing import List, Dict, Any

# ============================================================
# Sample Data Fixtures | 示例数据 Fixtures
# ============================================================

@pytest.fixture
def sample_feature_vector() -> np.ndarray:
    """Sample 3D feature vector for testing."""
    return np.array([0.5, 0.2, 0.8], dtype=np.float64)


@pytest.fixture
def sample_features_2d() -> np.ndarray:
    """Sample 2D feature vector."""
    return np.array([1.0, 2.0], dtype=np.float64)


@pytest.fixture
def sample_features_5d() -> np.ndarray:
    """Sample 5D feature vector."""
    return np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)


@pytest.fixture
def sample_fitness_scores() -> List[float]:
    """Sample fitness scores for multiple behaviors."""
    return [0.9, 0.5, 0.2, 0.8]


# ============================================================
# Sample Configuration Fixtures | 示例配置 Fixtures
# ============================================================

@pytest.fixture
def default_config():
    """Default Metathin configuration."""
    from metathin.config import MetathinConfig
    return MetathinConfig()


@pytest.fixture
def minimal_config():
    """Minimal configuration (no memory, no history)."""
    from metathin.config import MetathinConfig, MemoryConfig, ObservabilityConfig, PipelineConfig
    return MetathinConfig(
        pipeline=PipelineConfig(enable_learning=False),
        memory=MemoryConfig(enabled=False),
        observability=ObservabilityConfig(keep_history=False, enable_metrics=False),
    )


@pytest.fixture
def production_config():
    """Production configuration."""
    from metathin.config import MetathinConfig
    return MetathinConfig.create_production("TestAgent")


# ============================================================
# Sample Component Fixtures | 示例组件 Fixtures
# ============================================================

@pytest.fixture
def mock_pattern_space():
    """Mock pattern space for testing."""
    from metathin.core import PatternSpace
    
    class MockPattern(PatternSpace):
        def extract(self, raw_input):
            if isinstance(raw_input, (int, float)):
                return np.array([float(raw_input)], dtype=np.float64)
            return np.array([len(str(raw_input))], dtype=np.float64)
        
        def get_feature_names(self):
            return ['length']
    
    return MockPattern()


@pytest.fixture
def mock_selector():
    """Mock selector that returns fixed fitness."""
    from metathin.core import Selector
    
    class MockSelector(Selector):
        def __init__(self, fixed_fitness: float = 0.5):
            super().__init__()
            self.fixed_fitness = fixed_fitness
        
        def compute_fitness(self, behavior, features):
            return self.fixed_fitness
        
        def get_parameters(self):
            return {'w0': 0.5, 'w1': -0.2}
    
    return MockSelector()


@pytest.fixture
def mock_decision_strategy():
    """Mock decision strategy that selects first behavior."""
    from metathin.core import DecisionStrategy
    
    class MockStrategy(DecisionStrategy):
        def select(self, behaviors, fitness_scores, features):
            return behaviors[0]
    
    return MockStrategy()


# ============================================================
# Sample Behavior Fixtures | 示例行为 Fixtures
# ============================================================

@pytest.fixture
def greet_behavior():
    """Greeting behavior."""
    from metathin.components import FunctionBehavior
    return FunctionBehavior(
        name="greet",
        func=lambda f, **k: f"Hello! (input length: {f[0] if len(f) > 0 else 0})"
    )


@pytest.fixture
def echo_behavior():
    """Echo behavior that returns the input."""
    from metathin.components import FunctionBehavior
    return FunctionBehavior(
        name="echo",
        func=lambda f, **k: k.get('user_input', f"Features: {f}")
    )


@pytest.fixture
def add_behavior():
    """Addition behavior."""
    from metathin.components import FunctionBehavior
    return FunctionBehavior(
        name="add",
        func=lambda f, **k: float(np.sum(f)) if len(f) > 0 else 0.0
    )


@pytest.fixture
def multiply_behavior():
    """Multiplication behavior."""
    from metathin.components import FunctionBehavior
    return FunctionBehavior(
        name="multiply",
        func=lambda f, **k: float(np.prod(f)) if len(f) > 0 else 0.0
    )


@pytest.fixture
def failing_behavior():
    """Behavior that always fails."""
    from metathin.components import FunctionBehavior
    return FunctionBehavior(
        name="failing",
        func=lambda f, **k: exec("raise ValueError('Intentional failure')")
    )


@pytest.fixture
def slow_behavior():
    """Slow behavior that takes time."""
    import time
    from metathin.components import FunctionBehavior
    return FunctionBehavior(
        name="slow",
        func=lambda f, **k: time.sleep(0.05) or "done",
        complexity=5.0
    )


@pytest.fixture
def sample_behaviors(greet_behavior, echo_behavior, add_behavior):
    """List of sample behaviors."""
    return [greet_behavior, echo_behavior, add_behavior]


# ============================================================
# Sample Agent Fixtures | 示例 Agent Fixtures
# ============================================================

@pytest.fixture
def basic_agent(mock_pattern_space, minimal_config):
    """Basic agent without behaviors."""
    from metathin.agent import Metathin
    return Metathin(
        pattern_space=mock_pattern_space,
        config=minimal_config,
        name="TestAgent"
    )


@pytest.fixture
def configured_agent(mock_pattern_space, mock_selector, mock_decision_strategy, 
                     sample_behaviors, minimal_config):
    """Fully configured agent for testing."""
    from metathin.agent import Metathin
    agent = Metathin(
        pattern_space=mock_pattern_space,
        selector=mock_selector,
        decision_strategy=mock_decision_strategy,
        config=minimal_config,
        name="TestAgent"
    )
    for behavior in sample_behaviors:
        agent.register_behavior(behavior)
    return agent


# ============================================================
# Sample Memory Fixtures | 示例记忆 Fixtures
# ============================================================

@pytest.fixture
def in_memory_backend():
    """In-memory backend for testing."""
    from metathin.core.memory_backend import InMemoryBackend
    return InMemoryBackend()


@pytest.fixture
def temp_json_backend(tmp_path):
    """Temporary JSON backend for testing."""
    from metathin.core.memory_backend import JSONMemoryBackend
    return JSONMemoryBackend(tmp_path / "test_memory.json", auto_save=True)


@pytest.fixture
def temp_sqlite_backend(tmp_path):
    """Temporary SQLite backend for testing."""
    from metathin.core.memory_backend import SQLiteMemoryBackend
    return SQLiteMemoryBackend(tmp_path / "test_memory.db", auto_commit=True)


@pytest.fixture
def memory_manager(in_memory_backend):
    """Memory manager with in-memory backend."""
    from metathin.services import MemoryManager
    return MemoryManager(backend=in_memory_backend, cache_size=100)


# ============================================================
# Sample Context Fixtures | 示例上下文 Fixtures
# ============================================================

@pytest.fixture
def thinking_context(sample_feature_vector):
    """Sample thinking context."""
    from metathin.engine import ThinkingContext
    return ThinkingContext(
        raw_input="test_input",
        expected=42.0,
        features=sample_feature_vector,
        selected_behavior="test_behavior",
    )


# ============================================================
# Utility Fixtures | 工具 Fixtures
# ============================================================

@pytest.fixture
def random_seed():
    """Set random seed for reproducible tests."""
    np.random.seed(42)
    import random
    random.seed(42)
    return 42


@pytest.fixture
def temp_dir(tmp_path):
    """Temporary directory for file-based tests."""
    return tmp_path


# ============================================================
# pytest Configuration | pytest 配置
# ============================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks benchmark tests"
    )