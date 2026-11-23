"""
Integration tests for agent save/load.

Components are supplied explicitly on load; only configuration and lightweight
statistics are persisted.
"""

import pytest

from metathin.agent import Metathin
from metathin.config import MemoryConfig, MetathinConfig, ObservabilityConfig
from tests.helpers import (
    ConstantLearning,
    FirstBehaviorStrategy,
    FixedSelector,
    FunctionBehavior,
    ListPatternSpace,
)


def make_selector(*names):
    return FixedSelector({name: 0.5 for name in names})


class TestSerialization:
    """Test save and load functionality."""

    def test_save_and_load_agent(self, tmp_path):
        pattern = ListPatternSpace(lambda x: [len(str(x))])
        behavior = FunctionBehavior("greet", lambda f, **k: "Hello!")

        original = Metathin(
            pattern_space=pattern,
            selector=make_selector("greet"),
            decision_strategy=FirstBehaviorStrategy(),
            config=MetathinConfig.create_minimal(),
            name="TestAgent",
        )
        original.register_behavior(behavior)
        original.think("test")

        filepath = tmp_path / "agent.pkl"
        assert original.save(filepath) is True

        loaded = Metathin.load(
            filepath,
            pattern_space=pattern,
            behaviors=[FunctionBehavior("greet", lambda f, **k: "Hello!")],
            selector=make_selector("greet"),
            decision_strategy=FirstBehaviorStrategy(),
        )

        assert loaded.name == original.name
        assert len(loaded.B) == 1

    def test_save_and_load_with_history(self, tmp_path):
        pattern = ListPatternSpace(lambda x: [len(str(x))])
        behavior = FunctionBehavior("echo", lambda f, **k: k.get("input", ""))

        config = MetathinConfig(
            observability=ObservabilityConfig(keep_history=True),
            pipeline=MetathinConfig.create_minimal().pipeline,
        )
        original = Metathin(
            pattern_space=pattern,
            selector=make_selector("echo"),
            decision_strategy=FirstBehaviorStrategy(),
            config=config,
            name="HistoryAgent",
        )
        original.register_behavior(behavior)

        original.think("first", input="first")
        original.think("second", input="second")
        original.think("third", input="third")

        filepath = tmp_path / "agent_with_history.pkl"
        original.save(filepath)

        loaded = Metathin.load(
            filepath,
            pattern_space=pattern,
            behaviors=[FunctionBehavior("echo", lambda f, **k: k.get("input", ""))],
            selector=make_selector("echo"),
            decision_strategy=FirstBehaviorStrategy(),
        )
        assert loaded.name == original.name

    def test_save_and_load_with_memory(self, tmp_path):
        pattern = ListPatternSpace(lambda x: [len(str(x))])
        behavior = FunctionBehavior("recall", lambda f, **k: None)

        config = MetathinConfig(
            memory=MemoryConfig(enabled=True, backend_type="memory"),
            pipeline=MetathinConfig.create_minimal().pipeline,
        )
        original = Metathin(
            pattern_space=pattern,
            selector=make_selector("recall"),
            decision_strategy=FirstBehaviorStrategy(),
            config=config,
            name="MemoryAgent",
        )
        original.register_behavior(behavior)
        assert original.remember("key", "value") is True

        filepath = tmp_path / "agent_with_memory.pkl"
        original.save(filepath)

        loaded = Metathin.load(
            filepath,
            pattern_space=pattern,
            behaviors=[FunctionBehavior("recall", lambda f, **k: None)],
            selector=make_selector("recall"),
            decision_strategy=FirstBehaviorStrategy(),
        )
        assert loaded.name == original.name

    def test_config_persistence(self, tmp_path):
        config = MetathinConfig.create_production("ProdAgent")
        pattern = ListPatternSpace(lambda x: [len(str(x))])

        original = Metathin(
            pattern_space=pattern,
            selector=FixedSelector(),
            decision_strategy=FirstBehaviorStrategy(),
            config=config,
            name="ConfigAgent",
            learning_mechanism=ConstantLearning({"w_0": 0.0}),
        )

        filepath = tmp_path / "config_agent.pkl"
        original.save(filepath)

        loaded = Metathin.load(
            filepath,
            pattern_space=pattern,
            behaviors=[],
            selector=FixedSelector(),
            decision_strategy=FirstBehaviorStrategy(),
            learning_mechanism=ConstantLearning({"w_0": 0.0}),
        )

        assert loaded._config.pipeline.enable_learning == config.pipeline.enable_learning
        assert loaded._config.memory.backend_type == config.memory.backend_type

    def test_save_creates_directory(self, tmp_path):
        pattern = ListPatternSpace(lambda x: [len(str(x))])
        agent = Metathin(
            pattern_space=pattern,
            selector=FixedSelector(),
            decision_strategy=FirstBehaviorStrategy(),
            config=MetathinConfig.create_minimal(),
        )

        nested_path = tmp_path / "nested" / "deep" / "agent.pkl"
        assert agent.save(nested_path) is True
        assert nested_path.exists()

    def test_load_nonexistent_file_raises(self):
        pattern = ListPatternSpace(lambda x: [0.0])

        with pytest.raises(FileNotFoundError):
            Metathin.load(
                "nonexistent.pkl",
                pattern_space=pattern,
                behaviors=[],
                selector=FixedSelector(),
                decision_strategy=FirstBehaviorStrategy(),
            )

    def test_load_missing_behaviors_raises(self, tmp_path):
        pattern = ListPatternSpace(lambda x: [len(str(x))])
        agent = Metathin(
            pattern_space=pattern,
            selector=FixedSelector(),
            decision_strategy=FirstBehaviorStrategy(),
            config=MetathinConfig.create_minimal(),
        )

        filepath = tmp_path / "agent.pkl"
        agent.save(filepath)

        with pytest.raises(ValueError, match="behaviors parameter must be provided"):
            Metathin.load(
                filepath,
                pattern_space=pattern,
                selector=FixedSelector(),
                decision_strategy=FirstBehaviorStrategy(),
            )

    def test_load_missing_selector_raises(self, tmp_path):
        pattern = ListPatternSpace(lambda x: [len(str(x))])
        agent = Metathin(
            pattern_space=pattern,
            selector=FixedSelector(),
            decision_strategy=FirstBehaviorStrategy(),
            config=MetathinConfig.create_minimal(),
        )
        filepath = tmp_path / "agent.pkl"
        agent.save(filepath)

        with pytest.raises(ValueError, match="selector parameter must be provided"):
            Metathin.load(
                filepath,
                pattern_space=pattern,
                behaviors=[],
                decision_strategy=FirstBehaviorStrategy(),
            )
