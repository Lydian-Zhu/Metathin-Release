"""
Integration tests for the agent facade and optional services.

All P/B/S/D/Ψ components are supplied by the test, not by the package.
"""

from metathin.agent import Metathin, MetathinBuilder
from metathin.config import (
    MemoryConfig,
    MetathinConfig,
    ObservabilityConfig,
    PipelineConfig,
)
from tests.helpers import (
    ConstantLearning,
    FirstBehaviorStrategy,
    FixedSelector,
    FunctionBehavior,
    ListPatternSpace,
    MaxBehaviorStrategy,
)


class TestAgentWithServices:
    """Test agent with services."""

    def test_agent_with_full_observability(self):
        config = MetathinConfig(
            observability=ObservabilityConfig(
                keep_history=True,
                max_history_size=100,
                enable_metrics=True,
            ),
            memory=MemoryConfig(enabled=False),
            pipeline=PipelineConfig(enable_learning=False),
        )
        pattern = ListPatternSpace(lambda x: [len(str(x))])
        behavior = FunctionBehavior("echo", lambda f, **k: k.get("input", ""))

        agent = Metathin(
            pattern_space=pattern,
            selector=FixedSelector({"echo": 0.5}),
            decision_strategy=FirstBehaviorStrategy(),
            config=config,
            name="ObservableAgent",
        )
        agent.register_behavior(behavior)

        for i in range(5):
            agent.think(f"test_{i}", input=f"test_{i}")

        stats = agent.get_stats()
        assert stats["total_thoughts"] == 5
        assert stats["history_stats"]["total_thoughts"] == 5
        assert agent.get_last_thought() is not None

    def test_agent_with_memory_service(self):
        config = MetathinConfig(
            memory=MemoryConfig(enabled=True, backend_type="memory"),
            observability=ObservabilityConfig(keep_history=False),
            pipeline=PipelineConfig(enable_learning=False),
        )
        pattern = ListPatternSpace(lambda x: [len(str(x))])
        agent = Metathin(
            pattern_space=pattern,
            selector=FixedSelector(),
            decision_strategy=FirstBehaviorStrategy(),
            config=config,
            name="MemoryAgent",
        )

        assert agent.remember("stored_value", "persisted_value") is True
        assert agent.recall("stored_value", "default") == "persisted_value"

    def test_agent_with_learning_and_history(self):
        config = MetathinConfig(
            observability=ObservabilityConfig(keep_history=True, enable_metrics=True),
            pipeline=PipelineConfig(enable_learning=True),
        )
        pattern = ListPatternSpace(lambda x: [float(x)])
        selector = FixedSelector({"predict_10": 0.6, "predict_20": 0.4})
        learner = ConstantLearning({"w_0": 0.1})

        agent = Metathin(
            pattern_space=pattern,
            selector=selector,
            decision_strategy=MaxBehaviorStrategy(),
            learning_mechanism=learner,
            config=config,
            name="LearningAgent",
        )
        agent.register_behaviors(
            [
                FunctionBehavior("predict_10", lambda f, **k: 10.0),
                FunctionBehavior("predict_20", lambda f, **k: 20.0),
            ]
        )

        for _ in range(10):
            agent.think(15.0, expected=10.0)

        assert agent.get_stats()["total_thoughts"] == 10
        assert learner.calls
        assert agent.get_last_thought().success is True

    def test_agent_with_all_services_and_error_recovery(self):
        config = MetathinConfig(
            memory=MemoryConfig(enabled=True, backend_type="memory"),
            observability=ObservabilityConfig(
                keep_history=True,
                keep_failed=True,
                enable_metrics=True,
            ),
            pipeline=PipelineConfig(enable_learning=False, raise_on_error=False),
        )
        pattern = ListPatternSpace(lambda x: [len(str(x))])

        def sometimes_fails(features, **kwargs):
            if features[0] > 5:
                raise ValueError("Input too long")
            return "Success"

        agent = Metathin(
            pattern_space=pattern,
            selector=FixedSelector({"risky": 0.5}),
            decision_strategy=FirstBehaviorStrategy(),
            config=config,
            name="RobustAgent",
        )
        agent.register_behavior(FunctionBehavior("risky", sometimes_fails))

        assert agent.think("short") == "Success"
        assert agent.think("this is a very long input") is None
        assert agent.think("ok") == "Success"
        assert agent.get_stats()["error_count"] == 1

    def test_agent_builder_with_services(self):
        agent = (
            MetathinBuilder()
            .with_name("BuiltAgent")
            .with_pattern_space(ListPatternSpace(lambda x: [len(str(x))]))
            .with_behavior(FunctionBehavior("test", lambda f, **k: "result"))
            .with_selector(FixedSelector({"test": 0.5}))
            .with_decision_strategy(FirstBehaviorStrategy())
            .enable_memory(True)
            .enable_history(True)
            .enable_metrics(True)
            .with_log_level("DEBUG")
            .build()
        )

        assert agent.name == "BuiltAgent"
        assert len(agent.B) == 1
        assert agent.think("test") == "result"
        assert agent.get_stats() is not None
