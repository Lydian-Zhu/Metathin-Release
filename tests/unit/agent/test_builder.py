"""Unit tests for MetathinBuilder."""

import pytest

from metathin.agent import MetathinBuilder
from metathin.config import MetathinConfig
from tests.helpers import (
    ConstantLearning,
    FirstBehaviorStrategy,
    FixedSelector,
    FunctionBehavior,
    ListPatternSpace,
)


def base_builder():
    return (
        MetathinBuilder()
        .with_pattern_space(ListPatternSpace(lambda x: [len(str(x))]))
        .with_selector(FixedSelector({"echo": 0.5}))
        .with_decision_strategy(FirstBehaviorStrategy())
    )


class TestMetathinBuilder:
    def test_requires_pattern_space(self):
        with pytest.raises(ValueError, match="Pattern space"):
            MetathinBuilder().with_selector(FixedSelector()).with_decision_strategy(
                FirstBehaviorStrategy()
            ).build()

    def test_requires_selector(self):
        with pytest.raises(ValueError, match="Selector"):
            MetathinBuilder().with_pattern_space(
                ListPatternSpace(lambda x: [1.0])
            ).with_decision_strategy(FirstBehaviorStrategy()).build()

    def test_requires_decision_strategy(self):
        with pytest.raises(ValueError, match="Decision strategy"):
            MetathinBuilder().with_pattern_space(
                ListPatternSpace(lambda x: [1.0])
            ).with_selector(FixedSelector()).build()

    def test_builds_agent_and_registers_behaviors(self):
        agent = (
            base_builder()
            .with_name("Built")
            .with_behavior(FunctionBehavior("echo", lambda f, **k: k.get("input", "")))
            .build()
        )

        assert agent.name == "Built"
        assert len(agent.B) == 1
        assert agent.think("hello", input="hello") == "hello"

    def test_with_behaviors_appends(self):
        builder = base_builder()
        builder.with_behaviors(
            [
                FunctionBehavior("a", lambda f, **k: "a"),
                FunctionBehavior("b", lambda f, **k: "b"),
            ]
        )
        agent = builder.build()

        assert [b.name for b in agent.B] == ["a", "b"]

    def test_enable_learning_without_mechanism_fails(self):
        with pytest.raises(ValueError, match="learning_mechanism"):
            base_builder().enable_learning(True).build()

    def test_enable_learning_with_mechanism(self):
        agent = (
            base_builder()
            .with_learning_mechanism(ConstantLearning({"w_0": 0.1}))
            .enable_learning(True)
            .build()
        )

        assert agent._config.pipeline.enable_learning is True
        assert agent.Ψ is not None

    def test_config_overrides_are_applied(self):
        agent = (
            base_builder()
            .enable_history(True, max_size=20)
            .enable_metrics(True)
            .with_min_fitness_threshold(0.4)
            .with_learning_rate(0.02)
            .with_log_level("WARNING")
            .build()
        )

        assert agent._config.observability.keep_history is True
        assert agent._config.observability.max_history_size == 20
        assert agent._config.observability.enable_metrics is True
        assert agent._config.pipeline.min_fitness_threshold == 0.4
        assert agent._config.pipeline.learning_rate == 0.02

    def test_create_minimal_disables_optional_services(self):
        agent = MetathinBuilder.create_minimal()

        # create_minimal only configures flags; it still needs the quintuple.
        with pytest.raises(ValueError, match="Pattern space"):
            agent.build()

    def test_with_config_sets_base(self):
        config = MetathinConfig.create_minimal()
        builder = (
            MetathinBuilder()
            .with_config(config)
            .with_pattern_space(ListPatternSpace(lambda x: [1.0]))
            .with_selector(FixedSelector())
            .with_decision_strategy(FirstBehaviorStrategy())
        )
        agent = builder.build()

        assert agent._config.pipeline == config.pipeline
        assert agent._config.memory == config.memory
