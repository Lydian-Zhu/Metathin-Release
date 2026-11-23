"""
Integration tests for the complete thinking pipeline.

The framework itself ships interfaces only; these tests use tiny local doubles
from :mod:`tests.helpers`.
"""

import numpy as np

from metathin import Metathin, MetathinBuilder
from metathin.config import MetathinConfig, PipelineConfig
from tests.helpers import (
    ConstantLearning,
    FirstBehaviorStrategy,
    FixedSelector,
    FunctionBehavior,
    ListPatternSpace,
    MaxBehaviorStrategy,
)


class TestFullPipeline:
    """Test complete pipeline from end to end."""

    def test_end_to_end_with_user_components(self):
        pattern = ListPatternSpace(lambda x: [len(x), len(x.split()), x.count("a")])
        behaviors = [
            FunctionBehavior("summarize", lambda f, **k: f"len={int(f[0])}"),
            FunctionBehavior("uppercase", lambda f, **k: k.get("user_input", "").upper()),
        ]
        selector = FixedSelector({"summarize": 0.9, "uppercase": 0.2})

        agent = (
            MetathinBuilder()
            .with_pattern_space(pattern)
            .with_behaviors(behaviors)
            .with_selector(selector)
            .with_decision_strategy(MaxBehaviorStrategy())
            .build()
        )

        result = agent.think("hello amazing world", user_input="hello amazing world")
        assert result == "len=19"

    def test_learning_uses_the_injected_mechanism(self):
        pattern = ListPatternSpace(lambda x: [float(x)])
        behaviors = [
            FunctionBehavior("predict_10", lambda f, **k: 10.0),
            FunctionBehavior("predict_20", lambda f, **k: 20.0),
        ]
        selector = FixedSelector({"predict_10": 0.6, "predict_20": 0.4})
        learner = ConstantLearning({"w_0": 0.1})

        agent = Metathin(
            pattern_space=pattern,
            selector=selector,
            decision_strategy=MaxBehaviorStrategy(),
            learning_mechanism=learner,
            config=MetathinConfig(pipeline=PipelineConfig(enable_learning=True)),
            name="LearningAgent",
        )
        agent.register_behaviors(behaviors)

        agent.think(15.0, expected=15.0)

        assert selector.get_parameters()["w_0"] == 0.1
        assert learner.calls
        assert learner.calls[0]["behavior_name"] == "predict_10"

    def test_pipeline_works_without_learning(self):
        pattern = ListPatternSpace(lambda x: [len(str(x))])
        behavior = FunctionBehavior("echo", lambda f, **k: k.get("input", ""))
        agent = Metathin(
            pattern_space=pattern,
            selector=FixedSelector({"echo": 0.5}),
            decision_strategy=FirstBehaviorStrategy(),
            config=MetathinConfig.create_minimal(),
        )
        agent.register_behavior(behavior)

        result = agent.think("hello", input="hello", expected="hello")
        assert result == "hello"

    def test_behavior_failure_is_reported_without_crashing(self):
        pattern = ListPatternSpace(lambda x: [len(str(x))])

        def failing(features, **kwargs):
            raise ValueError("intentional failure")

        config = MetathinConfig(
            pipeline=PipelineConfig(enable_learning=False, raise_on_error=False)
        )
        agent = Metathin(
            pattern_space=pattern,
            selector=FixedSelector({"failing": 0.5}),
            decision_strategy=FirstBehaviorStrategy(),
            config=config,
        )
        agent.register_behavior(FunctionBehavior("failing", failing))

        result = agent.think("test")
        assert result is None
        assert agent.get_stats()["error_count"] == 1

    def test_invalid_fitness_is_rejected(self):
        pattern = ListPatternSpace(lambda x: [1.0])

        class BadSelector(FixedSelector):
            def compute_fitness(self, behavior, features):
                return 1.5

        agent = Metathin(
            pattern_space=pattern,
            selector=BadSelector(),
            decision_strategy=FirstBehaviorStrategy(),
            config=MetathinConfig(
                pipeline=PipelineConfig(enable_learning=False, raise_on_error=False)
            ),
        )
        agent.register_behavior(FunctionBehavior("noop", lambda f, **k: "ok"))

        result = agent.think("x")
        assert result is None
        assert "Fitness" in str(agent.get_last_thought().error_message)

    def test_pattern_space_must_return_valid_features(self):
        class BadPattern(ListPatternSpace):
            def extract(self, raw_input):
                return np.array([1.0, np.nan], dtype=np.float64)

        agent = Metathin(
            pattern_space=BadPattern(lambda x: [1.0]),
            selector=FixedSelector({"noop": 0.5}),
            decision_strategy=FirstBehaviorStrategy(),
            config=MetathinConfig(
                pipeline=PipelineConfig(enable_learning=False, raise_on_error=False)
            ),
        )
        agent.register_behavior(FunctionBehavior("noop", lambda f, **k: "ok"))

        result = agent.think("x")
        assert result is None
        assert "finite" in str(agent.get_last_thought().error_message)
