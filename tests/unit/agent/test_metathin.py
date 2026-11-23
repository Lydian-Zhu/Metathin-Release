"""Unit tests for the Metathin facade contract."""

import pytest

from metathin import Metathin
from metathin.config import MetathinConfig, ObservabilityConfig, PipelineConfig
from metathin.core import MetaBehavior
from tests.helpers import (
    ConstantLearning,
    FirstBehaviorStrategy,
    FixedSelector,
    FunctionBehavior,
    ListPatternSpace,
)


def make_agent(**overrides):
    kwargs = dict(
        pattern_space=ListPatternSpace(lambda x: [len(str(x))]),
        selector=FixedSelector({"echo": 0.5}),
        decision_strategy=FirstBehaviorStrategy(),
        config=MetathinConfig.create_minimal(),
    )
    kwargs.update(overrides)
    return Metathin(**kwargs)


class TestMetathinFacade:
    def test_requires_pattern_space(self):
        with pytest.raises(ValueError, match="pattern_space"):
            Metathin(
                selector=FixedSelector(),
                decision_strategy=FirstBehaviorStrategy(),
                config=MetathinConfig.create_minimal(),
            )

    def test_requires_selector(self):
        with pytest.raises(ValueError, match="selector"):
            Metathin(
                pattern_space=ListPatternSpace(lambda x: [1.0]),
                decision_strategy=FirstBehaviorStrategy(),
                config=MetathinConfig.create_minimal(),
            )

    def test_requires_decision_strategy(self):
        with pytest.raises(ValueError, match="decision_strategy"):
            Metathin(
                pattern_space=ListPatternSpace(lambda x: [1.0]),
                selector=FixedSelector(),
                config=MetathinConfig.create_minimal(),
            )

    def test_rejects_wrong_component_types(self):
        with pytest.raises(TypeError, match="PatternSpace"):
            Metathin(
                pattern_space=object(),
                selector=FixedSelector(),
                decision_strategy=FirstBehaviorStrategy(),
                config=MetathinConfig.create_minimal(),
            )

    def test_learning_requires_an_explicit_mechanism(self):
        config = MetathinConfig(
            pipeline=PipelineConfig(enable_learning=True)
        )
        with pytest.raises(ValueError, match="learning_mechanism"):
            Metathin(
                pattern_space=ListPatternSpace(lambda x: [1.0]),
                selector=FixedSelector(),
                decision_strategy=FirstBehaviorStrategy(),
                config=config,
            )

    def test_think_returns_behavior_result(self):
        agent = make_agent()
        agent.register_behavior(
            FunctionBehavior("echo", lambda f, **k: k.get("input", ""))
        )

        assert agent.think("hello", input="hello") == "hello"

    def test_register_behavior_validates_name_and_duplicates(self):
        agent = make_agent()
        agent.register_behavior(FunctionBehavior("echo", lambda f, **k: "ok"))

        with pytest.raises(ValueError, match="already exists"):
            agent.register_behavior(FunctionBehavior("echo", lambda f, **k: "ok"))

        with pytest.raises(TypeError, match="MetaBehavior"):
            agent.register_behavior(object())

    def test_get_unregister_and_contains(self):
        agent = make_agent()
        behavior = FunctionBehavior("echo", lambda f, **k: "ok")
        agent.register_behavior(behavior)

        assert "echo" in agent
        assert agent.get_behavior("echo") is behavior
        assert agent["echo"] is behavior
        assert agent.unregister_behavior("echo") is True
        assert agent.unregister_behavior("echo") is False

    def test_empty_history_tracker_is_truthy(self):
        config = MetathinConfig(
            observability=ObservabilityConfig(keep_history=True),
            pipeline=PipelineConfig(enable_learning=False),
        )
        agent = make_agent(config=config)
        agent.register_behavior(FunctionBehavior("noop", lambda f, **k: "ok"))

        assert agent.get_history() == []
        assert agent._ensure_services() is None
        assert agent._history is not None
        assert list(agent._history) == []

    def test_history_records_success_and_failure(self):
        config = MetathinConfig(
            observability=ObservabilityConfig(
                keep_history=True,
                keep_failed=True,
            ),
            pipeline=PipelineConfig(enable_learning=False, raise_on_error=False),
        )
        agent = make_agent(config=config)

        def maybe_fail(features, **kwargs):
            if features[0] > 10:
                raise ValueError("too long")
            return "ok"

        agent.register_behavior(FunctionBehavior("maybe", maybe_fail))

        assert agent.think("short") == "ok"
        assert agent.think("this input is long") is None
        assert len(agent.get_history()) == 2
        assert agent.get_history(failure_only=True)[0].error_message is not None
        assert agent.get_history(success_only=True)[0].error_message is None

    def test_reentrant_think_is_rejected(self):
        agent = make_agent()
        caught = []

        def reenter(features, **kwargs):
            try:
                agent.think("nested")
            except Exception as exc:  # noqa: BLE001 - asserted below
                caught.append(exc)
            return "done"

        agent.register_behavior(FunctionBehavior("reenter", reenter))
        agent.think("outer")

        assert caught
        assert "already thinking" in str(caught[0])

    def test_learning_mechanism_is_called(self):
        config = MetathinConfig(pipeline=PipelineConfig(enable_learning=True))
        learner = ConstantLearning({"w_0": 0.1})
        selector = FixedSelector({"noop": 0.5})
        agent = make_agent(
            config=config,
            learning_mechanism=learner,
            selector=selector,
        )
        agent.register_behavior(FunctionBehavior("noop", lambda f, **k: "ok"))

        agent.think("x", expected="ok")

        assert selector.get_parameters()["w_0"] == 0.1
        assert learner.calls[0]["behavior_name"] == "noop"

    def test_service_init_failure_does_not_lock_agent(self, monkeypatch):
        agent = make_agent()
        agent.register_behavior(FunctionBehavior("noop", lambda f, **k: "ok"))
        original = agent._ensure_services

        def boom():
            raise RuntimeError("service init failed")

        monkeypatch.setattr(agent, "_ensure_services", boom)
        with pytest.raises(RuntimeError, match="service init failed"):
            agent.think("x")
        assert agent._is_thinking is False

        monkeypatch.setattr(agent, "_ensure_services", original)
        assert agent.think("x") == "ok"

    def test_package_does_not_export_concrete_components(self):
        import metathin

        with pytest.raises(AttributeError):
            getattr(metathin, "SimplePatternSpace")
        with pytest.raises(ModuleNotFoundError):
            __import__("metathin.components")
        with pytest.raises(ModuleNotFoundError):
            __import__("metathin_plus")
