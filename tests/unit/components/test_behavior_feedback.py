import numpy as np
import pytest

from metathin import (
    MetathinBuilder, SimplePatternSpace, FunctionBehavior, SimpleSelector,
    MaxFitnessStrategy, BehaviorFeedbackLearning,
)


class TestBehaviorFeedbackLearning:
    def test_negative_feedback_penalizes_selected_behavior(self):
        learner = BehaviorFeedbackLearning(learning_rate=0.1)
        context = {
            "behavior_index": 1,
            "features": np.array([2.0, 1.0]),
            "parameters": {"w_1_0": 0.0, "w_1_1": 0.0},
        }

        adjustment = learner.compute_adjustment("monitor", "inspect", context)

        assert adjustment["w_1_0"] < 0
        assert adjustment["w_1_1"] < 0

    def test_correct_answer_does_not_penalize_by_default(self):
        learner = BehaviorFeedbackLearning(learning_rate=0.1)
        context = {
            "behavior_index": 0,
            "features": np.array([2.0, 1.0]),
            "parameters": {"w_0_0": 0.0, "w_0_1": 0.0},
        }

        adjustment = learner.compute_adjustment("monitor", "monitor", context)

        assert adjustment is None or all(v == 0 for v in adjustment.values())

    def test_optional_positive_feedback(self):
        learner = BehaviorFeedbackLearning(learning_rate=0.1, positive_feedback=1.0)
        context = {
            "behavior_index": 0,
            "features": np.array([2.0, 1.0]),
            "parameters": {"w_0_0": 0.0, "w_0_1": 0.0},
        }

        adjustment = learner.compute_adjustment("monitor", "monitor", context)

        assert adjustment["w_0_0"] > 0
        assert adjustment["w_0_1"] > 0

    def test_requires_behavior_index(self):
        learner = BehaviorFeedbackLearning()
        context = {
            "features": np.array([1.0]),
            "parameters": {"w_0_0": 0.0},
        }

        assert learner.compute_adjustment("a", "b", context) is None

    def test_numeric_target_uses_graded_error(self):
        learner = BehaviorFeedbackLearning(learning_rate=0.1)
        context = {
            "behavior_index": 0,
            "features": np.array([1.0, 1.0]),
            "parameters": {"w_0_0": 0.0, "w_0_1": 0.0},
        }

        adjustment = learner.compute_adjustment(10.0, 9.0, context)

        # relative error 0.1 -> feedback -0.1, not a full -1 penalty
        assert adjustment["w_0_0"] == pytest.approx(-0.01)
        assert learner.feedback_history[-1] == pytest.approx(-0.1)

    def test_exploration_samples_and_decays(self, monkeypatch):
        learner = BehaviorFeedbackLearning(
            exploration_rate=0.5, exploration_decay=0.5, min_exploration=0.01,
        )
        monkeypatch.setattr("metathin.components.learning.random.random", lambda: 0.1)

        assert learner.should_explore() is True
        assert learner.get_exploration_rate() == pytest.approx(0.25)

    def test_exploration_is_skipped_when_confident(self, monkeypatch):
        learner = BehaviorFeedbackLearning(
            exploration_rate=1.0, exploration_margin=0.1,
        )
        monkeypatch.setattr("metathin.components.learning.random.random", lambda: 0.0)

        assert learner.should_explore([0.9, 0.2]) is False
        assert learner.should_explore([0.6, 0.58]) is True

    def test_zero_exploration_never_samples(self):
        learner = BehaviorFeedbackLearning(exploration_rate=0.0)

        assert learner.should_explore([0.5, 0.5]) is False

    def test_classification_learning_improves_accuracy(self):
        np.random.seed(42)
        space = SimplePatternSpace(lambda s: [s["vib"], 1.0])
        behaviors = [
            FunctionBehavior("monitor", lambda f, **k: "monitor"),
            FunctionBehavior("inspect", lambda f, **k: "inspect"),
        ]
        agent = (
            MetathinBuilder()
            .with_name("feedback-test")
            .with_pattern_space(space)
            .with_behaviors(behaviors)
            .with_selector(SimpleSelector(n_features=2, n_behaviors=2))
            .with_decision_strategy(MaxFitnessStrategy())
            .enable_learning(True)
            .with_learning_mechanism(BehaviorFeedbackLearning(learning_rate=0.01))
            .build()
        )

        errors = []
        for _ in range(1200):
            vib = float(np.random.uniform(0.5, 5.0))
            target = "inspect" if vib > 2.8 else "monitor"
            output = agent.think({"vib": vib}, expected=target)
            errors.append(0.0 if output == target else 1.0)

        first = 1 - np.mean(errors[:20])
        last = 1 - np.mean(errors[-20:])
        assert last > first
        assert last >= 0.9
