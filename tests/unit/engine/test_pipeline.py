"""
Unit tests for ThinkingPipeline.
思考流水线单元测试。
"""

import pytest
import numpy as np
import time
from metathin.engine import ThinkingPipeline, PipelineResult, ThinkingContext
from metathin.core import (
    PatternSpace, MetaBehavior, Selector, DecisionStrategy, LearningMechanism,
    PatternExtractionError, BehaviorExecutionError, NoBehaviorError
)
from metathin.core.types import FeatureVector, FitnessScore, ParameterDict


# ============================================================
# Test Components
# ============================================================

class SimplePattern(PatternSpace):
    def extract(self, raw_input) -> FeatureVector:
        if raw_input is None:
            raise PatternExtractionError("Input is None")
        return np.array([float(len(str(raw_input)))], dtype=np.float64)


class FailingPattern(PatternSpace):
    def extract(self, raw_input) -> FeatureVector:
        raise PatternExtractionError("Feature extraction failed")


class DummyBehavior(MetaBehavior):
    def __init__(self, name: str, should_fail: bool = False, delay: float = 0.0):
        super().__init__()
        self._name = name
        self._should_fail = should_fail
        self._delay = delay
    
    @property
    def name(self) -> str:
        return self._name
    
    def execute(self, features: FeatureVector, **kwargs):
        if self._delay:
            time.sleep(self._delay)
        if self._should_fail:
            raise ValueError(f"Behavior {self._name} failed")
        return f"Result from {self._name}"
    
    def can_execute(self, features: FeatureVector) -> bool:
        return features[0] >= 0 if len(features) > 0 else True


class SimpleSelector(Selector):
    def __init__(self):
        super().__init__()
        self._fitness_map = {}
    
    def set_fitness(self, behavior_name: str, fitness: float):
        self._fitness_map[behavior_name] = fitness
    
    def compute_fitness(self, behavior: MetaBehavior, features: FeatureVector) -> FitnessScore:
        return self._fitness_map.get(behavior.name, 0.5)


class GreedyStrategy(DecisionStrategy):
    def select(self, behaviors, fitness_scores, features):
        if not behaviors:
            raise NoBehaviorError("No behaviors available")
        max_idx = max(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
        return behaviors[max_idx]


class SimpleLearning(LearningMechanism):
    def __init__(self):
        self.last_adjustment = None
    
    def compute_adjustment(self, expected, actual, context):
        self.last_adjustment = {'adjusted': True}
        return self.last_adjustment


# ============================================================
# Test ThinkingPipeline
# ============================================================

class TestThinkingPipeline:
    """Test ThinkingPipeline core functionality."""
    
    def test_pipeline_initialization(self):
        """Pipeline should initialize with default config."""
        pipeline = ThinkingPipeline()
        assert pipeline.min_fitness_threshold == 0.0
        assert pipeline.enable_learning is True
    
    def test_pipeline_with_custom_config(self):
        """Pipeline should accept custom configuration."""
        pipeline = ThinkingPipeline({
            'min_fitness_threshold': 0.5,
            'enable_learning': False
        })
        
        assert pipeline.min_fitness_threshold == 0.5
        assert pipeline.enable_learning is False
    
    def test_pipeline_successful_execution(self):
        """Pipeline should execute successfully with valid components."""
        pipeline = ThinkingPipeline()
        
        pattern = SimplePattern()
        behaviors = [DummyBehavior("b1"), DummyBehavior("b2")]
        selector = SimpleSelector()
        selector.set_fitness("b1", 0.9)
        selector.set_fitness("b2", 0.3)
        strategy = GreedyStrategy()
        
        result = pipeline.run(
            raw_input="hello",
            pattern_space=pattern,
            behaviors=behaviors,
            selector=selector,
            decision_strategy=strategy,
        )
        
        assert result.success is True
        assert result.result == "Result from b1"
        assert result.selected_behavior == "b1"
        assert result.error is None
    
    def test_pipeline_returns_fitness_scores(self):
        """Pipeline should return fitness scores for all behaviors."""
        pipeline = ThinkingPipeline()
        
        pattern = SimplePattern()
        behaviors = [DummyBehavior("b1"), DummyBehavior("b2")]
        selector = SimpleSelector()
        selector.set_fitness("b1", 0.9)
        selector.set_fitness("b2", 0.5)
        strategy = GreedyStrategy()
        
        result = pipeline.run(
            raw_input="test",
            pattern_space=pattern,
            behaviors=behaviors,
            selector=selector,
            decision_strategy=strategy,
        )
        
        assert result.fitness_scores == {"b1": 0.9, "b2": 0.5}
    
    def test_pipeline_returns_stage_times(self):
        """Pipeline should return timing information for each stage."""
        pipeline = ThinkingPipeline()
        
        pattern = SimplePattern()
        behaviors = [DummyBehavior("b1")]
        selector = SimpleSelector()
        selector.set_fitness("b1", 0.9)
        strategy = GreedyStrategy()
        
        result = pipeline.run(
            raw_input="test",
            pattern_space=pattern,
            behaviors=behaviors,
            selector=selector,
            decision_strategy=strategy,
        )
        
        assert 'perceive' in result.stage_times
        assert 'hypothesize' in result.stage_times
        assert 'decide' in result.stage_times
        assert 'execute' in result.stage_times
        assert all(t >= 0 for t in result.stage_times.values())
    
    def test_pipeline_with_learning(self):
        """Pipeline should perform learning when expected is provided."""
        pipeline = ThinkingPipeline()
        
        pattern = SimplePattern()
        behaviors = [DummyBehavior("b1")]
        selector = SimpleSelector()
        selector.set_fitness("b1", 0.5)
        strategy = GreedyStrategy()
        learner = SimpleLearning()
        
        result = pipeline.run(
            raw_input="test",
            pattern_space=pattern,
            behaviors=behaviors,
            selector=selector,
            decision_strategy=strategy,
            learning_mechanism=learner,
            expected=42.0,
        )
        
        assert result.success is True
        assert learner.last_adjustment is not None
    
    def test_pipeline_without_learning_skips(self):
        """Pipeline should skip learning when no learner provided."""
        pipeline = ThinkingPipeline()
        
        pattern = SimplePattern()
        behaviors = [DummyBehavior("b1")]
        selector = SimpleSelector()
        selector.set_fitness("b1", 0.5)
        strategy = GreedyStrategy()
        
        result = pipeline.run(
            raw_input="test",
            pattern_space=pattern,
            behaviors=behaviors,
            selector=selector,
            decision_strategy=strategy,
            expected=42.0,  # expected provided but no learner
        )
        
        assert result.success is True
        assert 'learn' not in result.stage_times
    
    def test_pipeline_with_learning_disabled(self):
        """Pipeline should skip learning when disabled in config."""
        pipeline = ThinkingPipeline({'enable_learning': False})
        
        pattern = SimplePattern()
        behaviors = [DummyBehavior("b1")]
        selector = SimpleSelector()
        selector.set_fitness("b1", 0.5)
        strategy = GreedyStrategy()
        learner = SimpleLearning()
        
        result = pipeline.run(
            raw_input="test",
            pattern_space=pattern,
            behaviors=behaviors,
            selector=selector,
            decision_strategy=strategy,
            learning_mechanism=learner,
            expected=42.0,
        )
        
        assert result.success is True
        assert learner.last_adjustment is None  # Learning didn't happen
    
    def test_pipeline_with_context_params(self):
        """Pipeline should pass context parameters to behaviors."""
        pipeline = ThinkingPipeline()
        
        class ContextBehavior(DummyBehavior):
            def execute(self, features, **kwargs):
                return kwargs.get('custom_param', 'not_passed')
        
        pattern = SimplePattern()
        behaviors = [ContextBehavior("ctx")]
        selector = SimpleSelector()
        selector.set_fitness("ctx", 0.9)
        strategy = GreedyStrategy()
        
        result = pipeline.run(
            raw_input="test",
            pattern_space=pattern,
            behaviors=behaviors,
            selector=selector,
            decision_strategy=strategy,
            context_params={'custom_param': 'passed_value'}
        )
        
        assert result.result == 'passed_value'
    
    def test_pipeline_filters_behaviors_by_can_execute(self):
        """Pipeline should filter behaviors that cannot execute."""
        pipeline = ThinkingPipeline()
        
        pattern = SimplePattern()
        # Behavior that can only execute when feature > 10
        class ConditionalBehavior(DummyBehavior):
            def can_execute(self, features):
                return features[0] > 10
        
        behaviors = [
            DummyBehavior("always"),
            ConditionalBehavior("conditional")
        ]
        selector = SimpleSelector()
        selector.set_fitness("always", 0.9)
        selector.set_fitness("conditional", 0.8)
        strategy = GreedyStrategy()
        
        result = pipeline.run(
            raw_input="short",  # len("short") = 5, not > 10
            pattern_space=pattern,
            behaviors=behaviors,
            selector=selector,
            decision_strategy=strategy,
        )
        
        # Only "always" should be considered
        assert result.selected_behavior == "always"
        assert "conditional" not in result.fitness_scores
    
    def test_pipeline_filters_by_fitness_threshold(self):
        """Pipeline should filter behaviors below fitness threshold."""
        pipeline = ThinkingPipeline({'min_fitness_threshold': 0.7})
        
        pattern = SimplePattern()
        behaviors = [DummyBehavior("high"), DummyBehavior("low")]
        selector = SimpleSelector()
        selector.set_fitness("high", 0.9)
        selector.set_fitness("low", 0.5)
        strategy = GreedyStrategy()
        
        result = pipeline.run(
            raw_input="test",
            pattern_space=pattern,
            behaviors=behaviors,
            selector=selector,
            decision_strategy=strategy,
        )
        
        # Only "high" should be in candidate_behaviors
        assert result.selected_behavior == "high"
    
    def test_pipeline_raises_no_behavior_error_when_no_candidates(self):
        """Pipeline should return error when no behaviors pass filters."""
        pipeline = ThinkingPipeline({'min_fitness_threshold': 0.9})
        
        pattern = SimplePattern()
        behaviors = [DummyBehavior("b1"), DummyBehavior("b2")]
        selector = SimpleSelector()
        selector.set_fitness("b1", 0.5)
        selector.set_fitness("b2", 0.3)
        strategy = GreedyStrategy()
        
        result = pipeline.run(
            raw_input="test",
            pattern_space=pattern,
            behaviors=behaviors,
            selector=selector,
            decision_strategy=strategy,
        )
        
        assert result.success is False
        assert isinstance(result.error, NoBehaviorError)
        assert result.selected_behavior is None
    
    def test_pipeline_handles_pattern_extraction_error(self):
        """Pipeline should handle pattern extraction failures."""
        pipeline = ThinkingPipeline()
        
        pattern = FailingPattern()
        behaviors = [DummyBehavior("b1")]
        selector = SimpleSelector()
        strategy = GreedyStrategy()
        
        result = pipeline.run(
            raw_input="test",
            pattern_space=pattern,
            behaviors=behaviors,
            selector=selector,
            decision_strategy=strategy,
        )
        
        assert result.success is False
        assert isinstance(result.error, PatternExtractionError)
    
    def test_pipeline_handles_behavior_execution_error(self):
        """Pipeline should handle behavior execution failures."""
        pipeline = ThinkingPipeline()
        
        pattern = SimplePattern()
        behaviors = [DummyBehavior("failing", should_fail=True)]
        selector = SimpleSelector()
        selector.set_fitness("failing", 0.9)
        strategy = GreedyStrategy()
        
        result = pipeline.run(
            raw_input="test",
            pattern_space=pattern,
            behaviors=behaviors,
            selector=selector,
            decision_strategy=strategy,
        )
        
        assert result.success is False
        assert isinstance(result.error, BehaviorExecutionError)
    
    def test_pipeline_returns_context(self):
        """Pipeline should return the final context."""
        pipeline = ThinkingPipeline()
        
        pattern = SimplePattern()
        behaviors = [DummyBehavior("b1")]
        selector = SimpleSelector()
        selector.set_fitness("b1", 0.9)
        strategy = GreedyStrategy()
        
        result = pipeline.run(
            raw_input="hello world",
            pattern_space=pattern,
            behaviors=behaviors,
            selector=selector,
            decision_strategy=strategy,
        )
        
        assert result.context is not None
        assert result.context.raw_input == "hello world"
        assert result.context.selected_behavior == "b1"
        assert result.context.result == "Result from b1"


class TestPipelineWithMultipleBehaviors:
    """Test pipeline behavior with multiple behaviors."""
    
    def test_pipeline_selects_best_behavior(self):
        """Pipeline should select behavior with highest fitness."""
        pipeline = ThinkingPipeline()
        
        pattern = SimplePattern()
        behaviors = [DummyBehavior(f"b{i}") for i in range(5)]
        selector = SimpleSelector()
        for i, b in enumerate(behaviors):
            selector.set_fitness(b.name, i / 4.0)  # 0.0, 0.25, 0.5, 0.75, 1.0
        strategy = GreedyStrategy()
        
        result = pipeline.run(
            raw_input="test",
            pattern_space=pattern,
            behaviors=behaviors,
            selector=selector,
            decision_strategy=strategy,
        )
        
        # b4 should have highest fitness (1.0)
        assert result.selected_behavior == "b4"
    
    def test_pipeline_respects_can_execute_override(self):
        """Pipeline should respect can_execute() overrides."""
        pipeline = ThinkingPipeline()
        
        class EvenOnlyBehavior(DummyBehavior):
            def can_execute(self, features):
                return int(features[0]) % 2 == 0
        
        pattern = SimplePattern()
        behaviors = [
            EvenOnlyBehavior("even"),
            DummyBehavior("always")
        ]
        selector = SimpleSelector()
        selector.set_fitness("even", 0.9)
        selector.set_fitness("always", 0.5)
        strategy = GreedyStrategy()
        
        # Input length 5 (odd) -> even behavior cannot execute
        result = pipeline.run(
            raw_input="hello",  # len=5, odd
            pattern_space=pattern,
            behaviors=behaviors,
            selector=selector,
            decision_strategy=strategy,
        )
        
        assert result.selected_behavior == "always"
        
        # Input length 4 (even) -> even behavior can execute and has higher fitness
        result = pipeline.run(
            raw_input="four",  # len=4, even
            pattern_space=pattern,
            behaviors=behaviors,
            selector=selector,
            decision_strategy=strategy,
        )
        
        assert result.selected_behavior == "even"


class TestPipelinePerformance:
    """Test pipeline performance characteristics."""
    
    @pytest.mark.slow
    def test_pipeline_execution_time(self):
        """Pipeline execution time should be reasonable."""
        pipeline = ThinkingPipeline()
        
        pattern = SimplePattern()
        # Use slower behavior to measure
        behaviors = [DummyBehavior("slow", delay=0.01)]
        selector = SimpleSelector()
        selector.set_fitness("slow", 0.9)
        strategy = GreedyStrategy()
        
        start = time.time()
        result = pipeline.run(
            raw_input="test",
            pattern_space=pattern,
            behaviors=behaviors,
            selector=selector,
            decision_strategy=strategy,
        )
        elapsed = time.time() - start
        
        assert result.success is True
        # Should be at least the behavior delay
        assert elapsed >= 0.01
        # Should have recorded execution time
        assert result.stage_times['execute'] >= 0.01