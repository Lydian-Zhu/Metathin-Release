"""
Unit tests for ThinkingContext.
思考上下文单元测试。
"""

import pytest
import numpy as np
from datetime import datetime
from metathin.engine import ThinkingContext


class TestThinkingContext:
    """Test ThinkingContext data class."""
    
    def test_context_initialization(self):
        """Context should initialize with default values."""
        context = ThinkingContext(raw_input="test")
        
        assert context.raw_input == "test"
        assert context.expected is None
        assert context.features is None
        assert context.selected_behavior is None
        assert context.result is None
        assert context.fitness_scores == {}
        assert context.candidate_behaviors == []
        assert context.stage_times == {}
        assert context.metadata == {}
        assert context.error is None
    
    def test_context_with_expected(self):
        """Context should store expected value."""
        context = ThinkingContext(raw_input="test", expected=42.0)
        
        assert context.expected == 42.0
    
    def test_context_with_features(self):
        """Context should store feature vector."""
        features = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        context = ThinkingContext(raw_input="test", features=features)
        
        np.testing.assert_array_equal(context.features, features)
    
    def test_context_with_fitness_scores(self):
        """Context should store fitness scores."""
        fitness_scores = {"b1": 0.9, "b2": 0.5}
        context = ThinkingContext(raw_input="test", fitness_scores=fitness_scores)
        
        assert context.fitness_scores == fitness_scores
    
    def test_context_with_candidate_behaviors(self):
        """Context should store candidate behaviors."""
        candidates = ["b1", "b2", "b3"]
        context = ThinkingContext(raw_input="test", candidate_behaviors=candidates)
        
        assert context.candidate_behaviors == candidates
    
    def test_context_with_selected_behavior(self):
        """Context should store selected behavior."""
        context = ThinkingContext(raw_input="test", selected_behavior="greet")
        
        assert context.selected_behavior == "greet"
    
    def test_context_with_result(self):
        """Context should store execution result."""
        context = ThinkingContext(raw_input="test", result="Hello, world!")
        
        assert context.result == "Hello, world!"
    
    def test_context_with_stage_times(self):
        """Context should store stage timing information."""
        stage_times = {"perceive": 0.001, "decide": 0.002, "execute": 0.005}
        context = ThinkingContext(raw_input="test", stage_times=stage_times)
        
        assert context.stage_times == stage_times
    
    def test_context_with_metadata(self):
        """Context should store metadata."""
        metadata = {"user_id": 123, "session": "abc"}
        context = ThinkingContext(raw_input="test", metadata=metadata)
        
        assert context.metadata == metadata
    
    def test_context_with_error(self):
        """Context should store error information."""
        error = ValueError("Test error")
        context = ThinkingContext(raw_input="test", error=error)
        
        assert context.error == error
    
    def test_total_time_calculation(self):
        """total_time() should calculate total time."""
        context = ThinkingContext(
            raw_input="test",
            stage_times={"perceive": 0.1, "hypothesize": 0.2, 
                        "decide": 0.05, "execute": 0.3, "learn": 0.01}
        )
        
        assert context.total_time() == 0.66
    
    def test_total_time_with_error(self):
        """total_time() should use stage_times when error occurred."""
        context = ThinkingContext(
            raw_input="test",
            stage_times={"perceive": 0.1, "hypothesize": 0.2},
            error=ValueError("test")
        )
        
        # 使用 pytest.approx 处理浮点精度
        assert context.total_time() == pytest.approx(0.3)
    
    def test_to_thought_dict(self):
        """to_thought_dict() should convert to dictionary."""
        features = np.array([1.0, 2.0], dtype=np.float64)
        context = ThinkingContext(
            raw_input="test_input",
            expected=42.0,
            features=features,
            selected_behavior="greet",
            fitness_scores={"greet": 0.9, "echo": 0.5},
            candidate_behaviors=["greet", "echo"],
            result="Hello!",
            stage_times={"decide": 0.1, "execute": 0.2},
            metadata={"key": "value"}
        )
        
        thought_dict = context.to_thought_dict()
        
        assert thought_dict['raw_input'] == "test_input"
        assert thought_dict['expected'] == 42.0
        assert thought_dict['features'] == [1.0, 2.0]
        assert thought_dict['selected_behavior'] == "greet"
        assert thought_dict['fitness_scores'] == {"greet": 0.9, "echo": 0.5}
        assert thought_dict['candidate_behaviors'] == ["greet", "echo"]
        assert thought_dict['result'] == "Hello!"
        assert thought_dict['success'] is True
        assert thought_dict['error_message'] is None
        assert thought_dict['decision_time'] == 0.1
        assert thought_dict['execution_time'] == 0.2
        assert thought_dict['metadata'] == {"key": "value"}
    
    def test_to_thought_dict_with_error(self):
        """to_thought_dict() should include error information."""
        error = ValueError("Something went wrong")
        context = ThinkingContext(
            raw_input="test",
            error=error,
            stage_times={"perceive": 0.1}
        )
        
        thought_dict = context.to_thought_dict()
        
        assert thought_dict['success'] is False
        assert "Something went wrong" in thought_dict['error_message']
    
    def test_with_stage_time_returns_new_context(self):
        """with_stage_time() should return a new context (immutable)."""
        context = ThinkingContext(raw_input="test")
        new_context = context.with_stage_time("perceive", 0.123)
        
        # Original unchanged
        assert context.stage_times == {}
        
        # New has the stage time
        assert new_context.stage_times == {"perceive": 0.123}
    
    def test_with_stage_time_preserves_other_fields(self):
        """with_stage_time() should preserve other fields."""
        context = ThinkingContext(
            raw_input="test",
            expected=42.0,
            selected_behavior="greet"
        )
        
        new_context = context.with_stage_time("decide", 0.05)
        
        assert new_context.raw_input == "test"
        assert new_context.expected == 42.0
        assert new_context.selected_behavior == "greet"
        assert new_context.stage_times == {"decide": 0.05}
    
    def test_multiple_stage_times(self):
        """Multiple with_stage_time calls should accumulate times."""
        context = ThinkingContext(raw_input="test")
        
        context = context.with_stage_time("perceive", 0.1)
        context = context.with_stage_time("hypothesize", 0.2)
        context = context.with_stage_time("decide", 0.05)
        
        assert context.stage_times == {
            "perceive": 0.1,
            "hypothesize": 0.2,
            "decide": 0.05
        }
    
    def test_with_error_returns_new_context(self):
        """with_error() should return a new context with error set."""
        context = ThinkingContext(raw_input="test")
        error = ValueError("Test error")
        
        new_context = context.with_error(error, "perceive")
        
        assert new_context.error == error
        assert new_context.stage_times == {}
        
        # Original unchanged
        assert context.error is None
    
    def test_with_error_preserves_stage_times(self):
        """with_error() should preserve existing stage times."""
        context = ThinkingContext(
            raw_input="test",
            stage_times={"perceive": 0.1}
        )
        error = ValueError("Test error")
        
        new_context = context.with_error(error, "hypothesize")
        
        assert new_context.stage_times == {"perceive": 0.1}
        assert new_context.error == error


class TestStageResultClasses:
    """Test stage result data classes."""
    
    def test_perceive_result(self):
        from metathin.engine import PerceiveResult
        
        features = np.array([1.0, 2.0], dtype=np.float64)
        result = PerceiveResult(features=features, duration=0.001)
        
        np.testing.assert_array_equal(result.features, features)
        assert result.duration == 0.001
    
    def test_hypothesize_result(self):
        from metathin.engine import HypothesizeResult
        
        result = HypothesizeResult(
            candidates=["b1", "b2"],
            fitness_scores={"b1": 0.9, "b2": 0.5},
            duration=0.002
        )
        
        assert result.candidates == ["b1", "b2"]
        assert result.fitness_scores == {"b1": 0.9, "b2": 0.5}
        assert result.duration == 0.002
    
    def test_decide_result(self):
        from metathin.engine import DecideResult
        
        result = DecideResult(
            selected_behavior="greet",
            confidence=0.85,
            duration=0.0005
        )
        
        assert result.selected_behavior == "greet"
        assert result.confidence == 0.85
        assert result.duration == 0.0005
    
    def test_execute_result(self):
        from metathin.engine import ExecuteResult
        
        result = ExecuteResult(result="Hello!", duration=0.01)
        
        assert result.result == "Hello!"
        assert result.duration == 0.01
    
    def test_learn_result(self):
        from metathin.engine import LearnResult
        
        result = LearnResult(
            learning_occurred=True,
            parameter_changes={"w_0_0": 0.05, "b_0": -0.01},
            duration=0.003
        )
        
        assert result.learning_occurred is True
        assert result.parameter_changes == {"w_0_0": 0.05, "b_0": -0.01}
        assert result.duration == 0.003