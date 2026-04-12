"""
Unit tests for core exceptions.
核心异常单元测试。
"""

import pytest
from metathin.core import (
    MetathinError,
    PatternExtractionError,
    BehaviorExecutionError,
    FitnessComputationError,
    DecisionError,
    NoBehaviorError,
    LearningError,
    ParameterUpdateError,
)


class TestExceptionHierarchy:
    """Test exception inheritance hierarchy."""
    
    def test_all_exceptions_inherit_from_metathin_error(self):
        """All custom exceptions should inherit from MetathinError."""
        exceptions = [
            PatternExtractionError,
            BehaviorExecutionError,
            FitnessComputationError,
            DecisionError,
            NoBehaviorError,
            LearningError,
            ParameterUpdateError,
        ]
        
        for exc_class in exceptions:
            assert issubclass(exc_class, MetathinError)
    
    def test_no_behavior_error_inherits_from_decision_error(self):
        """NoBehaviorError should inherit from DecisionError."""
        assert issubclass(NoBehaviorError, DecisionError)
    
    def test_parameter_update_error_inherits_from_learning_error(self):
        """ParameterUpdateError should inherit from LearningError."""
        assert issubclass(ParameterUpdateError, LearningError)


class TestExceptionUsage:
    """Test exception raising and catching."""
    
    def test_pattern_extraction_error_can_be_raised(self):
        """PatternExtractionError should be raiseable."""
        with pytest.raises(PatternExtractionError, match="Feature extraction failed"):
            raise PatternExtractionError("Feature extraction failed")
    
    def test_behavior_execution_error_can_be_raised(self):
        """BehaviorExecutionError should be raiseable."""
        with pytest.raises(BehaviorExecutionError, match="Behavior failed"):
            raise BehaviorExecutionError("Behavior failed")
    
    def test_fitness_computation_error_can_be_raised(self):
        """FitnessComputationError should be raiseable."""
        with pytest.raises(FitnessComputationError, match="Fitness computation failed"):
            raise FitnessComputationError("Fitness computation failed")
    
    def test_decision_error_can_be_raised(self):
        """DecisionError should be raiseable."""
        with pytest.raises(DecisionError, match="Decision failed"):
            raise DecisionError("Decision failed")
    
    def test_no_behavior_error_can_be_raised(self):
        """NoBehaviorError should be raiseable."""
        with pytest.raises(NoBehaviorError, match="No behaviors available"):
            raise NoBehaviorError("No behaviors available")
    
    def test_learning_error_can_be_raised(self):
        """LearningError should be raiseable."""
        with pytest.raises(LearningError, match="Learning failed"):
            raise LearningError("Learning failed")
    
    def test_parameter_update_error_can_be_raised(self):
        """ParameterUpdateError should be raiseable."""
        with pytest.raises(ParameterUpdateError, match="Parameter update failed"):
            raise ParameterUpdateError("Parameter update failed")
    
    def test_catch_base_exception(self):
        """Should be able to catch all Metathin errors with base class."""
        exceptions = [
            PatternExtractionError("test"),
            BehaviorExecutionError("test"),
            FitnessComputationError("test"),
        ]
        
        for exc in exceptions:
            with pytest.raises(MetathinError):
                raise exc