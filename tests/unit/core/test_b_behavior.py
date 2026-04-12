"""
Unit tests for MetaBehavior interface.
元行为接口单元测试。
"""

import pytest
import numpy as np
from metathin.core import MetaBehavior, BehaviorExecutionError, FeatureVector


class SimpleBehavior(MetaBehavior):
    """Simple behavior for testing."""
    
    def __init__(self, name: str = "simple", should_fail: bool = False):
        super().__init__()
        self._name = name
        self._should_fail = should_fail
        self._last_result = None
    
    @property
    def name(self) -> str:
        return self._name
    
    def execute(self, features: FeatureVector, **kwargs) -> str:
        if self._should_fail:
            from metathin.core.exceptions import BehaviorExecutionError
            error = BehaviorExecutionError(f"Behavior {self.name} execution failed: Intentional failure")
            self._last_error = error
            raise error
        
        self._execution_count += 1
        self._last_result = f"Executed {self._name} with features: {features}"
        return self._last_result
    
    def get_last_result(self):
        return self._last_result


class CountingBehavior(MetaBehavior):
    """Behavior that counts executions."""
    
    def __init__(self, name: str = "counter"):
        super().__init__()
        self._name = name
        self._count = 0
    
    @property
    def name(self) -> str:
        return self._name
    
    def execute(self, features: FeatureVector, **kwargs) -> int:
        self._count += 1
        self._execution_count += 1
        return self._count
    
    def get_count(self) -> int:
        return self._count


class TestMetaBehavior:
    """Test MetaBehavior interface."""
    
    def test_behavior_has_name(self):
        """Behavior should have a name property."""
        behavior = SimpleBehavior("test_name")
        assert behavior.name == "test_name"
    
    def test_behavior_execute_returns_result(self):
        """execute() should return a result."""
        behavior = SimpleBehavior("test")
        features = np.array([1.0, 2.0], dtype=np.float64)
        result = behavior.execute(features)
        
        assert isinstance(result, str)
        assert "test" in result
    
    def test_behavior_tracks_execution_count(self):
        """Behavior should track execution count."""
        behavior = CountingBehavior()
        features = np.array([1.0], dtype=np.float64)
        
        assert behavior._execution_count == 0
        
        behavior.execute(features)
        assert behavior._execution_count == 1
        
        behavior.execute(features)
        assert behavior._execution_count == 2
    
    def test_behavior_can_execute_default_true(self):
        """can_execute() should default to True."""
        behavior = SimpleBehavior()
        features = np.array([1.0], dtype=np.float64)
        
        assert behavior.can_execute(features) is True
    
    def test_behavior_complexity_default_one(self):
        """get_complexity() should default to 1.0."""
        behavior = SimpleBehavior()
        assert behavior.get_complexity() == 1.0
    
    def test_behavior_get_stats_returns_dict(self):
        """get_stats() should return a dictionary with statistics."""
        behavior = SimpleBehavior("stats_test")
        features = np.array([1.0], dtype=np.float64)
        
        behavior.execute(features)
        stats = behavior.get_stats()
        
        assert 'name' in stats
        assert stats['name'] == "stats_test"
        assert 'execution_count' in stats
        assert stats['execution_count'] == 1
        assert 'avg_execution_time' in stats
    
    def test_behavior_hooks_are_callable(self):
        """Hook methods should be callable without error."""
        behavior = SimpleBehavior()
        features = np.array([1.0], dtype=np.float64)
        
        # These should not raise exceptions
        behavior.before_execute(features)
        behavior.after_execute("result", 0.1)
        behavior.on_error(ValueError("test"))
    
    def test_behavior_reset_stats(self):
        """reset_stats() should reset execution statistics."""
        behavior = CountingBehavior()
        features = np.array([1.0], dtype=np.float64)
        
        behavior.execute(features)
        behavior.execute(features)
        assert behavior._execution_count == 2
        
        behavior.reset_stats()
        assert behavior._execution_count == 0


class TestBehaviorExecution:
    """Test behavior execution and error handling."""
    
    def test_behavior_execution_success(self):
        """Successful execution should return result and not raise."""
        behavior = SimpleBehavior("success")
        features = np.array([1.0], dtype=np.float64)
        
        result = behavior.execute(features)
        assert result is not None
    
    def test_behavior_execution_failure_raises(self):
        """Failed execution should raise BehaviorExecutionError."""
        behavior = SimpleBehavior("failing", should_fail=True)
        features = np.array([1.0], dtype=np.float64)
        
        with pytest.raises(BehaviorExecutionError, match="execution failed"):
            behavior.execute(features)
    
    def test_behavior_error_recorded_in_stats(self):
        """Error should be recorded in stats."""
        behavior = SimpleBehavior("error_test", should_fail=True)
        features = np.array([1.0], dtype=np.float64)
        
        try:
            behavior.execute(features)
        except BehaviorExecutionError:
            pass
        
        stats = behavior.get_stats()
        assert stats['last_error'] is not None
        assert "Intentional failure" in stats['last_error']


class TestCustomBehavior:
    """Test custom behavior implementations."""
    
    def test_behavior_with_state(self):
        """Behavior can maintain internal state."""
        
        class StatefulBehavior(MetaBehavior):
            def __init__(self):
                super().__init__()
                self._name = "stateful"
                self._history = []
            
            @property
            def name(self) -> str:
                return self._name
            
            def execute(self, features: FeatureVector, **kwargs) -> list:
                self._history.append(features.tolist())
                return self._history.copy()
        
        behavior = StatefulBehavior()
        features1 = np.array([1.0], dtype=np.float64)
        features2 = np.array([2.0], dtype=np.float64)
        
        history = behavior.execute(features1)
        assert len(history) == 1
        
        history = behavior.execute(features2)
        assert len(history) == 2
    
    def test_behavior_with_can_execute_condition(self):
        """Behavior can override can_execute() for conditional execution."""
        
        class ConditionalBehavior(MetaBehavior):
            def __init__(self):
                super().__init__()
                self._name = "conditional"
            
            @property
            def name(self) -> str:
                return self._name
            
            def execute(self, features: FeatureVector, **kwargs) -> str:
                return "executed"
            
            def can_execute(self, features: FeatureVector) -> bool:
                # Only execute if first feature > 0
                return features[0] > 0 if len(features) > 0 else False
        
        behavior = ConditionalBehavior()
        features_positive = np.array([1.0], dtype=np.float64)
        features_negative = np.array([-1.0], dtype=np.float64)
        
        assert behavior.can_execute(features_positive) == True
        assert behavior.can_execute(features_negative) == False