"""
Unit tests for behavior library components.
行为库组件单元测试。
"""

import pytest
import numpy as np
import time
from metathin.components import (
    FunctionBehavior,
    LambdaBehavior,
    CompositeBehavior,
    RetryBehavior,
    TimeoutBehavior,
    ConditionalBehavior,
    CachedBehavior
)


class TestFunctionBehavior:
    """Test FunctionBehavior."""
    
    def test_basic_execution(self):
        """Should execute the provided function."""
        behavior = FunctionBehavior("test", lambda f, **k: f"Result: {f[0]}")
        
        features = np.array([42.0], dtype=np.float64)
        result = behavior.execute(features)
        
        assert result == "Result: 42.0"
    
    def test_name_property(self):
        """Should return the behavior name."""
        behavior = FunctionBehavior("my_behavior", lambda f, **k: None)
        
        assert behavior.name == "my_behavior"
    
    def test_complexity(self):
        """Should return specified complexity."""
        behavior = FunctionBehavior("test", lambda f, **k: None, complexity=2.5)
        
        assert behavior.get_complexity() == 2.5
    
    def test_description(self):
        """Should store description."""
        behavior = FunctionBehavior(
            "test",
            lambda f, **k: None,
            description="Test behavior description"
        )
        
        assert behavior.description == "Test behavior description"
    
    def test_tracks_execution_count(self):
        """Should track execution count."""
        behavior = FunctionBehavior("test", lambda f, **k: "result")
        features = np.array([1.0], dtype=np.float64)
        
        assert behavior._execution_count == 0
        
        behavior.execute(features)
        assert behavior._execution_count == 1
        
        behavior.execute(features)
        assert behavior._execution_count == 2
    
    def test_passes_kwargs(self):
        """Should pass keyword arguments to function."""
        behavior = FunctionBehavior("test", lambda f, **k: k.get('custom', 'not_passed'))
        
        features = np.array([1.0], dtype=np.float64)
        result = behavior.execute(features, custom="passed_value")
        
        assert result == "passed_value"


class TestLambdaBehavior:
    """Test LambdaBehavior."""
    
    def test_basic_execution(self):
        """Should execute lambda function."""
        behavior = LambdaBehavior("test", lambda f, **k: f[0] * 2)
        
        features = np.array([21.0], dtype=np.float64)
        result = behavior.execute(features)
        
        assert result == 42.0
    
    def test_name_property(self):
        """Should return behavior name."""
        behavior = LambdaBehavior("lambda_test", lambda f, **k: None)
        
        assert behavior.name == "lambda_test"


class TestCompositeBehavior:
    """Test CompositeBehavior."""
    
    def test_sequential_execution(self):
        """Should execute behaviors in sequence."""
        results = []
        
        def step1(f, **k):
            results.append(1)
            return np.array([10.0], dtype=np.float64)
        
        def step2(f, **k):
            results.append(2)
            return "final"
        
        b1 = FunctionBehavior("step1", step1)
        b2 = FunctionBehavior("step2", step2)
        
        composite = CompositeBehavior("pipeline", [b1, b2])
        
        features = np.array([0.0], dtype=np.float64)
        result = composite.execute(features)
        
        assert results == [1, 2]
        assert result == "final"
    
    def test_passes_data_between_steps(self):
        """Should pass data between steps."""
        def add_one(f, **k):
            return np.array([f[0] + 1], dtype=np.float64)
        
        def multiply_two(f, **k):
            return f[0] * 2
        
        b1 = FunctionBehavior("add", add_one)
        b2 = FunctionBehavior("multiply", multiply_two)
        
        composite = CompositeBehavior("pipeline", [b1, b2])
        
        features = np.array([5.0], dtype=np.float64)
        result = composite.execute(features)
        
        # 5 + 1 = 6, 6 * 2 = 12
        assert result == 12
    
    def test_stop_on_error(self):
        """Should stop on error when stop_on_error=True."""
        def failing(f, **k):
            raise ValueError("Step failed")
        
        def succeeding(f, **k):
            return "should not reach"
        
        b1 = FunctionBehavior("fail", failing)
        b2 = FunctionBehavior("success", succeeding)
        
        composite = CompositeBehavior("pipeline", [b1, b2], stop_on_error=True)
        
        features = np.array([0.0], dtype=np.float64)
        
        with pytest.raises(Exception, match="failed"):
            composite.execute(features)
    
    def test_continue_on_error(self):
        """Should continue on error when stop_on_error=False."""
        results = []
        
        def failing(f, **k):
            results.append("fail")
            raise ValueError("Step failed")
        
        def succeeding(f, **k):
            results.append("success")
            return "done"
        
        b1 = FunctionBehavior("fail", failing)
        b2 = FunctionBehavior("success", succeeding)
        
        composite = CompositeBehavior("pipeline", [b1, b2], stop_on_error=False)
        
        features = np.array([0.0], dtype=np.float64)
        result = composite.execute(features)
        
        assert results == ["fail", "success"]
        assert result == "done"
    
    def test_add_behavior(self):
        """Should support adding behaviors dynamically."""
        b1 = FunctionBehavior("b1", lambda f, **k: 1)
        b2 = FunctionBehavior("b2", lambda f, **k: 2)
        
        composite = CompositeBehavior("pipeline", [b1])
        composite.add_behavior(b2)
        
        features = np.array([0.0], dtype=np.float64)
        result = composite.execute(features)
        
        assert result == 2  # Last behavior's result
    
    def test_insert_behavior(self):
        """Should support inserting behaviors at position."""
        b1 = FunctionBehavior("b1", lambda f, **k: 1)
        b2 = FunctionBehavior("b2", lambda f, **k: 2)
        b3 = FunctionBehavior("b3", lambda f, **k: 3)
        
        composite = CompositeBehavior("pipeline", [b1, b3])
        composite.insert_behavior(1, b2)
        
        features = np.array([0.0], dtype=np.float64)
        result = composite.execute(features)
        
        assert result == 3  # b3's result
    
    def test_remove_behavior(self):
        """Should support removing behaviors."""
        b1 = FunctionBehavior("b1", lambda f, **k: 1)
        b2 = FunctionBehavior("b2", lambda f, **k: 2)
        
        composite = CompositeBehavior("pipeline", [b1, b2])
        composite.remove_behavior("b1")
        
        features = np.array([0.0], dtype=np.float64)
        result = composite.execute(features)
        
        assert result == 2
    
    def test_get_step_results(self):
        """Should return step execution results."""
        b1 = FunctionBehavior("step1", lambda f, **k: "result1")
        b2 = FunctionBehavior("step2", lambda f, **k: "result2")
        
        composite = CompositeBehavior("pipeline", [b1, b2])
        
        features = np.array([0.0], dtype=np.float64)
        composite.execute(features)
        
        steps = composite.get_step_results()
        assert len(steps) == 2
        assert steps[0]['behavior'] == "step1"
        assert steps[0]['result'] == "result1"
        assert steps[1]['behavior'] == "step2"
        assert steps[1]['result'] == "result2"


class TestRetryBehavior:
    """Test RetryBehavior."""
    
    def test_success_on_first_try(self):
        """Should succeed without retry."""
        call_count = 0
        
        def sometimes_fail(f, **k):
            nonlocal call_count
            call_count += 1
            return "success"
        
        behavior = FunctionBehavior("test", sometimes_fail)
        retry = RetryBehavior("retry_test", behavior, max_retries=3)
        
        features = np.array([0.0], dtype=np.float64)
        result = retry.execute(features)
        
        assert call_count == 1
        assert result == "success"
    
    def test_retry_on_failure(self):
        """Should retry on failure."""
        call_count = 0
        
        def fails_twice(f, **k):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError(f"Attempt {call_count} failed")
            return "success"
        
        behavior = FunctionBehavior("test", fails_twice)
        retry = RetryBehavior("retry_test", behavior, max_retries=3, delay=0.01)
        
        features = np.array([0.0], dtype=np.float64)
        result = retry.execute(features)
        
        assert call_count == 3
        assert result == "success"
    
    def test_max_retries_exceeded(self):
        """Should raise after max retries exceeded."""
        def always_fail(f, **k):
            raise ValueError("Always fails")
        
        behavior = FunctionBehavior("test", always_fail)
        retry = RetryBehavior("retry_test", behavior, max_retries=2, delay=0.01)
        
        features = np.array([0.0], dtype=np.float64)
        
        with pytest.raises(Exception, match="Failed after 2 retries"):
            retry.execute(features)
    
    def test_retry_on_specific_exceptions(self):
        """Should only retry on specified exceptions."""
        call_count = 0
        
        def raises_type_error(f, **k):
            nonlocal call_count
            call_count += 1
            raise TypeError("Type error")
        
        behavior = FunctionBehavior("test", raises_type_error)
        retry = RetryBehavior(
            "retry_test",
            behavior,
            max_retries=3,
            retry_on_exceptions=[ValueError]  # Only retry on ValueError
        )
        
        features = np.array([0.0], dtype=np.float64)
        
        # RetryBehavior wraps exceptions in BehaviorExecutionError
        with pytest.raises(Exception):
            retry.execute(features)
        
        # Should not retry (TypeError not in list)
        assert call_count == 1
    
    def test_exponential_backoff(self):
        """Should implement exponential backoff."""
        call_count = 0
        start = time.time()
        
        def fail(f, **k):
            nonlocal call_count
            call_count += 1
            raise ValueError("Fail")
        
        behavior = FunctionBehavior("test", fail)
        retry = RetryBehavior(
            "retry_test",
            behavior,
            max_retries=3,
            delay=0.05,
            backoff_factor=2.0
        )
        
        features = np.array([0.0], dtype=np.float64)
        
        with pytest.raises(Exception):
            retry.execute(features)
        
        elapsed = time.time() - start
        # Should have waited: 0.05 + 0.10 + 0.20 = 0.35 seconds
        assert elapsed >= 0.35


class TestTimeoutBehavior:
    """Test TimeoutBehavior."""
    
    def test_completes_before_timeout(self):
        """Should return result when completes before timeout."""
        def slow(f, **k):
            time.sleep(0.01)
            return "done"
        
        behavior = FunctionBehavior("slow", slow)
        timeout_behavior = TimeoutBehavior("timeout_test", behavior, timeout=0.1)
        
        features = np.array([0.0], dtype=np.float64)
        result = timeout_behavior.execute(features)
        
        assert result == "done"
    
    def test_timeout_occurs(self):
        """Should raise timeout when execution exceeds timeout."""
        def very_slow(f, **k):
            time.sleep(0.2)
            return "done"
        
        behavior = FunctionBehavior("slow", very_slow)
        timeout_behavior = TimeoutBehavior("timeout_test", behavior, timeout=0.05)
        
        features = np.array([0.0], dtype=np.float64)
        
        with pytest.raises(TimeoutError):
            timeout_behavior.execute(features)
    
    def test_timeout_with_default_result(self):
        """Should return default result on timeout."""
        def very_slow(f, **k):
            time.sleep(0.2)
            return "done"
        
        behavior = FunctionBehavior("slow", very_slow)
        timeout_behavior = TimeoutBehavior(
            "timeout_test",
            behavior,
            timeout=0.05,
            timeout_result="timeout_default"
        )
        
        features = np.array([0.0], dtype=np.float64)
        result = timeout_behavior.execute(features)
        
        assert result == "timeout_default"


class TestConditionalBehavior:
    """Test ConditionalBehavior."""
    
    def test_true_branch_executed(self):
        """Should execute true branch when condition is True."""
        true_executed = False
        false_executed = False
        
        def true_branch(f, **k):
            nonlocal true_executed
            true_executed = True
            return "true"
        
        def false_branch(f, **k):
            nonlocal false_executed
            false_executed = True
            return "false"
        
        true_behavior = FunctionBehavior("true", true_branch)
        false_behavior = FunctionBehavior("false", false_branch)
        
        conditional = ConditionalBehavior(
            "conditional",
            condition=lambda f: f[0] > 0,
            true_behavior=true_behavior,
            false_behavior=false_behavior
        )
        
        features = np.array([5.0], dtype=np.float64)
        result = conditional.execute(features)
        
        assert true_executed is True
        assert false_executed is False
        assert result == "true"
    
    def test_false_branch_executed(self):
        """Should execute false branch when condition is False."""
        true_executed = False
        false_executed = False
        
        def true_branch(f, **k):
            nonlocal true_executed
            true_executed = True
            return "true"
        
        def false_branch(f, **k):
            nonlocal false_executed
            false_executed = True
            return "false"
        
        true_behavior = FunctionBehavior("true", true_branch)
        false_behavior = FunctionBehavior("false", false_branch)
        
        conditional = ConditionalBehavior(
            "conditional",
            condition=lambda f: f[0] > 0,
            true_behavior=true_behavior,
            false_behavior=false_behavior
        )
        
        features = np.array([-5.0], dtype=np.float64)
        result = conditional.execute(features)
        
        assert true_executed is False
        assert false_executed is True
        assert result == "false"
    
    def test_without_false_branch(self):
        """Should return None when condition false and no false branch."""
        true_executed = False
        
        def true_branch(f, **k):
            nonlocal true_executed
            true_executed = True
            return "true"
        
        true_behavior = FunctionBehavior("true", true_branch)
        
        conditional = ConditionalBehavior(
            "conditional",
            condition=lambda f: f[0] > 0,
            true_behavior=true_behavior,
            false_behavior=None
        )
        
        features = np.array([-5.0], dtype=np.float64)
        result = conditional.execute(features)
        
        assert true_executed is False
        assert result is None
    
    def test_execution_stats(self):
        """Should track execution statistics."""
        true_behavior = FunctionBehavior("true", lambda f, **k: "true")
        false_behavior = FunctionBehavior("false", lambda f, **k: "false")
        
        conditional = ConditionalBehavior(
            "conditional",
            condition=lambda f: f[0] > 0,
            true_behavior=true_behavior,
            false_behavior=false_behavior
        )
        
        features = np.array([5.0], dtype=np.float64)
        conditional.execute(features)
        conditional.execute(features)
        
        features = np.array([-5.0], dtype=np.float64)
        conditional.execute(features)
        
        stats = conditional.get_execution_stats()
        
        assert stats['total_executions'] == 3
        assert stats['true_branch'] == 2
        assert stats['false_branch'] == 1
        assert stats['true_ratio'] == 2/3


class TestCachedBehavior:
    """Test CachedBehavior."""
    
    def test_cache_hit(self):
        """Should return cached result on repeated calls."""
        call_count = 0
        
        def compute(f, **k):
            nonlocal call_count
            call_count += 1
            return f[0] * 2
        
        behavior = FunctionBehavior("compute", compute)
        cached = CachedBehavior("cached", behavior, cache_size=10)
        
        features = np.array([21.0], dtype=np.float64)
        
        result1 = cached.execute(features)
        result2 = cached.execute(features)
        
        assert call_count == 1
        assert result1 == result2 == 42.0
    
    def test_cache_miss_different_inputs(self):
        """Should compute on cache miss for different inputs."""
        call_count = 0
        
        def compute(f, **k):
            nonlocal call_count
            call_count += 1
            return f[0] * 2
        
        behavior = FunctionBehavior("compute", compute)
        cached = CachedBehavior("cached", behavior, cache_size=10)
        
        features1 = np.array([10.0], dtype=np.float64)
        features2 = np.array([20.0], dtype=np.float64)
        
        cached.execute(features1)
        cached.execute(features2)
        
        assert call_count == 2
    
    def test_cache_size_limit(self):
        """Should respect cache size limit (LRU eviction)."""
        call_count = 0
        
        def compute(f, **k):
            nonlocal call_count
            call_count += 1
            return f[0]
        
        behavior = FunctionBehavior("compute", compute)
        cached = CachedBehavior("cached", behavior, cache_size=2)
        
        cached.execute(np.array([1.0], dtype=np.float64))
        cached.execute(np.array([2.0], dtype=np.float64))
        cached.execute(np.array([3.0], dtype=np.float64))  # Evicts one
        
        assert call_count == 3
    
    def test_clear_cache(self):
        """Should clear all cached results."""
        call_count = 0
        
        def compute(f, **k):
            nonlocal call_count
            call_count += 1
            return f[0]
        
        behavior = FunctionBehavior("compute", compute)
        cached = CachedBehavior("cached", behavior, cache_size=10)
        
        features = np.array([42.0], dtype=np.float64)
        
        cached.execute(features)
        cached.execute(features)
        assert call_count == 1
        
        cached.clear_cache()
        
        cached.execute(features)
        assert call_count == 2
    
    def test_cache_stats(self):
        """Should provide cache statistics."""
        behavior = FunctionBehavior("compute", lambda f, **k: f[0])
        cached = CachedBehavior("cached", behavior, cache_size=10)
        
        features1 = np.array([1.0], dtype=np.float64)
        features2 = np.array([2.0], dtype=np.float64)
        
        cached.execute(features1)
        cached.execute(features1)  # hit
        cached.execute(features2)
        cached.execute(features2)  # hit
        
        stats = cached.get_cache_stats()
        
        assert stats['hits'] == 2
        assert stats['misses'] == 2
        assert stats['hit_rate'] == 0.5
        assert stats['size'] == 2
        assert stats['capacity'] == 10