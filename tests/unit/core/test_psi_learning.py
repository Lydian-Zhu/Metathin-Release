"""
Unit tests for LearningMechanism interface.
学习机制接口单元测试。
"""

import pytest
import numpy as np
from metathin.core import LearningMechanism, LearningError, ParameterDict


class GradientDescentLearning(LearningMechanism):
    """Simple gradient descent learning for testing."""
    
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self._update_count = 0
    
    def compute_adjustment(self, expected, actual, context) -> ParameterDict:
        # Convert to float if possible
        try:
            exp_val = float(expected)
            act_val = float(actual)
        except (TypeError, ValueError):
            # Non-numeric, treat as match/mismatch
            error = 0.0 if expected == actual else 1.0
        else:
            error = exp_val - act_val
        
        params = context.get('parameters', {})
        features = context.get('features', np.array([1.0]))
        
        adjustments = {}
        for key, value in params.items():
            if key.startswith('w_'):
                # Simple gradient: -2 * error * feature
                parts = key.split('_')
                if len(parts) >= 3:
                    feat_idx = int(parts[2]) % len(features)
                    grad = -2 * error * features[feat_idx]
                    adjustments[key] = self.learning_rate * grad
            elif key.startswith('b_'):
                adjustments[key] = self.learning_rate * (-2 * error)
        
        self._update_count += 1
        return adjustments
    
    def should_learn(self, expected, actual):
        # Only learn if error is significant
        try:
            return abs(float(expected) - float(actual)) > 0.01
        except (TypeError, ValueError):
            return expected != actual
    
    def get_stats(self):
        return {
            'name': 'GradientDescentLearning',
            'type': 'learning_mechanism',
            'update_count': self._update_count
        }


class NoOpLearning(LearningMechanism):
    """Learning mechanism that does nothing."""
    
    def compute_adjustment(self, expected, actual, context):
        return None


class TestLearningMechanism:
    """Test LearningMechanism interface."""
    
    def test_compute_adjustment_returns_dict_or_none(self):
        """compute_adjustment() should return dict or None."""
        learner = NoOpLearning()
        adjustment = learner.compute_adjustment(1.0, 2.0, {})
        
        assert adjustment is None
    
    def test_compute_adjustment_with_parameters(self):
        """compute_adjustment() should return parameter adjustments."""
        learner = GradientDescentLearning(learning_rate=0.1)
        context = {
            'parameters': {'w_0_0': 0.5, 'b_0': 0.0},
            'features': np.array([1.0, 2.0], dtype=np.float64)
        }
        
        adjustment = learner.compute_adjustment(1.0, 0.5, context)
        
        assert isinstance(adjustment, dict)
        assert 'w_0_0' in adjustment
        assert 'b_0' in adjustment
    
    def test_adjustment_values_are_floats(self):
        """Adjustment values should be floats."""
        learner = GradientDescentLearning(learning_rate=0.1)
        context = {
            'parameters': {'w_0_0': 0.5},
            'features': np.array([1.0], dtype=np.float64)
        }
        
        adjustment = learner.compute_adjustment(1.0, 0.5, context)
        
        for value in adjustment.values():
            assert isinstance(value, float)
    
    def test_should_learn_default_true(self):
        """Default should_learn() should return True."""
        learner = NoOpLearning()
        assert learner.should_learn(1.0, 2.0) is True
    
    def test_custom_should_learn_condition(self):
        """Custom should_learn() can implement conditions."""
        learner = GradientDescentLearning()
        
        # Small error -> should not learn
        assert learner.should_learn(1.0, 0.999) is False
        
        # Large error -> should learn
        assert learner.should_learn(1.0, 0.5) is True
    
    def test_get_stats_returns_dict(self):
        """get_stats() should return statistics."""
        learner = GradientDescentLearning()
        stats = learner.get_stats()
        
        assert 'name' in stats
        assert 'type' in stats
        assert stats['type'] == 'learning_mechanism'


class TestGradientDescentLearning:
    """Test gradient descent learning implementation."""
    
    def test_gradient_descent_updates_weights(self):
        """Gradient descent should compute appropriate weight updates."""
        learner = GradientDescentLearning(learning_rate=0.1)
        
        # Error = 1.0 - 0.5 = 0.5
        # Gradient for w_0_0 = -2 * error * feature = -2 * 0.5 * 2.0 = -2.0
        # Adjustment = learning_rate * gradient = 0.1 * (-2.0) = -0.2
        context = {
            'parameters': {'w_0_0': 0.5},
            'features': np.array([2.0], dtype=np.float64)
        }
        
        adjustment = learner.compute_adjustment(1.0, 0.5, context)
        
        assert adjustment['w_0_0'] == -0.2
    
    def test_gradient_descent_with_multiple_features(self):
        """Gradient descent should work with multiple features."""
        learner = GradientDescentLearning(learning_rate=0.1)
        
        context = {
            'parameters': {'w_0_0': 0.5, 'w_0_1': -0.3},
            'features': np.array([1.0, 2.0], dtype=np.float64)
        }
        
        adjustment = learner.compute_adjustment(1.0, 0.5, context)
        
        assert 'w_0_0' in adjustment
        assert 'w_0_1' in adjustment
    
    def test_gradient_descent_non_numeric_values(self):
        """Gradient descent should handle non-numeric values."""
        learner = GradientDescentLearning(learning_rate=0.1)
        context = {'parameters': {'w_0_0': 0.5}}
        
        # Expected and actual are strings
        adjustment = learner.compute_adjustment("apple", "banana", context)
        
        # Error should be 1.0 (mismatch)
        # Should still produce an adjustment
        assert adjustment is not None
    
    def test_gradient_descent_tracks_updates(self):
        """Gradient descent should track update count."""
        learner = GradientDescentLearning()
        context = {'parameters': {'w_0_0': 0.5}}
        
        assert learner.get_stats()['update_count'] == 0
        
        learner.compute_adjustment(1.0, 0.5, context)
        assert learner.get_stats()['update_count'] == 1
        
        learner.compute_adjustment(1.0, 0.5, context)
        assert learner.get_stats()['update_count'] == 2


class TestCustomLearningMechanism:
    """Test custom learning mechanism implementations."""
    
    def test_reward_based_learning(self):
        """Test reward-based learning (reinforcement learning)."""
        
        class RewardLearning(LearningMechanism):
            def __init__(self, learning_rate: float = 0.1):
                self.learning_rate = learning_rate
                self.reward_history = []
            
            def compute_adjustment(self, expected, actual, context):
                # Use reward from context instead of error
                reward = context.get('reward', 0.0)
                self.reward_history.append(reward)
                
                params = context.get('parameters', {})
                adjustments = {}
                
                for key in params:
                    if key.startswith('w_'):
                        # Increase weight for positive reward, decrease for negative
                        adjustments[key] = self.learning_rate * reward
                
                return adjustments
            
            def get_average_reward(self):
                if not self.reward_history:
                    return 0.0
                return sum(self.reward_history) / len(self.reward_history)
        
        learner = RewardLearning(learning_rate=0.1)
        context = {'parameters': {'w_0_0': 0.5, 'w_0_1': -0.2}, 'reward': 0.8}
        
        adjustment = learner.compute_adjustment(None, None, context)
        
        assert adjustment['w_0_0'] == pytest.approx(0.08)
        assert adjustment['w_0_1'] == pytest.approx(0.08)
        assert learner.get_average_reward() == pytest.approx(0.8)
    
    def test_hebbian_learning(self):
        """Test Hebbian learning (unsupervised)."""
        
        class HebbianLearning(LearningMechanism):
            def __init__(self, learning_rate: float = 0.01):
                self.learning_rate = learning_rate
            
            def _activation(self, x: float) -> float:
                return 1.0 / (1.0 + np.exp(-x))
            
            def compute_adjustment(self, expected, actual, context):
                features = context.get('features', np.array([1.0]))
                params = context.get('parameters', {})
                
                # Use actual as post-synaptic activation
                post = self._activation(float(actual) if isinstance(actual, (int, float)) else 0.5)
                
                adjustments = {}
                for key, value in params.items():
                    if key.startswith('w_'):
                        parts = key.split('_')
                        if len(parts) >= 3:
                            feat_idx = int(parts[2]) % len(features)
                            pre = self._activation(features[feat_idx])
                            # Hebbian rule: Δw = η * pre * post
                            adjustments[key] = self.learning_rate * pre * post
                
                return adjustments
        
        learner = HebbianLearning(learning_rate=0.1)
        context = {
            'parameters': {'w_0_0': 0.5},
            'features': np.array([2.0], dtype=np.float64),
            'actual': 1.0
        }
        
        adjustment = learner.compute_adjustment(None, 1.0, context)
        
        # pre = sigmoid(2.0) ≈ 0.88, post = sigmoid(1.0) ≈ 0.73
        # Δw ≈ 0.1 * 0.88 * 0.73 ≈ 0.064
        assert 0.06 < adjustment['w_0_0'] < 0.07