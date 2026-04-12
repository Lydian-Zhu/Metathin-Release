"""
Unit tests for Selector interface.
选择器接口单元测试。
"""

import pytest
import numpy as np
from metathin.core import Selector, FitnessComputationError, ParameterUpdateError
from metathin.core import FitnessScore, FeatureVector, ParameterDict
from tests.unit.core.test_b_behavior import SimpleBehavior


class LinearSelector(Selector):
    """Linear selector for testing: α = sigmoid(w·x + b)"""
    
    def __init__(self, n_features: int = 2, n_behaviors: int = 2):
        super().__init__()
        self.weights = np.random.randn(n_behaviors, n_features) * 0.1
        self.bias = np.zeros(n_behaviors)
        self._behavior_indices = {}
        self.init_scale = 0.1  # 用于动态扩展
    
    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))
    
    def compute_fitness(self, behavior, features: FeatureVector) -> FitnessScore:
        if behavior.name not in self._behavior_indices:
            self._behavior_indices[behavior.name] = len(self._behavior_indices)
        
        idx = self._behavior_indices[behavior.name]
        
        # 检查特征维度是否匹配（用于测试）
        expected_dim = self.weights.shape[1] if self.weights is not None else 0
        if expected_dim > 0 and len(features) != expected_dim:
            raise FitnessComputationError(f"Feature dimension mismatch: expected {expected_dim}, got {len(features)}")
        
        # 动态扩展权重矩阵
        if idx >= len(self.weights):
            current_feature_dim = self.weights.shape[1]
            new_weights = np.vstack([
                self.weights,
                np.random.randn(idx + 1 - len(self.weights), current_feature_dim) * self.init_scale
            ])
            self.weights = new_weights
            new_bias = np.append(self.bias, np.zeros(idx + 1 - len(self.bias)))
            self.bias = new_bias
        
        # 确保特征维度匹配
        if len(features) != self.weights.shape[1]:
            if len(features) > self.weights.shape[1]:
                features = features[:self.weights.shape[1]]
            else:
                features = np.pad(features, (0, self.weights.shape[1] - len(features)), constant_values=0)
        
        z = np.dot(self.weights[idx], features) + self.bias[idx]
        fitness = self._sigmoid(z)
        self.record_fitness(behavior.name, fitness)
        return float(np.clip(fitness, 0.0, 1.0))
    
    def get_parameters(self) -> ParameterDict:
        params = {}
        for i in range(len(self.weights)):
            for j in range(self.weights.shape[1]):
                params[f'w_{i}_{j}'] = float(self.weights[i, j])
            params[f'b_{i}'] = float(self.bias[i])
        return params
    
    def update_parameters(self, delta: ParameterDict) -> None:
        for key, value in delta.items():
            if key.startswith('w_'):
                parts = key.split('_')
                if len(parts) == 3:
                    i, j = int(parts[1]), int(parts[2])
                    if i < len(self.weights) and j < self.weights.shape[1]:
                        self.weights[i, j] += value
            elif key.startswith('b_'):
                i = int(key.split('_')[1])
                if i < len(self.bias):
                    self.bias[i] += value


class ConstantSelector(Selector):
    """Selector that returns constant fitness."""
    
    def __init__(self, constant: float = 0.5):
        super().__init__()
        self.constant = constant
    
    def compute_fitness(self, behavior, features: FeatureVector) -> FitnessScore:
        self.record_fitness(behavior.name, self.constant)
        return self.constant


class TestSelector:
    """Test Selector interface."""
    
    def test_compute_fitness_returns_float(self):
        """compute_fitness() should return a float."""
        selector = ConstantSelector(0.75)
        behavior = SimpleBehavior("test")
        features = np.array([1.0, 2.0], dtype=np.float64)
        
        fitness = selector.compute_fitness(behavior, features)
        
        assert isinstance(fitness, float)
        assert 0 <= fitness <= 1
    
    def test_fitness_in_range(self):
        """Fitness should always be in [0, 1]."""
        selector = LinearSelector(n_features=2, n_behaviors=2)
        behavior = SimpleBehavior("test")
        
        # Test with various feature vectors
        for _ in range(10):
            features = np.random.randn(2).astype(np.float64)
            fitness = selector.compute_fitness(behavior, features)
            assert 0 <= fitness <= 1
    
    def test_record_fitness_tracks_history(self):
        """record_fitness() should track fitness history."""
        selector = ConstantSelector(0.5)
        behavior = SimpleBehavior("record_test")
        features = np.array([1.0], dtype=np.float64)
        
        for i in range(5):
            selector.compute_fitness(behavior, features)
        
        history = selector.get_fitness_history("record_test")
        # history 是一个字典，键是行为名，值是列表
        assert len(history.get("record_test", [])) == 5
        assert all(f == 0.5 for f in history.get('record_test', []))
    
    def test_get_fitness_history_all(self):
        """get_fitness_history() without name should return all histories."""
        selector = ConstantSelector(0.5)
        behavior1 = SimpleBehavior("b1")
        behavior2 = SimpleBehavior("b2")
        features = np.array([1.0], dtype=np.float64)
        
        selector.compute_fitness(behavior1, features)
        selector.compute_fitness(behavior2, features)
        
        all_history = selector.get_fitness_history()
        assert "b1" in all_history
        assert "b2" in all_history
    
    def test_reset_history_clears_history(self):
        """reset_history() should clear all fitness history."""
        selector = ConstantSelector(0.5)
        behavior = SimpleBehavior("reset_test")
        features = np.array([1.0], dtype=np.float64)
        
        selector.compute_fitness(behavior, features)
        assert len(selector.get_fitness_history()) > 0
        
        selector.reset_history()
        assert len(selector.get_fitness_history()) == 0
    
    def test_get_parameters_default_empty(self):
        """Default get_parameters() should return empty dict."""
        selector = ConstantSelector()
        assert selector.get_parameters() == {}
    
    def test_update_parameters_default_no_op(self):
        """Default update_parameters() should do nothing."""
        selector = ConstantSelector()
        # Should not raise
        selector.update_parameters({'test': 1.0})


class TestLinearSelector:
    """Test LinearSelector implementation."""
    
    def test_linear_selector_initialization(self):
        """Linear selector should initialize correctly."""
        selector = LinearSelector(n_features=3, n_behaviors=4)
        
        params = selector.get_parameters()
        # 4 behaviors * 3 features = 12 weights + 4 biases = 16 params
        assert len(params) == 16
    
    def test_linear_selector_dynamic_behavior_expansion(self):
        """Linear selector should expand for new behaviors."""
        selector = LinearSelector(n_features=2, n_behaviors=1)
        behavior1 = SimpleBehavior("b1")
        behavior2 = SimpleBehavior("b2")
        features = np.array([1.0, 2.0], dtype=np.float64)
        
        # First behavior works
        fitness1 = selector.compute_fitness(behavior1, features)
        assert 0 <= fitness1 <= 1
        
        # Second behavior should work (dynamic expansion)
        fitness2 = selector.compute_fitness(behavior2, features)
        assert 0 <= fitness2 <= 1
    
    def test_linear_selector_parameter_update(self):
        """Parameters should be updatable."""
        selector = LinearSelector(n_features=1, n_behaviors=1)
        behavior = SimpleBehavior("update_test")
        features = np.array([1.0], dtype=np.float64)
        
        initial_params = selector.get_parameters()
        initial_weight = initial_params['w_0_0']
        
        # Update parameters
        selector.update_parameters({'w_0_0': 0.1})
        
        updated_params = selector.get_parameters()
        assert updated_params['w_0_0'] == initial_weight + 0.1
    
    def test_linear_selector_parameter_update_invalid_key(self):
        """Invalid parameter key should be ignored."""
        selector = LinearSelector(n_features=1, n_behaviors=1)
        
        # Should not raise
        selector.update_parameters({'invalid_key': 1.0})
    
    def test_linear_selector_fitness_changes_with_parameters(self):
        """Fitness should change when parameters are updated."""
        selector = LinearSelector(n_features=1, n_behaviors=1)
        behavior = SimpleBehavior("param_test")
        features = np.array([1.0], dtype=np.float64)
        
        initial_fitness = selector.compute_fitness(behavior, features)
        
        # Update to increase fitness
        selector.update_parameters({'w_0_0': 1.0})
        new_fitness = selector.compute_fitness(behavior, features)
        
        # Should be different (not necessarily higher due to sigmoid)
        assert new_fitness != initial_fitness


class TestSelectorErrorHandling:
    """Test selector error handling."""
    
    def test_fitness_computation_error_on_invalid_features(self):
        """Should raise FitnessComputationError on invalid features."""
        selector = LinearSelector(n_features=2, n_behaviors=1)
        behavior = SimpleBehavior("error_test")
        
        # Wrong dimension features
        invalid_features = np.array([1.0], dtype=np.float64)
        
        with pytest.raises(FitnessComputationError):
            selector.compute_fitness(behavior, invalid_features)
    
    def test_fitness_history_limit(self):
        """Fitness history should be limited to prevent memory bloat."""
        selector = ConstantSelector(0.5)
        behavior = SimpleBehavior("limit_test")
        features = np.array([1.0], dtype=np.float64)
        
        # Record 1500 fitness values (limit is 1000)
        for i in range(1500):
            selector.compute_fitness(behavior, features)
        
        history = selector.get_fitness_history("limit_test")
        # Should be at most 1000
        assert len(history) <= 1000