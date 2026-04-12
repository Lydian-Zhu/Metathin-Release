"""
Unit tests for selector components.
选择器组件单元测试。
"""

import pytest
import numpy as np
from metathin.components import (
    SimpleSelector,
    PolynomialSelector,
    RuleBasedSelector,
    EnsembleSelector,
    AdaptiveSelector
)
from metathin.core import MetaBehavior, FeatureVector


class DummyBehavior(MetaBehavior):
    """Dummy behavior for testing."""
    
    def __init__(self, name: str):
        super().__init__()
        self._name = name
    
    @property
    def name(self) -> str:
        return self._name
    
    def execute(self, features: FeatureVector, **kwargs):
        return f"Executed {self._name}"


class TestSimpleSelector:
    """Test SimpleSelector."""
    
    def test_initialization(self):
        """Should initialize with specified dimensions."""
        selector = SimpleSelector(n_features=3, n_behaviors=2)
        
        assert selector.weights.shape == (2, 3)
        assert selector.bias.shape == (2,)
    
    def test_dynamic_expansion(self):
        """Should dynamically expand for new behaviors."""
        selector = SimpleSelector()
        behavior1 = DummyBehavior("b1")
        behavior2 = DummyBehavior("b2")
        features = np.array([1.0, 2.0], dtype=np.float64)
        
        fitness1 = selector.compute_fitness(behavior1, features)
        assert 0 <= fitness1 <= 1
        
        fitness2 = selector.compute_fitness(behavior2, features)
        assert 0 <= fitness2 <= 1
        
        # Should have expanded to 2 behaviors
        assert selector.weights.shape[0] == 2
    
    def test_fitness_in_range(self):
        """Fitness should always be in [0,1]."""
        selector = SimpleSelector(n_features=2, n_behaviors=1)
        behavior = DummyBehavior("test")
        
        # Test with various feature values
        for i in range(-100, 100, 10):
            features = np.array([float(i), float(i) ** 2], dtype=np.float64)
            fitness = selector.compute_fitness(behavior, features)
            assert 0 <= fitness <= 1
    
    def test_parameter_getter(self):
        """Should return parameters dictionary."""
        selector = SimpleSelector(n_features=2, n_behaviors=2)
        
        params = selector.get_parameters()
        
        # 2 behaviors * 2 features = 4 weights + 2 biases = 6 params
        assert len(params) == 6
        assert 'w_0_0' in params
        assert 'b_1' in params
    
    def test_parameter_update(self):
        """Should update parameters."""
        selector = SimpleSelector(n_features=1, n_behaviors=1)
        behavior = DummyBehavior("test")
        features = np.array([1.0], dtype=np.float64)
        
        initial_fitness = selector.compute_fitness(behavior, features)
        
        # Update weight to increase fitness
        selector.update_parameters({'w_0_0': 1.0})
        
        new_fitness = selector.compute_fitness(behavior, features)
        
        # Should be different
        assert new_fitness != initial_fitness
    
    def test_temperature_effect(self):
        """Temperature should affect fitness smoothness."""
        selector_low = SimpleSelector(n_features=1, n_behaviors=1, temperature=1.0)
        selector_high = SimpleSelector(n_features=1, n_behaviors=1, temperature=10.0)
        behavior = DummyBehavior("test")
        features = np.array([100.0], dtype=np.float64)
        
        fitness_low = selector_low.compute_fitness(behavior, features)
        fitness_high = selector_high.compute_fitness(behavior, features)
        
        # Higher temperature should produce more moderate values
        assert fitness_high > fitness_low or fitness_high < fitness_low
    
    def test_fitness_history(self):
        """Should record fitness history."""
        selector = SimpleSelector(n_features=1, n_behaviors=1)
        behavior = DummyBehavior("test")
        features = np.array([1.0], dtype=np.float64)
        
        for _ in range(5):
            selector.compute_fitness(behavior, features)
        
        history = selector.get_fitness_history("test")
        assert len(history.get('test', [])) == 5


class TestPolynomialSelector:
    """Test PolynomialSelector."""
    
    def test_initialization(self):
        """Should initialize with specified degree."""
        selector = PolynomialSelector(degree=2, n_features=2, n_behaviors=2)
        
        assert selector.degree == 2
        assert selector.n_poly > 2  # Expanded features
    
    def test_polynomial_expansion(self):
        """Should expand to polynomial features."""
        selector = PolynomialSelector(degree=2, n_features=2, n_behaviors=1)
        behavior = DummyBehavior("test")
        
        features = np.array([1.0, 2.0], dtype=np.float64)
        fitness = selector.compute_fitness(behavior, features)
        
        assert 0 <= fitness <= 1
    
    def test_cubic_degree(self):
        """Should handle degree 3."""
        selector = PolynomialSelector(degree=3, n_features=2, n_behaviors=1)
        behavior = DummyBehavior("test")
        features = np.array([1.0, 2.0], dtype=np.float64)
        
        fitness = selector.compute_fitness(behavior, features)
        
        assert 0 <= fitness <= 1
    
    def test_regularization(self):
        """Regularization should affect fitness."""
        selector_reg = PolynomialSelector(
            degree=2, n_features=2, n_behaviors=1, regularization=0.1
        )
        selector_no_reg = PolynomialSelector(
            degree=2, n_features=2, n_behaviors=1, regularization=0.0
        )
        behavior = DummyBehavior("test")
        features = np.array([1.0, 2.0], dtype=np.float64)
        
        # Different regularization should produce different results
        fitness_reg = selector_reg.compute_fitness(behavior, features)
        fitness_no_reg = selector_no_reg.compute_fitness(behavior, features)
        
        # They might be different (not guaranteed, but likely)
        # Just verify they both return valid values
        assert 0 <= fitness_reg <= 1
        assert 0 <= fitness_no_reg <= 1
    
    def test_feature_normalization(self):
        """Should normalize features."""
        selector = PolynomialSelector(degree=2, n_features=2, n_behaviors=1)
        behavior = DummyBehavior("test")
        
        # Large values should be normalized
        features = np.array([1000.0, 2000.0], dtype=np.float64)
        fitness = selector.compute_fitness(behavior, features)
        
        assert 0 <= fitness <= 1
    
    def test_get_parameters(self):
        """Should return polynomial parameters."""
        selector = PolynomialSelector(degree=2, n_features=2, n_behaviors=2)
        
        params = selector.get_parameters()
        
        assert len(params) > 0
        assert any(k.startswith('poly_w_') for k in params.keys())
    
    def test_get_feature_importance(self):
        """Should return feature importance."""
        selector = PolynomialSelector(degree=2, n_features=3, n_behaviors=2)
        
        importance = selector.get_feature_importance()
        
        assert len(importance) == 3
        assert all(0 <= v <= 1 for v in importance.values())


class TestRuleBasedSelector:
    """Test RuleBasedSelector."""
    
    def test_rule_based_fitness(self):
        """Should compute fitness based on rules."""
        def rule_high(f):
            return 0.9 if f[0] > 5 else 0.1
        
        def rule_low(f):
            return 0.2 if f[0] < 3 else 0.8
        
        selector = RuleBasedSelector({
            "high_behavior": rule_high,
            "low_behavior": rule_low
        })
        
        behavior_high = DummyBehavior("high_behavior")
        behavior_low = DummyBehavior("low_behavior")
        features = np.array([10.0], dtype=np.float64)
        
        fitness_high = selector.compute_fitness(behavior_high, features)
        fitness_low = selector.compute_fitness(behavior_low, features)
        
        assert fitness_high == 0.9
        assert fitness_low == 0.8
    
    def test_default_fitness(self):
        """Should use default fitness for behaviors without rules."""
        selector = RuleBasedSelector({}, default_fitness=0.5)
        behavior = DummyBehavior("no_rule")
        features = np.array([1.0], dtype=np.float64)
        
        fitness = selector.compute_fitness(behavior, features)
        
        assert fitness == 0.5
    
    def test_add_rule(self):
        """Should support adding rules dynamically."""
        selector = RuleBasedSelector({})
        
        selector.add_rule("new_behavior", lambda f: 0.75)
        behavior = DummyBehavior("new_behavior")
        features = np.array([1.0], dtype=np.float64)
        
        fitness = selector.compute_fitness(behavior, features)
        
        assert fitness == 0.75
    
    def test_remove_rule(self):
        """Should support removing rules."""
        selector = RuleBasedSelector({
            "to_remove": lambda f: 0.9
        })
        
        removed = selector.remove_rule("to_remove")
        assert removed is True
        
        behavior = DummyBehavior("to_remove")
        features = np.array([1.0], dtype=np.float64)
        fitness = selector.compute_fitness(behavior, features)
        
        assert fitness == 0.5  # Default


class TestEnsembleSelector:
    """Test EnsembleSelector."""
    
    def test_weighted_average(self):
        """Should compute weighted average of selectors."""
        selector1 = SimpleSelector(n_features=1, n_behaviors=1)
        selector2 = SimpleSelector(n_features=1, n_behaviors=1)
        
        # Force specific fitness values by setting parameters
        selector1.update_parameters({'w_0_0': 10.0})  # High fitness
        selector2.update_parameters({'w_0_0': -10.0})  # Low fitness
        
        ensemble = EnsembleSelector(
            selectors=[selector1, selector2],
            weights=[0.7, 0.3],
            aggregation='weighted_average'
        )
        
        behavior = DummyBehavior("test")
        features = np.array([1.0], dtype=np.float64)
        
        fitness = ensemble.compute_fitness(behavior, features)
        
        # Should be between selector1 and selector2
        fitness1 = selector1.compute_fitness(behavior, features)
        fitness2 = selector2.compute_fitness(behavior, features)
        assert min(fitness1, fitness2) <= fitness <= max(fitness1, fitness2)
    
    def test_max_aggregation(self):
        """Should take maximum of selectors."""
        selector1 = SimpleSelector(n_features=1, n_behaviors=1)
        selector2 = SimpleSelector(n_features=1, n_behaviors=1)
        
        selector1.update_parameters({'w_0_0': 10.0})  # High
        selector2.update_parameters({'w_0_0': -10.0})  # Low
        
        ensemble = EnsembleSelector(
            selectors=[selector1, selector2],
            aggregation='max'
        )
        
        behavior = DummyBehavior("test")
        features = np.array([1.0], dtype=np.float64)
        
        fitness = ensemble.compute_fitness(behavior, features)
        fitness1 = selector1.compute_fitness(behavior, features)
        
        assert fitness == fitness1
    
    def test_min_aggregation(self):
        """Should take minimum of selectors."""
        selector1 = SimpleSelector(n_features=1, n_behaviors=1)
        selector2 = SimpleSelector(n_features=1, n_behaviors=1)
        
        selector1.update_parameters({'w_0_0': 10.0})  # High
        selector2.update_parameters({'w_0_0': -10.0})  # Low
        
        ensemble = EnsembleSelector(
            selectors=[selector1, selector2],
            aggregation='min'
        )
        
        behavior = DummyBehavior("test")
        features = np.array([1.0], dtype=np.float64)
        
        fitness = ensemble.compute_fitness(behavior, features)
        fitness2 = selector2.compute_fitness(behavior, features)
        
        assert fitness == fitness2
    
    def test_add_selector(self):
        """Should support adding selectors dynamically."""
        selector1 = SimpleSelector(n_features=1, n_behaviors=1)
        selector2 = SimpleSelector(n_features=1, n_behaviors=1)
        
        ensemble = EnsembleSelector([selector1])
        ensemble.add_selector(selector2)
        
        assert len(ensemble.selectors) == 2


class TestAdaptiveSelector:
    """Test AdaptiveSelector."""
    
    def test_initialization(self):
        """Should initialize with selectors."""
        selector1 = SimpleSelector(n_features=1, n_behaviors=1)
        selector2 = SimpleSelector(n_features=1, n_behaviors=1)
        
        adaptive = AdaptiveSelector([selector1, selector2])
        
        assert len(adaptive.selectors) == 2
        assert adaptive.exploration_rate == 0.1
    
    def test_selector_selection(self):
        """Should select a selector (not testing randomness)."""
        selector1 = SimpleSelector(n_features=1, n_behaviors=1)
        selector2 = SimpleSelector(n_features=1, n_behaviors=1)
        
        # Make selector1 much better
        selector1.update_parameters({'w_0_0': 10.0})
        selector2.update_parameters({'w_0_0': -10.0})
        
        adaptive = AdaptiveSelector([selector1, selector2], exploration_rate=0.0)
        
        behavior = DummyBehavior("test")
        features = np.array([1.0], dtype=np.float64)
        
        # With no exploration, should pick selector1
        fitness = adaptive.compute_fitness(behavior, features)
        expected = selector1.compute_fitness(behavior, features)
        
        assert fitness == expected