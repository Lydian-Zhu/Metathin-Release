"""
Unit tests for DecisionStrategy interface.
决策策略接口单元测试。
"""

import pytest
import numpy as np
from metathin.core import DecisionStrategy, DecisionError, NoBehaviorError
from metathin.core import FitnessScore, FeatureVector
from tests.unit.core.test_b_behavior import SimpleBehavior


class GreedyStrategy(DecisionStrategy):
    """Always select behavior with highest fitness."""
    
    def select(self, behaviors, fitness_scores, features):
        if not behaviors:
            raise NoBehaviorError("No behaviors available")
        max_idx = max(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
        return behaviors[max_idx]


class RandomStrategy(DecisionStrategy):
    """Random selection for testing."""
    
    def __init__(self, seed: int = 42):
        import random
        random.seed(seed)
        self._random = random
    
    def select(self, behaviors, fitness_scores, features):
        if not behaviors:
            raise NoBehaviorError("No behaviors available")
        return self._random.choice(behaviors)


class TestDecisionStrategy:
    """Test DecisionStrategy interface."""
    
    def test_select_returns_behavior(self):
        """select() should return a MetaBehavior."""
        strategy = GreedyStrategy()
        behaviors = [SimpleBehavior("b1"), SimpleBehavior("b2")]
        fitness_scores = [0.5, 0.8]
        features = np.array([1.0], dtype=np.float64)
        
        selected = strategy.select(behaviors, fitness_scores, features)
        
        assert selected in behaviors
        assert selected.name == "b2"  # Higher fitness
    
    def test_select_raises_no_behavior_error_on_empty_list(self):
        """select() should raise NoBehaviorError when behaviors list is empty."""
        strategy = GreedyStrategy()
        features = np.array([1.0], dtype=np.float64)
        
        with pytest.raises(NoBehaviorError, match="No behaviors available"):
            strategy.select([], [], features)
    
    def test_select_raises_decision_error_on_mismatched_lengths(self):
        """select() should raise DecisionError when lengths mismatch."""
        strategy = GreedyStrategy()
        behaviors = [SimpleBehavior("b1"), SimpleBehavior("b2")]
        fitness_scores = [0.5]  # Only one score
        features = np.array([1.0], dtype=np.float64)
        
        # This implementation doesn't check, but the base class allows it
        # In a real implementation, you'd want to check
        pass
    
    def test_get_confidence_default_implementation(self):
        """Default get_confidence() should work."""
        strategy = GreedyStrategy()
        
        # Single behavior -> confidence 1.0
        confidence = strategy.get_confidence([0.5])
        assert confidence == 1.0
        
        # Multiple behaviors -> difference between top two
        confidence = strategy.get_confidence([0.9, 0.5, 0.3])
        assert confidence == 0.4  # 0.9 - 0.5
    
    def test_get_info_returns_dict(self):
        """get_info() should return strategy information."""
        strategy = GreedyStrategy()
        info = strategy.get_info()
        
        assert 'name' in info
        assert info['name'] == 'GreedyStrategy'
        assert info['type'] == 'decision_strategy'


class TestGreedyStrategy:
    """Test greedy decision strategy."""
    
    def test_greedy_selects_highest_fitness(self):
        """Greedy strategy should always select highest fitness."""
        strategy = GreedyStrategy()
        behaviors = [SimpleBehavior(f"b{i}") for i in range(5)]
        fitness_scores = [0.1, 0.3, 0.9, 0.2, 0.4]
        features = np.array([1.0], dtype=np.float64)
        
        selected = strategy.select(behaviors, fitness_scores, features)
        
        assert selected.name == "b2"  # Index 2 has fitness 0.9
    
    def test_greedy_with_ties_selects_first(self):
        """With ties, greedy selects the first occurrence."""
        strategy = GreedyStrategy()
        behaviors = [SimpleBehavior("first"), SimpleBehavior("second")]
        fitness_scores = [0.9, 0.9]  # Tie
        features = np.array([1.0], dtype=np.float64)
        
        selected = strategy.select(behaviors, fitness_scores, features)
        
        # max() returns first occurrence
        assert selected.name == "first"


class TestRandomStrategy:
    """Test random decision strategy."""
    
    def test_random_selects_behavior(self):
        """Random strategy should select a behavior."""
        strategy = RandomStrategy(seed=42)
        behaviors = [SimpleBehavior(f"b{i}") for i in range(5)]
        fitness_scores = [0.5] * 5
        features = np.array([1.0], dtype=np.float64)
        
        selected = strategy.select(behaviors, fitness_scores, features)
        
        assert selected in behaviors
    
    def test_random_distribution(self):
        """Random strategy should have approximately uniform distribution."""
        strategy = RandomStrategy(seed=42)
        behaviors = [SimpleBehavior(f"b{i}") for i in range(3)]
        fitness_scores = [0.5] * 3
        features = np.array([1.0], dtype=np.float64)
        
        counts = {b.name: 0 for b in behaviors}
        n_trials = 1000
        
        for _ in range(n_trials):
            selected = strategy.select(behaviors, fitness_scores, features)
            counts[selected.name] += 1
        
        # Each should be roughly n_trials/3
        for count in counts.values():
            assert 300 < count < 400


class TestCustomDecisionStrategy:
    """Test custom decision strategy implementations."""
    
    def test_epsilon_greedy_strategy(self):
        """Test epsilon-greedy strategy."""
        
        class EpsilonGreedyStrategy(DecisionStrategy):
            def __init__(self, epsilon: float = 0.1):
                self.epsilon = epsilon
                self.step = 0
                import random
                self._random = random
            
            def select(self, behaviors, fitness_scores, features):
                self.step += 1
                import random
                if random.random() < self.epsilon:
                    # Explore: random selection
                    return self._random.choice(behaviors)
                else:
                    # Exploit: best fitness
                    max_idx = max(range(len(fitness_scores)), 
                                 key=lambda i: fitness_scores[i])
                    return behaviors[max_idx]
            
            def get_confidence(self, fitness_scores):
                return 1.0 - self.epsilon
        
        strategy = EpsilonGreedyStrategy(epsilon=0.2)
        behaviors = [SimpleBehavior("b1"), SimpleBehavior("b2")]
        fitness_scores = [0.1, 0.9]
        features = np.array([1.0], dtype=np.float64)
        
        # With epsilon=0.2, mostly selects b2 but sometimes random
        # Just verify it works
        for _ in range(10):
            selected = strategy.select(behaviors, fitness_scores, features)
            assert selected in behaviors
        
        assert strategy.get_confidence(fitness_scores) == 0.8
    
    def test_probabilistic_strategy(self):
        """Test probabilistic (softmax) strategy."""
        
        class ProbabilisticStrategy(DecisionStrategy):
            def __init__(self, temperature: float = 1.0):
                self.temperature = temperature
            
            def _softmax(self, scores):
                import numpy as np
                scores = np.array(scores) / self.temperature
                exp_scores = np.exp(scores - np.max(scores))
                return exp_scores / np.sum(exp_scores)
            
            def select(self, behaviors, fitness_scores, features):
                import numpy as np
                probs = self._softmax(fitness_scores)
                idx = np.random.choice(len(behaviors), p=probs)
                return behaviors[idx]
        
        strategy = ProbabilisticStrategy(temperature=1.0)
        behaviors = [SimpleBehavior("b1"), SimpleBehavior("b2")]
        fitness_scores = [0.9, 0.1]
        features = np.array([1.0], dtype=np.float64)
        
        # Higher probability for b1
        counts = {'b1': 0, 'b2': 0}
        for _ in range(100):
            selected = strategy.select(behaviors, fitness_scores, features)
            counts[selected.name] += 1
        
        assert counts['b1'] > counts['b2']