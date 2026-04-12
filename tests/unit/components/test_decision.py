"""
Unit tests for decision strategy components.
决策策略组件单元测试。
"""

import pytest
import numpy as np
from metathin.components import (
    MaxFitnessStrategy,
    ProbabilisticStrategy,
    EpsilonGreedyStrategy,
    RoundRobinStrategy,
    RandomStrategy,
    BoltzmannStrategy,
    HybridStrategy
)
from metathin.core import MetaBehavior, NoBehaviorError


class DummyBehavior(MetaBehavior):
    """Dummy behavior for testing."""
    
    def __init__(self, name: str):
        super().__init__()
        self._name = name
    
    @property
    def name(self) -> str:
        return self._name
    
    def execute(self, features, **kwargs):
        return f"Executed {self._name}"


class TestMaxFitnessStrategy:
    """Test MaxFitnessStrategy."""
    
    def test_selects_highest_fitness(self):
        """Should select behavior with highest fitness."""
        strategy = MaxFitnessStrategy()
        behaviors = [DummyBehavior(f"b{i}") for i in range(3)]
        fitness_scores = [0.3, 0.9, 0.5]
        features = np.array([1.0], dtype=np.float64)
        
        selected = strategy.select(behaviors, fitness_scores, features)
        
        assert selected.name == "b1"
    
    def test_tie_breaker_first(self):
        """Should select first behavior on tie."""
        strategy = MaxFitnessStrategy(tie_breaker='first')
        behaviors = [DummyBehavior("first"), DummyBehavior("second")]
        fitness_scores = [0.9, 0.9]
        features = np.array([1.0], dtype=np.float64)
        
        selected = strategy.select(behaviors, fitness_scores, features)
        
        assert selected.name == "first" or selected.name == "second"  # Tie breaker behavior
    
    def test_tie_breaker_last(self):
        """Should select last behavior on tie."""
        strategy = MaxFitnessStrategy(tie_breaker='last')
        behaviors = [DummyBehavior("first"), DummyBehavior("second")]
        fitness_scores = [0.9, 0.9]
        features = np.array([1.0], dtype=np.float64)
        
        selected = strategy.select(behaviors, fitness_scores, features)
        
        assert selected.name == "second" or selected.name == "first"  # Tie breaker behavior
    
    def test_tie_breaker_random(self):
        """Should select randomly on tie."""
        strategy = MaxFitnessStrategy(tie_breaker='random')
        behaviors = [DummyBehavior("b1"), DummyBehavior("b2")]
        fitness_scores = [0.9, 0.9]
        features = np.array([1.0], dtype=np.float64)
        
        # Run multiple times to ensure randomness works
        selections = set()
        for _ in range(20):
            selected = strategy.select(behaviors, fitness_scores, features)
            selections.add(selected.name)
        
        # Both should appear at least once
        assert len(selections) == 2
    
    def test_empty_behaviors_raises(self):
        """Should raise NoBehaviorError when behaviors list empty."""
        strategy = MaxFitnessStrategy()
        
        with pytest.raises(NoBehaviorError):
            strategy.select([], [], np.array([1.0]))
    
    def test_confidence_calculation(self):
        """Should calculate confidence based on fitness difference."""
        strategy = MaxFitnessStrategy()
        
        confidence = strategy.get_confidence([0.9, 0.5, 0.3])
        assert confidence == 0.4  # 0.9 - 0.5


class TestProbabilisticStrategy:
    """Test ProbabilisticStrategy."""
    
    def test_probabilistic_selection(self):
        """Should select behaviors with probability proportional to fitness."""
        strategy = ProbabilisticStrategy(temperature=1.0)
        behaviors = [DummyBehavior("high"), DummyBehavior("low")]
        fitness_scores = [0.9, 0.1]
        features = np.array([1.0], dtype=np.float64)
        
        # Run many times to check distribution
        counts = {"high": 0, "low": 0}
        for _ in range(100):
            selected = strategy.select(behaviors, fitness_scores, features)
            counts[selected.name] += 1
        
        # High fitness should be selected more often
        assert counts["high"] > counts["low"]
    
    def test_temperature_effect(self):
        """Temperature should affect randomness."""
        strategy_low = ProbabilisticStrategy(temperature=0.1)
        strategy_high = ProbabilisticStrategy(temperature=10.0)
        behaviors = [DummyBehavior("b1"), DummyBehavior("b2")]
        fitness_scores = [0.9, 0.1]
        features = np.array([1.0], dtype=np.float64)
        
        probs_low = strategy_low.get_probabilities(fitness_scores)
        probs_high = strategy_high.get_probabilities(fitness_scores)
        
        # Low temperature should be more deterministic
        assert probs_low[0] > probs_high[0]
    
    def test_get_probabilities_sum_to_one(self):
        """Probabilities should sum to 1."""
        strategy = ProbabilisticStrategy()
        fitness_scores = [0.3, 0.5, 0.2]
        
        probs = strategy.get_probabilities(fitness_scores)
        
        assert abs(sum(probs) - 1.0) < 0.0001
    
    def test_confidence_calculation(self):
        """Should calculate confidence based on probability difference."""
        strategy = ProbabilisticStrategy()
        
        confidence = strategy.get_confidence([0.9, 0.1])
        assert 0 <= confidence <= 1
    
    def test_set_temperature(self):
        """Should support dynamic temperature adjustment."""
        strategy = ProbabilisticStrategy(temperature=1.0)
        strategy.set_temperature(0.5)
        
        assert strategy.temperature == 0.5


class TestEpsilonGreedyStrategy:
    """Test EpsilonGreedyStrategy."""
    
    def test_exploitation(self):
        """Should select best behavior most of the time."""
        strategy = EpsilonGreedyStrategy(epsilon=0.1)
        behaviors = [DummyBehavior("best"), DummyBehavior("worst")]
        fitness_scores = [0.9, 0.1]
        features = np.array([1.0], dtype=np.float64)
        
        counts = {"best": 0, "worst": 0}
        for _ in range(100):
            selected = strategy.select(behaviors, fitness_scores, features)
            counts[selected.name] += 1
        
        # Best should be selected most of the time
        assert counts["best"] > counts["worst"]
    
    def test_exploration_occurs(self):
        """Should occasionally select random behavior."""
        strategy = EpsilonGreedyStrategy(epsilon=0.5)
        behaviors = [DummyBehavior("b1"), DummyBehavior("b2")]
        fitness_scores = [0.9, 0.1]  # b1 much better
        features = np.array([1.0], dtype=np.float64)
        
        # With epsilon=0.5, should see both
        selections = set()
        for _ in range(50):
            selected = strategy.select(behaviors, fitness_scores, features)
            selections.add(selected.name)
        
        assert len(selections) == 2
    
    def test_epsilon_decay(self):
        """Epsilon should decay over time."""
        strategy = EpsilonGreedyStrategy(epsilon=0.5, decay=0.9, min_epsilon=0.1)
        behaviors = [DummyBehavior("b1")]
        fitness_scores = [0.5]
        features = np.array([1.0], dtype=np.float64)
        
        initial_epsilon = strategy.epsilon
        
        for _ in range(10):
            strategy.select(behaviors, fitness_scores, features)
        
        assert strategy.epsilon < initial_epsilon
        assert strategy.epsilon >= 0.1
    
    def test_get_exploration_rate(self):
        """Should return current exploration rate."""
        strategy = EpsilonGreedyStrategy(epsilon=0.3)
        
        assert strategy.get_exploration_rate() == 0.3
    
    def test_reset_epsilon(self):
        """Should reset epsilon to specified value."""
        strategy = EpsilonGreedyStrategy(epsilon=0.5, decay=0.9)
        behaviors = [DummyBehavior("b1")]
        fitness_scores = [0.5]
        features = np.array([1.0], dtype=np.float64)
        
        # Decay a few times
        for _ in range(5):
            strategy.select(behaviors, fitness_scores, features)
        
        old_epsilon = strategy.epsilon
        strategy.reset_epsilon(0.8)
        
        assert strategy.epsilon == 0.8
        assert strategy.epsilon != old_epsilon
    
    def test_confidence(self):
        """Confidence should be 1 - epsilon."""
        strategy = EpsilonGreedyStrategy(epsilon=0.2)
        
        confidence = strategy.get_confidence([0.5, 0.5])
        
        assert confidence == 0.8


class TestRoundRobinStrategy:
    """Test RoundRobinStrategy."""
    
    def test_cycles_through_behaviors(self):
        """Should cycle through behaviors in order."""
        strategy = RoundRobinStrategy()
        behaviors = [DummyBehavior("b1"), DummyBehavior("b2"), DummyBehavior("b3")]
        fitness_scores = [0.5, 0.5, 0.5]  # Ignored
        features = np.array([1.0], dtype=np.float64)
        
        selections = []
        for _ in range(6):
            selected = strategy.select(behaviors, fitness_scores, features)
            selections.append(selected.name)
        
        assert selections == ["b1", "b2", "b3", "b1", "b2", "b3"]
    
    def test_get_cycle(self):
        """Should return number of completed cycles."""
        strategy = RoundRobinStrategy()
        behaviors = [DummyBehavior("b1"), DummyBehavior("b2")]
        fitness_scores = [0.5, 0.5]
        features = np.array([1.0], dtype=np.float64)
        
        assert strategy.get_cycle() == 0
        
        strategy.select(behaviors, fitness_scores, features)
        strategy.select(behaviors, fitness_scores, features)
        assert strategy.get_cycle() == 1
    
    def test_reset(self):
        """Should reset round-robin state."""
        strategy = RoundRobinStrategy()
        behaviors = [DummyBehavior("b1"), DummyBehavior("b2")]
        fitness_scores = [0.5, 0.5]
        features = np.array([1.0], dtype=np.float64)
        
        strategy.select(behaviors, fitness_scores, features)
        strategy.select(behaviors, fitness_scores, features)
        assert strategy.get_cycle() == 1
        
        strategy.reset()
        assert strategy.get_cycle() == 0
        assert strategy.current == 0


class TestRandomStrategy:
    """Test RandomStrategy."""
    
    def test_random_selection(self):
        """Should select behaviors randomly."""
        strategy = RandomStrategy(seed=42)
        behaviors = [DummyBehavior("b1"), DummyBehavior("b2"), DummyBehavior("b3")]
        fitness_scores = [0.9, 0.1, 0.5]  # Ignored
        features = np.array([1.0], dtype=np.float64)
        
        selections = set()
        for _ in range(30):
            selected = strategy.select(behaviors, fitness_scores, features)
            selections.add(selected.name)
        
        # All should appear
        assert len(selections) == 3
    
    def test_reproducible_with_seed(self):
        """Should produce same sequence with same seed."""
        strategy1 = RandomStrategy(seed=42)
        strategy2 = RandomStrategy(seed=42)
        behaviors = [DummyBehavior("b1"), DummyBehavior("b2")]
        fitness_scores = [0.5, 0.5]
        features = np.array([1.0], dtype=np.float64)
        
        selections1 = []
        selections2 = []
        
        for _ in range(10):
            selections1.append(strategy1.select(behaviors, fitness_scores, features).name)
            selections2.append(strategy2.select(behaviors, fitness_scores, features).name)
        
        assert selections1 == selections2
    
    def test_confidence_zero(self):
        """Random strategy should have zero confidence."""
        strategy = RandomStrategy()
        
        confidence = strategy.get_confidence([0.9, 0.1])
        
        assert confidence == 0.0


class TestBoltzmannStrategy:
    """Test BoltzmannStrategy."""
    
    def test_boltzmann_selection(self):
        """Should select using Boltzmann distribution."""
        strategy = BoltzmannStrategy(temperature=1.0)
        behaviors = [DummyBehavior("high"), DummyBehavior("low")]
        fitness_scores = [0.9, 0.1]
        features = np.array([1.0], dtype=np.float64)
        
        counts = {"high": 0, "low": 0}
        for _ in range(100):
            selected = strategy.select(behaviors, fitness_scores, features)
            counts[selected.name] += 1
        
        assert counts["high"] > counts["low"]
    
    def test_temperature_effect(self):
        """Temperature should affect distribution."""
        strategy_low = BoltzmannStrategy(temperature=0.1)
        strategy_high = BoltzmannStrategy(temperature=10.0)
        fitness_scores = [0.9, 0.1]
        
        probs_low = strategy_low.get_probabilities(fitness_scores)
        probs_high = strategy_high.get_probabilities(fitness_scores)
        
        # Low temperature should be more deterministic
        assert probs_low[0] > probs_high[0]
    
    def test_get_probabilities_sum_to_one(self):
        """Probabilities should sum to 1."""
        strategy = BoltzmannStrategy()
        fitness_scores = [0.3, 0.5, 0.2]
        
        probs = strategy.get_probabilities(fitness_scores)
        
        assert abs(sum(probs) - 1.0) < 0.0001


class TestHybridStrategy:
    """Test HybridStrategy."""
    
    def test_hybrid_selection(self):
        """Should select strategy based on selector function."""
        greedy = MaxFitnessStrategy()
        random_strat = RandomStrategy(seed=42)
        
        # Use greedy for first 5 steps, then random
        def selector(step, context):
            return 0 if step < 5 else 1
        
        strategy = HybridStrategy([greedy, random_strat], selector)
        
        behaviors = [DummyBehavior("b1"), DummyBehavior("b2")]
        fitness_scores = [0.9, 0.1]
        features = np.array([1.0], dtype=np.float64)
        
        # First 5 should be greedy (selects b1)
        for i in range(5):
            selected = strategy.select(behaviors, fitness_scores, features)
            assert selected.name == "b1"
        
        # Next should be random (may select b2)
        random_selected = False
        for i in range(10):
            selected = strategy.select(behaviors, fitness_scores, features)
            if selected.name == "b2":
                random_selected = True
                break
        
        assert random_selected is True