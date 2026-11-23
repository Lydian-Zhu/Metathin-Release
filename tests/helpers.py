"""Local test doubles for Metathin's interfaces.

These are deliberately tiny and live under ``tests`` only.  The package itself
ships no concrete P/B/S/D/Ψ implementations.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from metathin.core import (
    DecisionStrategy,
    FeatureVector,
    FitnessScore,
    LearningMechanism,
    MetaBehavior,
    ParameterDict,
    PatternSpace,
    Selector,
)


class FunctionBehavior(MetaBehavior):
    """Minimal behavior backed by a callable."""

    def __init__(self, name: str, func: Callable[..., Any], complexity: float = 1.0):
        super().__init__()
        self._name = name
        self._func = func
        self._complexity = complexity

    @property
    def name(self) -> str:
        return self._name

    def execute(self, features: FeatureVector, **kwargs) -> Any:
        return self._func(features, **kwargs)

    def get_complexity(self) -> float:
        return self._complexity


class ListPatternSpace(PatternSpace):
    """Pattern space that wraps a fixed feature extractor."""

    def __init__(self, extractor: Callable[[Any], Any]):
        self._extractor = extractor

    def extract(self, raw_input: Any) -> FeatureVector:
        values = self._extractor(raw_input)
        return np.asarray(values, dtype=np.float64).reshape(-1)


class FixedSelector(Selector):
    """Selector with a fixed score per behavior and simple learnable weights."""

    def __init__(self, scores: Optional[Dict[str, float]] = None):
        super().__init__()
        self._scores = dict(scores or {})
        self._weights: ParameterDict = {'w_0': 0.0}

    def set_fitness(self, behavior_name: str, fitness: float) -> None:
        self._scores[behavior_name] = fitness

    def compute_fitness(self, behavior: MetaBehavior, features: FeatureVector) -> FitnessScore:
        return self._scores.get(behavior.name, 0.5)

    def get_parameters(self) -> ParameterDict:
        return dict(self._weights)

    def update_parameters(self, delta: ParameterDict) -> None:
        for key, value in delta.items():
            if key in self._weights:
                self._weights[key] += value


class FirstBehaviorStrategy(DecisionStrategy):
    """Always pick the first candidate."""

    def select(
        self,
        behaviors: List[MetaBehavior],
        fitness_scores: List[FitnessScore],
        features: FeatureVector,
    ) -> MetaBehavior:
        return behaviors[0]


class MaxBehaviorStrategy(DecisionStrategy):
    """Pick the highest fitness candidate."""

    def select(
        self,
        behaviors: List[MetaBehavior],
        fitness_scores: List[FitnessScore],
        features: FeatureVector,
    ) -> MetaBehavior:
        return behaviors[max(range(len(fitness_scores)), key=lambda i: fitness_scores[i])]


class ConstantLearning(LearningMechanism):
    """Learning mechanism that emits a fixed valid adjustment."""

    def __init__(self, adjustment: Optional[ParameterDict] = None):
        self.adjustment = adjustment if adjustment is not None else {'w_0': 0.1}
        self.calls: List[Dict[str, Any]] = []

    def compute_adjustment(self, expected: Any, actual: Any, context: Dict[str, Any]):
        self.calls.append(dict(context))
        return dict(self.adjustment)
