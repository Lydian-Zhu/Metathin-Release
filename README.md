# Metathin

**An interface-first framework for constructing meta-cognitive agents.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/Lydian-Zhu/Metathin-Release/pulls)

Metathin describes an agent as a five-tuple:

```
Metathin = (P, B, S, D, Ψ)
```

- **P — PatternSpace**: turns raw input into a feature vector.
- **B — MetaBehavior**: an executable skill.
- **S — Selector**: scores each behavior's fitness.
- **D — DecisionStrategy**: chooses a behavior from the candidates.
- **Ψ — LearningMechanism**: adjusts selector parameters from feedback.

This repository deliberately ships **interfaces and orchestration only**. It
does not include built-in selectors, learners, decision strategies, or domain
modules. You bring the algorithms; Metathin owns the contracts, lifecycle,
validation, history, metrics, and memory plumbing.

---

## What is in the box

- The five interfaces and their type/exception contracts.
- The thinking pipeline (`P → S → D → B → Ψ`) with strict runtime validation.
- Configuration loading and validation.
- Optional memory, history, and metrics services.
- A facade and fluent builder that require every core component to be injected.

**Not in the box:**

- concrete `PatternSpace`, `Selector`, `LearningMechanism` implementations;
- concrete decision strategies or behavior wrappers;
- `metathin.components`;
- `metathin_plus` / the old `sci` module and its optional scientific stack.

This is a deliberate breaking change. The framework is meant to be a stable
core that applications and third-party packages build on.

---

## Quick start

Every component is supplied by you. The smallest working agent looks like this:

```python
import numpy as np

from metathin import Metathin, MetathinBuilder
from metathin.core import (
    DecisionStrategy,
    FeatureVector,
    FitnessScore,
    MetaBehavior,
    PatternSpace,
    Selector,
)


class TextFeatures(PatternSpace[str]):
    def extract(self, raw_input: str) -> FeatureVector:
        return np.array([len(raw_input), len(raw_input.split())], dtype=np.float64)

    def get_feature_names(self):
        return ["chars", "words"]


class Greet(MetaBehavior[str]):
    @property
    def name(self) -> str:
        return "greet"

    def execute(self, features: FeatureVector, **kwargs) -> str:
        return f"Hello! ({int(features[0])} chars)"


class Echo(MetaBehavior[str]):
    @property
    def name(self) -> str:
        return "echo"

    def execute(self, features: FeatureVector, **kwargs) -> str:
        return kwargs.get("user_input", "")


class LengthSelector(Selector):
    """Prefer behaviors that suit longer inputs; purely illustrative."""

    def compute_fitness(self, behavior: MetaBehavior, features: FeatureVector) -> FitnessScore:
        if behavior.name == "greet":
            return 0.8 if features[0] < 20 else 0.2
        return 0.5


class Greedy(DecisionStrategy):
    def select(self, behaviors, fitness_scores, features):
        return behaviors[max(range(len(fitness_scores)), key=fitness_scores.__getitem__)]


agent = (
    MetathinBuilder()
    .with_name("Demo")
    .with_pattern_space(TextFeatures())
    .with_behaviors([Greet(), Echo()])
    .with_selector(LengthSelector())
    .with_decision_strategy(Greedy())
    .build()
)

print(agent.think("hello world", user_input="hello world"))
```

Learning is opt-in. Pass a `LearningMechanism` and set `enable_learning=True`:

```python
from metathin.config import MetathinConfig, PipelineConfig

agent = Metathin(
    pattern_space=TextFeatures(),
    selector=LengthSelector(),
    decision_strategy=Greedy(),
    learning_mechanism=MyLearner(),   # a LearningMechanism implementation
    config=MetathinConfig(pipeline=PipelineConfig(enable_learning=True)),
)
```

If learning is enabled without a mechanism, construction fails instead of
silently doing nothing.

---

## Core contracts

`ThinkingPipeline` validates the following at runtime:

- `P`, `S`, `D` and `Ψ` must implement their respective interfaces.
- Every behavior must be a `MetaBehavior` with a unique, non-empty name.
- `PatternSpace.extract()` must return a 1-D `float64` NumPy array with finite values.
- `Selector.compute_fitness()` must return a finite `float` in `[0, 1]`.
- `DecisionStrategy.select()` must return one of the candidates it was given.
- Selector parameters and learning adjustments must be finite numeric mappings;
  learning may only touch keys that `get_parameters()` exposes.
- Invalid learning output is recorded on the thought context; it never corrupts
  the selector or crashes the thought.

This means a buggy custom component produces a clear `PatternExtractionError`,
`FitnessComputationError`, `DecisionError`, or `LearningError` instead of a
silently wrong result.

---

## What is preserved

- **Memory**: JSON, SQLite and in-memory backends, TTL and LRU caching.
- **History**: bounded thought records, success/failure filtering, export.
- **Metrics**: windowed timing and behavior-usage summaries.
- **Configuration**: dataclass validation plus file/dict/environment loading.
- **Builder**: explicit dependency injection, validation, and config overrides.

These are infrastructure, not algorithm implementations. They remain because
the core needs them to be observable and testable.

---

## Installation

```bash
# Core only
pip install git+https://github.com/Lydian-Zhu/Metathin-Release.git

# With YAML configuration support
pip install "metathin[yaml] @ git+https://github.com/Lydian-Zhu/Metathin-Release.git"

# Test tools / contributor tools
pip install -e ".[test]"
pip install -e ".[dev]"
```

Runtime dependency: `numpy`. Python 3.8+.

---

## Project structure

```
metathin/
├── core/       # P, B, S, D, Ψ interfaces, types, exceptions, memory backend
├── engine/     # thinking pipeline, context, hooks
├── services/   # memory manager, history tracker, metrics collector
├── config/     # schema, loader
└── agent/      # Metathin facade and MetathinBuilder
tests/          # interface, pipeline, config, service, and facade tests
```

No `components/` package exists. If you want reusable implementations, publish
them as a separate package that depends on this core.

---

## Status

- Core interfaces and pipeline: complete.
- Runtime contract validation: enforced.
- Concrete algorithms and the former `sci` module: removed in `0.5.0`.
- Test suite: see CI for the current count; the suite uses only local test
  doubles under `tests/helpers.py`.

The old `BehaviorFeedbackLearning`, `SimpleSelector`, `GradientLearning`,
`ScientificMetathin` and related classes are intentionally no longer part of
this distribution. They remain in the git history on earlier branches if you
need to recover an implementation.

---

## License

MIT.
