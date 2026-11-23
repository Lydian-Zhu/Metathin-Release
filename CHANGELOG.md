# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## 0.5.0 - 2026-11-23

### Breaking

- Removed `metathin.components` entirely. The package now ships interfaces and
  orchestration only; applications must inject their own P, B, S, D and Ψ.
- Removed `metathin_plus` and `metathin_plus.sci`, along with the `sci`,
  `deep`, `viz` and `data` extras.
- `Metathin` and `MetathinBuilder` now require an explicit `PatternSpace`,
  `Selector` and `DecisionStrategy`; no default implementations are created.
- `PipelineConfig.enable_learning` now defaults to `False`. Enabling learning
  without a `LearningMechanism` raises at construction time.
- Version bumped to `0.5.0`; package discovery only includes `metathin`.

### Added

- Runtime contract validation in `ThinkingPipeline`:
  - component interface and behavior-name checks;
  - feature vector shape/dtype/finite checks;
  - fitness must be a finite number in `[0, 1]`;
  - the decision strategy must return a candidate;
  - selector parameters and learning adjustments must be finite numeric maps,
    and learning may only update exposed parameter keys.
- Contract tests covering malformed selectors, foreign decisions, duplicate
  behavior names, bad parameter maps, unknown adjustment keys and bad
  `post_update` output.

### Changed

- README and packaging describe an interface-first framework rather than a
  bundle of built-in algorithms.
- Tests use local doubles under `tests/helpers.py`; component tests were removed.

### Removed

- `BehaviorFeedbackLearning`, `GradientLearning`, `RewardLearning`,
  `MemoryLearning`, `HebbianLearning`, `EnsembleLearning`.
- `SimplePatternSpace` and other built-in pattern spaces.
- `FunctionBehavior` and other built-in behavior wrappers.
- `SimpleSelector`, `PolynomialSelector` and other built-in selectors.
- `MaxFitnessStrategy` and other built-in decision strategies.
- `metathin_plus.sci` and its pretrained function libraries.

## 0.4.0

### Fixed

- `metathin_plus` is now a regular package: added `metathin_plus/__init__.py` and
  `metathin_plus/sci/memory/__init__.py`, so `import metathin_plus.sci` works.
- The built wheel now contains `metathin_plus`, its pretrained JSON libraries and
  `py.typed` markers; `pip install "metathin[sci]"` actually installs the
  scientific discovery module.
- `requirements.txt` is now a valid requirements file (previously it contained
  Markdown prose and a fenced code block, so `pip install -r` failed).
- Python 3.8 is supported again: `from __future__ import annotations` in
  `metathin/engine/pipeline.py` and `engine/hooks.py` makes the PEP 585
  annotations lazy (verified on CPython 3.8.20).
- `TestMemoryLearning::test_prune` is deterministic again: a test-local fixture
  freezes the clock for that case so the 30-second boundary cannot race.
- `metathin_plus.sci` no longer prints on import and reports version `0.4.0`,
  the same release version as the core package.
- History tracking now records every thought: empty `HistoryTracker` no longer
  evaluates as `False`, and the agent uses the existing `record(context)` API.
- Learning now defaults to `BehaviorFeedbackLearning`, a behavior-level negative
  feedback rule: a wrong execution lowers the selected behavior's fitness
  weights, instead of regressing selector weights against the final output.
  `GradientLearning` remains available and now also updates bias terms.

### Changed

- `BehaviorFeedbackLearning` is exported from `metathin` and
  `metathin.components`; it is the default when `enable_learning=True` and no
  explicit mechanism is supplied.
- Learning context now carries `behavior_index` and `reward`, and the pipeline
  applies `post_update` hooks (Hebbian normalization) and records experiences
  for `MemoryLearning`.
- `BehaviorFeedbackLearning` now supports training-time ε-greedy exploration
  (`exploration_rate`, `exploration_decay`, `min_exploration`), with
  `exploration_margin` suppressing noise when the top-2 fitness gap is
  already confident. Exploration never runs at inference (no `expected`).
- `BehaviorFeedbackLearning` applies competitive normalization after each
  update: each feature column is centered across behaviors, so one behavior's
  penalty raises the others' relative scores instead of letting all rows drift
  negative together. Disable with `normalize_weights=False`.
- `BehaviorFeedbackLearning` now performs counterfactual credit assignment:
  when the expected behavior is known and differs from the executed one, the
  expected behavior's row receives a positive update. Disable with
  `counterfactual=False`; scale with `counterfactual_feedback`.
- Selectors expose `get_learning_features()` and `parameter_prefix`, so
  learning mechanisms can update `SimpleSelector` (`w_`) and
  `PolynomialSelector` (`poly_w_`) with correctly shaped features and keys.

- Added `pyproject.toml` (PEP 517/518 build) and `.gitignore`.
- Added PEP 561 `py.typed` markers referenced by `setup.py` package data.
- Added a CI workflow: pytest on Python 3.8-3.12 plus a wheel packaging check.
- Removed the unused legacy module `metathin_plus/sci/memory/pretrained.py`
  (duplicated by the `pretrained/` package) and a committed `.pyc` file.

### Known issues

- `metathin_plus.sci` has no dedicated test suite yet.
- `User Manual(Now it is empty).md` is still an empty placeholder.
