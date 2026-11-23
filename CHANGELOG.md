# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

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

### Changed

- Added `pyproject.toml` (PEP 517/518 build) and `.gitignore`.
- Added PEP 561 `py.typed` markers referenced by `setup.py` package data.
- Added a CI workflow: pytest on Python 3.8-3.12 plus a wheel packaging check.
- Removed the unused legacy module `metathin_plus/sci/memory/pretrained.py`
  (duplicated by the `pretrained/` package) and a committed `.pyc` file.

### Known issues

- `metathin_plus.sci` has no dedicated test suite yet.
- `User Manual(Now it is empty).md` is still an empty placeholder.
