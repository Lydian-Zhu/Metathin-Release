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

### Changed

- Added `pyproject.toml` (PEP 517/518 build) and `.gitignore`.
- Added PEP 561 `py.typed` markers referenced by `setup.py` package data.
- Removed the unused legacy module `metathin_plus/sci/memory/pretrained.py`
  (duplicated by the `pretrained/` package) and a committed `.pyc` file.

### Known issues

- `TestMemoryLearning::test_prune` is time-boundary dependent and currently fails.
- `metathin_plus.sci` has no dedicated test suite yet.
- `User Manual(Now it is empty).md` is still an empty placeholder.
