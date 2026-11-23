"""Component test fixtures.

The ``TestMemoryLearning::test_prune`` case builds timestamps from
``time.time()`` and then checks a fixed 30-second boundary. Freezing the
clock for that single test removes the race between building the sample
data and pruning it, without changing production behaviour.
"""

from __future__ import annotations

import time

import pytest


@pytest.fixture(autouse=True)
def _freeze_clock_for_prune(request, monkeypatch):
    if request.node.name != "test_prune":
        return
    frozen = time.time()
    monkeypatch.setattr(time, "time", lambda: frozen)
