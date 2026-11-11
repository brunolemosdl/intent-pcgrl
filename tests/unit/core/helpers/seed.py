from __future__ import annotations

import random

import numpy as np
from pcgrl.core.helpers.seed import seed_all


def test_seed_all_numpy_reproducible() -> None:
    """Verify seed all numpy reproducible."""
    seed_all(42)
    a = np.random.randint(0, 1000, size=10)
    seed_all(42)
    b = np.random.randint(0, 1000, size=10)
    assert np.array_equal(a, b)


def test_seed_all_python_reproducible() -> None:
    """Verify seed all python reproducible."""
    seed_all(99)
    a = [random.random() for _ in range(5)]
    seed_all(99)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_seed_all_different_seeds_differ() -> None:
    """Verify seed all different seeds differ."""
    seed_all(1)
    a = np.random.randint(0, 10000)
    seed_all(2)
    b = np.random.randint(0, 10000)
    assert a != b


def test_seed_all_no_crash_without_torch() -> None:
    """Verify seed all no crash without torch."""
    from unittest.mock import patch

    with patch.dict("sys.modules", {"torch": None}):
        seed_all(7)
