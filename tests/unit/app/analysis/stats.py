from __future__ import annotations

import numpy as np
import pytest
from pcgrl.app.analysis.stats import (
    bootstrap_ci,
    factorial_effects,
    linear_fit,
    permutation_pvalue,
)


def test_bootstrap_ci_brackets_the_mean() -> None:
    """The interval must contain the sample mean and tighten with low variance."""
    rng = np.random.default_rng(0)
    sample = rng.normal(10.0, 1.0, size=200)
    low, high = bootstrap_ci(sample, seed=1)
    assert low < sample.mean() < high
    assert high - low < 1.0


def test_bootstrap_ci_degenerate_sample() -> None:
    """A single observation collapses to a point interval."""
    assert bootstrap_ci([3.5]) == (3.5, 3.5)


def test_permutation_pvalue_detects_separation() -> None:
    """Clearly separated samples earn a small p; identical ones a large p."""
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 0.1, size=30)
    b = rng.normal(5.0, 0.1, size=30)
    assert permutation_pvalue(a, b, n_permutations=500, seed=1) <= 1.0 / 500.0 + 1e-9
    same = rng.normal(0.0, 1.0, size=30)
    assert permutation_pvalue(same, same, n_permutations=200, seed=1) > 0.5


def test_permutation_pvalue_floor() -> None:
    """The p-value never claims more resolution than the permutation count."""
    p = permutation_pvalue([0.0] * 10, [10.0] * 10, n_permutations=100, seed=0)
    assert p >= 1.0 / 101.0


def test_factorial_effects_recovers_planted_effect() -> None:
    """A response built from factor A yields A as the dominant ranked effect."""
    factors = ["A", "B"]
    design = np.array([[a, b] for a in (-1.0, 1.0) for b in (-1.0, 1.0) for _ in range(4)])
    rng = np.random.default_rng(0)
    response = 3.0 * design[:, 0] + rng.normal(0.0, 0.05, size=len(design))
    effects = factorial_effects(design, response, factors, n_permutations=300, seed=1)
    assert effects[0]["name"] == "A"
    assert effects[0]["effect"] == pytest.approx(6.0, abs=0.2)
    assert float(effects[0]["p_value"]) < 0.05
    names = {item["name"] for item in effects}
    assert names == {"A", "B", "A:B"}


def test_factorial_effects_validates_shapes() -> None:
    """Mismatched design and response shapes raise."""
    with pytest.raises(ValueError, match="design"):
        factorial_effects(np.ones((4, 2)), [1.0, 2.0], ["A", "B"])


def test_linear_fit_exact_line() -> None:
    """An exact line recovers slope, intercept, and R^2 = 1."""
    x = np.arange(10, dtype=np.float64)
    y = 2.0 * x + 1.0
    fit = linear_fit(x, y)
    assert fit["slope"] == pytest.approx(2.0)
    assert fit["intercept"] == pytest.approx(1.0)
    assert fit["r2"] == pytest.approx(1.0)


def test_linear_fit_degenerate_x() -> None:
    """Constant predictors yield a flat fit instead of crashing."""
    fit = linear_fit([1.0, 1.0, 1.0], [2.0, 3.0, 4.0])
    assert fit["slope"] == 0.0
    assert fit["r2"] == 0.0
