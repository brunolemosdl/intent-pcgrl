from __future__ import annotations

import itertools

import numpy as np


def bootstrap_ci(
    values: np.ndarray | list[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean.

    Args:
        values: Sample values.
        confidence: Interval mass (default 95%).
        n_resamples: Bootstrap resamples.
        seed: RNG seed for reproducible intervals.

    Returns:
        Tuple of (lower, upper) bounds; degenerate samples collapse to the mean.
    """
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        return (float("nan"), float("nan"))
    if data.size == 1:
        return (float(data[0]), float(data[0]))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, data.size, size=(n_resamples, data.size))
    means = data[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(means, alpha)),
        float(np.quantile(means, 1.0 - alpha)),
    )


def permutation_pvalue(
    a: np.ndarray | list[float],
    b: np.ndarray | list[float],
    *,
    n_permutations: int = 10_000,
    seed: int = 0,
) -> float:
    """Two-sided permutation test p-value for a difference in means.

    Description:
        The p-value is floored at ``1 / (n_permutations + 1)`` — a permutation
        test can never claim more resolution than the permutations it ran.

    Args:
        a: First sample.
        b: Second sample.
        n_permutations: Label shuffles.
        seed: RNG seed.

    Returns:
        Two-sided p-value in ``(0, 1]``.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return 1.0
    observed = abs(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_permutations):
        rng.shuffle(pooled)
        delta = abs(pooled[: a.size].mean() - pooled[a.size :].mean())
        if delta >= observed - 1e-12:
            hits += 1
    return max(hits / n_permutations, 1.0 / (n_permutations + 1))


def factorial_effects(
    design: np.ndarray,
    response: np.ndarray | list[float],
    factor_names: list[str],
    *,
    max_interaction_order: int | None = None,
    n_permutations: int = 2_000,
    seed: int = 0,
) -> list[dict[str, object]]:
    """Estimate 2^k factorial effects with permutation p-values.

    Description:
        ``design`` uses ±1 coding, one row per run (replicates repeat rows).
        Each effect is ``mean(response | column == +1) - mean(response |
        column == -1)``; interaction columns are elementwise products of the
        main-factor columns. Effects are returned ranked by magnitude (the
        Pareto ordering used in the report figures).

    Args:
        design: Array of shape ``(n_runs, k)`` in ±1 coding.
        response: Response value per run.
        factor_names: One name per design column.
        max_interaction_order: Highest interaction order to include
            (default: all ``k`` orders).
        n_permutations: Permutations for each effect's p-value.
        seed: RNG seed.

    Returns:
        Ranked list of ``{name, effect, p_value, order}`` dicts.

    Raises:
        ValueError: When shapes or factor names are inconsistent.
    """
    design = np.asarray(design, dtype=np.float64)
    response = np.asarray(response, dtype=np.float64)
    if design.ndim != 2 or design.shape[0] != response.size:
        raise ValueError("design must be (n_runs, k) aligned with response")
    k = design.shape[1]
    if len(factor_names) != k:
        raise ValueError("factor_names must match design columns")
    order_cap = k if max_interaction_order is None else max_interaction_order
    rng = np.random.default_rng(seed)
    results: list[dict[str, object]] = []
    for order in range(1, order_cap + 1):
        for combo in itertools.combinations(range(k), order):
            column = design[:, combo].prod(axis=1)
            name = ":".join(factor_names[i] for i in combo)
            effect = _column_effect(column, response)
            hits = 0
            shuffled = response.copy()
            for _ in range(n_permutations):
                rng.shuffle(shuffled)
                if abs(_column_effect(column, shuffled)) >= abs(effect) - 1e-12:
                    hits += 1
            p_value = max(hits / n_permutations, 1.0 / (n_permutations + 1))
            results.append(
                {"name": name, "effect": float(effect), "p_value": float(p_value), "order": order}
            )
    results.sort(key=lambda item: abs(float(item["effect"])), reverse=True)  # type: ignore[arg-type]
    return results


def _column_effect(column: np.ndarray, response: np.ndarray) -> float:
    """High-minus-low mean difference along a ±1 contrast column."""
    high = response[column > 0]
    low = response[column < 0]
    if high.size == 0 or low.size == 0:
        return 0.0
    return float(high.mean() - low.mean())


def linear_fit(x: np.ndarray | list[float], y: np.ndarray | list[float]) -> dict[str, float]:
    """Least-squares line fit with R².

    Args:
        x: Predictor values.
        y: Response values.

    Returns:
        Dict with ``slope``, ``intercept``, and ``r2`` (0 for degenerate input).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or np.allclose(x, x[0]):
        return {"slope": 0.0, "intercept": float(y.mean()) if y.size else 0.0, "r2": 0.0}
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res = float(((y - predicted) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"slope": float(slope), "intercept": float(intercept), "r2": float(r2)}
