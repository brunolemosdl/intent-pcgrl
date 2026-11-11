from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np

from pcgrl.core.domain.level import LevelGrid
from pcgrl.core.settings.schema import RuntimeConfig
from pcgrl.core.tiles.catalog import TileCatalog, load_catalog
from pcgrl.runtime.metrics.intent import INTENT_KEYS, compute_intent_vector

logger = logging.getLogger(__name__)

GenerateFn = Callable[[np.ndarray, int], LevelGrid]


def _ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks, ties sharing the mean rank."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    index = 0
    while index < len(values):
        upper = index
        while upper + 1 < len(values) and values[order[upper + 1]] == values[order[index]]:
            upper += 1
        ranks[order[index : upper + 1]] = (index + upper) / 2.0
        index = upper + 1
    return ranks


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman rank correlation without a scipy dependency.

    Args:
        x: Requested target values along one intent dimension.
        y: Measured values for the same dimension.

    Returns:
        Spearman rho in ``[-1, 1]``, or ``0.0`` for degenerate inputs.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        return 0.0
    rank_x = _ranks(x)
    rank_y = _ranks(y)
    std_x = rank_x.std()
    std_y = rank_y.std()
    if std_x == 0.0 or std_y == 0.0:
        return 0.0
    covariance = ((rank_x - rank_x.mean()) * (rank_y - rank_y.mean())).mean()
    return float(covariance / (std_x * std_y))


def evaluate_controllability(
    config: RuntimeConfig,
    generate_fn: GenerateFn,
    *,
    catalog: TileCatalog | None = None,
    dims: tuple[str, ...] = INTENT_KEYS,
) -> dict[str, dict[str, float]]:
    """Run the per-dimension controllability sweep protocol.

    Args:
        config: Runtime settings for sweep values and pass thresholds.
        generate_fn: Callable ``(target, seed) -> LevelGrid`` under test.
        catalog: Optional tile catalog override.
        dims: Intent dimensions to sweep; defaults to ``INTENT_KEYS``.

    Returns:
        Per-dimension MAE, Spearman rho, pass flags, and sample count.
    """
    catalog = catalog or load_catalog(config)
    settings = config.evaluation.controllability
    default_target = np.asarray(config.default_target_vector(), dtype=np.float32)
    results: dict[str, dict[str, float]] = {}
    for dim in dims:
        dim_index = INTENT_KEYS.index(dim)
        requested: list[float] = []
        measured: list[float] = []
        for value in settings.sweep_values:
            target = default_target.copy()
            target[dim_index] = value
            for episode in range(settings.levels_per_value):
                seed = (
                    config.experiment.seed
                    + 100_000 * dim_index
                    + 1_000 * int(value * 100)
                    + episode
                )
                level = generate_fn(target, seed)
                vector = compute_intent_vector(level, catalog, config)
                requested.append(float(value))
                measured.append(float(vector[dim_index]))
        mae = float(np.mean(np.abs(np.array(measured) - np.array(requested))))
        rho = spearman_rho(np.array(requested), np.array(measured))
        results[dim] = {
            "mae": round(mae, 4),
            "spearman_rho": round(rho, 4),
            "passes_mae": mae <= settings.max_mae,
            "passes_rho": rho >= settings.min_spearman,
            "n": len(requested),
        }
        logger.info("controllability %s: MAE=%.3f rho=%.3f", dim, mae, rho)
    return results


def write_controllability_report(results: dict[str, dict[str, float]], output_dir: Path) -> Path:
    """Persist per-dimension controllability results to JSON.

    Args:
        results: Output of ``evaluate_controllability``.
        output_dir: Directory for ``controllability.json``.

    Returns:
        Path to the written report.

    Raises:
        OSError: When the report file cannot be written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "controllability.json"
    passing = sum(1 for stats in results.values() if stats["passes_mae"] and stats["passes_rho"])
    payload = {
        "per_dim": results,
        "dims_passing": passing,
        "dims_total": len(results),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def composer_baseline_generate_fn(
    config: RuntimeConfig, baseline: str = "greedy_intent"
) -> GenerateFn:
    """Build a controllability generator backed by a non-RL composer baseline.

    Args:
        config: Runtime settings for library and composer core.
        baseline: Baseline policy name registered in composer policies.

    Returns:
        Callable ``(target, seed) -> LevelGrid`` for controllability sweeps.
    """
    from pcgrl.core.helpers.path import resolve_path
    from pcgrl.core.library.types import ChunkLibrary
    from pcgrl.runtime.environment.composer_policies import (
        BigramMarkov,
        rollout_baseline,
    )
    from pcgrl.runtime.environment.composer_state import ComposerCore

    catalog = load_catalog(config)
    library = ChunkLibrary.load(resolve_path(config.library.output_root))
    core = ComposerCore(library, config, catalog)
    markov = BigramMarkov.fit(library) if baseline == "markov" else None

    def generate(target: np.ndarray, seed: int) -> LevelGrid:
        """Compose one level for the requested target and seed."""
        rng = np.random.default_rng(seed)
        state = rollout_baseline(baseline, core, target, rng, markov=markov)
        return core.stitch_level(state)

    return generate


def composer_policy_polished_generate_fn(config: RuntimeConfig, checkpoint: Path) -> GenerateFn:
    """Single policy rollout followed by the deterministic polish stage.

    Description:
        The product-path controllability: measures what a user receives with
        ``generation.polish`` on (no bridge, matching sweep semantics). The
        unpolished ``composer_policy_generate_fn`` stays the raw-policy
        protocol used by the pre-registered E5 criterion.

    Args:
        config: Runtime settings.
        checkpoint: Trained policy archive.

    Returns:
        Callable ``(target, seed) -> LevelGrid``.
    """
    from pcgrl.runtime.refine import IntentRefiner

    raw = composer_policy_generate_fn(config, checkpoint)
    refiner = IntentRefiner(
        config,
        load_catalog(config),
        tolerance=config.generation.polish.tolerance,
        max_passes=config.generation.polish.max_passes,
    )

    def generate(target: np.ndarray, seed: int) -> LevelGrid:
        """Roll out one episode and polish it toward the target."""
        return refiner.refine(raw(target, seed), np.asarray(target, np.float32)).level

    return generate


def composer_policy_generate_fn(config: RuntimeConfig, checkpoint: Path) -> GenerateFn:
    """Build a controllability generator backed by a trained composer policy.

    Args:
        config: Runtime settings for the composer environment.
        checkpoint: Trained ``MaskablePPO`` policy archive.

    Returns:
        Callable ``(target, seed) -> LevelGrid`` using a single rollout.

    Description:
        Uses a direct environment rollout without best-of-N selection so
        controllability measures the raw policy response.
    """
    from sb3_contrib import MaskablePPO

    from pcgrl.learning.env_factory import build_env_context, make_env

    context = build_env_context(config)
    model = MaskablePPO.load(str(checkpoint), device="cpu")

    def generate(target: np.ndarray, seed: int) -> LevelGrid:
        """Roll out one conditioned episode and return the stitched level."""
        env = make_env(config, context, seed=seed)
        obs, _ = env.reset(seed=seed, options={"target": target})
        terminated = truncated = False
        while not (terminated or truncated):
            mask = env.action_masks()
            action, _ = model.predict(obs, action_masks=mask, deterministic=False)
            obs, _, terminated, truncated, _ = env.step(int(action))
        return env.get_level()

    return generate
