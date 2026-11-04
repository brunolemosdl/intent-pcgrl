from __future__ import annotations

import csv
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from pcgrl.app.analysis.stats import bootstrap_ci, linear_fit
from pcgrl.app.evaluation.controllability import GenerateFn, spearman_rho
from pcgrl.app.intent.types import IntentTranslator
from pcgrl.app.intent.verbalizer import band_center, band_phrase, verbalize
from pcgrl.core.helpers.path import ensure_dir, resolve_path
from pcgrl.core.settings.schema import RuntimeConfig
from pcgrl.core.tiles.catalog import load_catalog
from pcgrl.runtime.metrics.intent import INTENT_KEYS, compute_intent_vector

logger = logging.getLogger(__name__)

_DENSITY_DIMS = (
    "enemy_density",
    "coin_density",
    "interactive_block_density",
    "pipe_density",
)
_HANDWRITTEN_PATH = "data/descriptions/handwritten.json"


def templated_suite(
    config: RuntimeConfig, *, cases: int = 24, seed: int | None = None
) -> list[tuple[str, np.ndarray]]:
    """Generate (description, reference vector) pairs from seeded target grids.

    Description:
        Targets are sampled uniformly, then snapped to the verbalizer band
        centers so the verbalized description and the reference vector agree
        by construction (anti-circularity: the reference is defined by the
        template, not by any translator).

    Args:
        config: Runtime settings.
        cases: Number of pairs to generate.
        seed: RNG seed; defaults to the experiment seed.

    Returns:
        List of (canonical description, reference vector) pairs.
    """
    from pcgrl.app.intent.verbalizer import band_index

    rng = np.random.default_rng(seed if seed is not None else config.experiment.seed)
    names = config.target_vector_field_names()
    pairs: list[tuple[str, np.ndarray]] = []
    for _ in range(cases):
        raw = rng.uniform(0.0, 1.0, size=len(names))
        reference = np.array(
            [
                band_center(name, band_index(name, float(value), config), config)
                for name, value in zip(names, raw, strict=True)
            ],
            dtype=np.float32,
        )
        pairs.append((verbalize(reference, config), reference))
    return pairs


def handwritten_suite(config: RuntimeConfig) -> list[tuple[str, np.ndarray | None]]:
    """Load the free-form description suite with optional reference vectors.

    Args:
        config: Runtime settings (for ``build_target_vector`` defaults).

    Returns:
        List of (description, reference vector or ``None``) pairs.
    """
    path = resolve_path(_HANDWRITTEN_PATH)
    if not path.is_file():
        return []
    entries = json.loads(path.read_text(encoding="utf-8"))
    pairs: list[tuple[str, np.ndarray | None]] = []
    for entry in entries:
        overrides = entry.get("overrides")
        reference = (
            np.array(config.build_target_vector(**overrides), dtype=np.float32)
            if overrides is not None
            else None
        )
        pairs.append((str(entry["description"]), reference))
    return pairs


def run_translation_eval(
    config: RuntimeConfig,
    translator: IntentTranslator,
    *,
    output_dir: Path | None = None,
    templated_cases: int = 24,
) -> Path:
    """Measure stage-(a) translation error over the description suites.

    Args:
        config: Runtime settings.
        translator: Description -> intent translator under test.
        output_dir: Destination directory; timestamped default otherwise.
        templated_cases: Number of templated pairs to generate.

    Returns:
        Path to the written ``translation_error.csv``.
    """
    out_dir = _resolve_output(output_dir, "llm_eval")
    names = config.target_vector_field_names()
    rows: list[dict[str, object]] = []
    suites: list[tuple[str, str, np.ndarray | None]] = [
        ("templated", description, reference)
        for description, reference in templated_suite(config, cases=templated_cases)
    ] + [
        ("handwritten", description, reference)
        for description, reference in handwritten_suite(config)
    ]
    for suite, description, reference in suites:
        translation = translator.translate(description)
        row: dict[str, object] = {
            "suite": suite,
            "description": description,
            "source": translation.source,
            "clamped": translation.clamped,
        }
        if reference is not None:
            deltas = np.abs(np.asarray(translation.vector) - reference)
            row["translation_error_mean"] = round(float(deltas.mean()), 4)
            row["translation_error_max"] = round(float(deltas.max()), 4)
            for index, name in enumerate(names):
                row[f"error_{name}"] = round(float(deltas[index]), 4)
        rows.append(row)
    csv_path = out_dir / "translation_error.csv"
    _write_csv(csv_path, rows)
    scored = [
        float(row["translation_error_mean"])  # type: ignore[arg-type]
        for row in rows
        if "translation_error_mean" in row
    ]
    per_dim = {}
    for name in names:
        values = [
            float(row[f"error_{name}"])  # type: ignore[arg-type]
            for row in rows
            if f"error_{name}" in row
        ]
        low, high = bootstrap_ci(values)
        per_dim[name] = {
            "mae": round(float(np.mean(values)), 4),
            "ci95_low": round(low, 4),
            "ci95_high": round(high, 4),
            "n": len(values),
        }
    low, high = bootstrap_ci(scored)
    summary = {
        "translation_error_mean": round(float(np.mean(scored)), 4),
        "ci95_low": round(low, 4),
        "ci95_high": round(high, 4),
        "n_scored": len(scored),
        "n_total": len(rows),
        "per_dim": per_dim,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    logger.info(
        "translation eval: MAE=%.3f over %d scored descriptions",
        summary["translation_error_mean"],
        len(scored),
    )
    return csv_path


def run_monotonicity(
    config: RuntimeConfig,
    translator: IntentTranslator,
    generate_fn: GenerateFn,
    *,
    output_dir: Path | None = None,
    episodes: int = 3,
) -> Path:
    """Prove the description measurably moves the level (impact monotonicity).

    Description:
        For each density dimension, five banded descriptions ("no enemies" ->
        "as many enemies as possible") are translated and generated; the
        measured value should rise with the requested band (Spearman rho and
        regression slope/R^2 per dimension).

    Args:
        config: Runtime settings.
        translator: Description -> intent translator.
        generate_fn: ``(target, seed) -> LevelGrid`` system under test.
        output_dir: Destination directory; timestamped default otherwise.
        episodes: Generations per (dimension, band).

    Returns:
        Path to the written ``monotonicity.csv``.
    """
    out_dir = _resolve_output(output_dir, "monotonicity")
    catalog = load_catalog(config)
    rows: list[dict[str, object]] = []
    summary: dict[str, dict[str, float]] = {}
    for dim in _DENSITY_DIMS:
        dim_index = INTENT_KEYS.index(dim)
        requested: list[float] = []
        measured: list[float] = []
        for band in range(5):
            description = "A level with " + band_phrase(dim, band) + "."
            translation = translator.translate(description)
            target = np.asarray(translation.vector, dtype=np.float32)
            for episode in range(episodes):
                seed = config.experiment.seed + 10_000 * dim_index + 100 * band + episode
                level = generate_fn(target, seed)
                vector = compute_intent_vector(level, catalog, config)
                requested.append(band_center(dim, band, config))
                measured.append(float(vector[dim_index]))
                rows.append(
                    {
                        "dim": dim,
                        "band": band,
                        "description": description,
                        "requested": round(band_center(dim, band, config), 4),
                        "translated": round(float(target[dim_index]), 4),
                        "episode": episode,
                        "seed": seed,
                        "measured": round(float(vector[dim_index]), 4),
                    }
                )
        rho = spearman_rho(np.asarray(requested), np.asarray(measured))
        fit = linear_fit(requested, measured)
        summary[dim] = {
            "spearman_rho": round(rho, 4),
            "slope": round(fit["slope"], 4),
            "r2": round(fit["r2"], 4),
            "n": len(requested),
        }
        logger.info(
            "monotonicity %s: rho=%.3f slope=%.3f r2=%.3f", dim, rho, fit["slope"], fit["r2"]
        )
    csv_path = out_dir / "monotonicity.csv"
    _write_csv(csv_path, rows)
    (out_dir / "monotonicity_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return csv_path


def _resolve_output(output_dir: Path | None, label: str) -> Path:
    """Resolve the output directory, defaulting to a timestamped tree."""
    if output_dir is not None:
        return ensure_dir(output_dir)
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    return ensure_dir(resolve_path("artifacts/experiments") / label / timestamp)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows to CSV with the union of all observed columns."""
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
