from __future__ import annotations

import csv
import itertools
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from pcgrl.app.analysis.figures import effects_pareto
from pcgrl.app.analysis.stats import factorial_effects
from pcgrl.app.experiments.spec import FactorialSpec, apply_overrides, split_cell_params
from pcgrl.core.helpers.path import ensure_dir, resolve_path
from pcgrl.core.settings.schema import RuntimeConfig
from pcgrl.core.tiles.catalog import load_catalog
from pcgrl.runtime.metrics.intent import compute_intent_vector

logger = logging.getLogger(__name__)

STAGES = ("train", "evaluate", "analyze", "all")


def design_matrix(spec: FactorialSpec) -> list[dict[str, Any]]:
    """Full 2^k design with replicates, one row per training/evaluation run.

    Args:
        spec: Factorial specification.

    Returns:
        Rows with ``cell``, ``replicate``, per-factor ±1 levels, and the
        merged dotted-path overrides for the cell.
    """
    rows: list[dict[str, Any]] = []
    for levels in itertools.product((-1, 1), repeat=len(spec.factors)):
        cell = "".join(
            f"{factor.name}{'+' if level > 0 else '-'}"
            for factor, level in zip(spec.factors, levels, strict=True)
        )
        overrides: dict[str, Any] = {}
        for factor, level in zip(spec.factors, levels, strict=True):
            overrides.update(
                factor.overrides_high if level > 0 else factor.overrides_low
            )
        for replicate in range(spec.replicates):
            rows.append(
                {
                    "cell": cell,
                    "replicate": replicate,
                    "levels": dict(
                        zip(
                            [factor.name for factor in spec.factors],
                            levels,
                            strict=True,
                        )
                    ),
                    "overrides": overrides,
                }
            )
    return rows


def run_factorial(
    config: RuntimeConfig,
    spec: FactorialSpec,
    *,
    stage: str = "all",
    output_dir: Path | None = None,
    checkpoint: Path | None = None,
) -> Path:
    """Run one stage (or all) of a factorial experiment.

    Description:
        ``train`` retrains one policy per (cell, replicate) — resumable, a
        cell with an existing ``policy_final.zip`` is skipped. ``evaluate``
        rolls the per-cell policy over the evaluation target suite (no
        bridge) and appends rows to ``cells.csv``. ``analyze`` computes
        ranked effects with permutation p-values from ``cells.csv``.
        Inference-kind specs skip training and evaluate the shared
        ``checkpoint`` under per-cell config overrides instead.

    Args:
        config: Base runtime configuration.
        spec: Factorial specification.
        stage: One of ``STAGES``.
        output_dir: Experiment directory (default: artifacts tree by name).
        checkpoint: Shared policy for inference-kind specs.

    Returns:
        The experiment output directory.

    Raises:
        ValueError: For unknown stages or a missing inference checkpoint.
    """
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r} (choose from {STAGES})")
    out_dir = ensure_dir(
        output_dir or resolve_path("artifacts/experiments") / spec.name
    )
    (out_dir / "spec_snapshot.yaml").write_text(
        yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    if stage in ("train", "all") and spec.kind == "training":
        _stage_train(config, spec, out_dir)
    if stage in ("evaluate", "all"):
        _stage_evaluate(config, spec, out_dir, checkpoint=checkpoint)
    if stage in ("analyze", "all"):
        _stage_analyze(spec, out_dir)
    return out_dir


def _cell_dir(out_dir: Path, row: dict[str, Any]) -> Path:
    """Per-(cell, replicate) artifact directory."""
    return out_dir / "cells" / f"{row['cell']}_r{row['replicate']}"


def _cell_config(
    config: RuntimeConfig, spec: FactorialSpec, row: dict[str, Any]
) -> tuple[RuntimeConfig, dict[str, Any]]:
    """Apply base + cell overrides plus a replicate seed; split runner params."""
    merged = {**spec.base_overrides, **row["overrides"]}
    config_overrides, params = split_cell_params(merged)
    updated = apply_overrides(config, config_overrides)
    experiment = updated.experiment.model_copy(
        update={"seed": config.experiment.seed + 1_000 * int(row["replicate"])}
    )
    return updated.model_copy(update={"experiment": experiment}), params


def _stage_train(config: RuntimeConfig, spec: FactorialSpec, out_dir: Path) -> None:
    """Train one policy per (cell, replicate); skip cells already trained."""
    from pcgrl.learning.training.runner import run_train

    for row in design_matrix(spec):
        cell_dir = _cell_dir(out_dir, row)
        checkpoint = cell_dir / "models" / "policy_final.zip"
        if checkpoint.is_file():
            logger.info("factorial train: %s already trained, skipping", cell_dir.name)
            continue
        logger.info("factorial train: %s", cell_dir.name)
        cell_config, _ = _cell_config(config, spec, row)
        run_train(cell_config, run_dir=ensure_dir(cell_dir))


def _stage_evaluate(
    config: RuntimeConfig,
    spec: FactorialSpec,
    out_dir: Path,
    *,
    checkpoint: Path | None,
) -> None:
    """Score every cell over the target suite and write ``cells.csv``."""
    from pcgrl.app.evaluation.controllability import composer_policy_generate_fn
    from pcgrl.runtime.metrics.tpkl import PatternReference
    from pcgrl.runtime.reward.composer_terminal import ComposerTerminalReward

    rows_out: list[dict[str, Any]] = []
    for row in design_matrix(spec):
        cell_config, cell_params = _cell_config(config, spec, row)
        if spec.kind == "inference":
            if checkpoint is None:
                raise ValueError("inference-kind specs require a shared checkpoint")
            rows_out.append(
                _evaluate_inference_cell(
                    cell_config, spec, row, cell_params, checkpoint
                )
            )
            continue
        cell_checkpoint = _cell_dir(out_dir, row) / "models" / "policy_final.zip"
        if not cell_checkpoint.is_file():
            logger.warning(
                "factorial evaluate: missing checkpoint for %s, skipping",
                row["cell"],
            )
            continue
        generate_fn = composer_policy_generate_fn(cell_config, cell_checkpoint)
        catalog = load_catalog(cell_config)
        reference = PatternReference.load(
            resolve_path(cell_config.dataset.processed_root), split="train"
        )
        terminal = ComposerTerminalReward(cell_config, catalog, reference)
        errors: list[float] = []
        rewards: list[float] = []
        for case_index, case in enumerate(cell_config.evaluation.target_suite):
            target = np.array(
                cell_config.build_target_vector(**case.overrides), dtype=np.float32
            )
            for episode in range(spec.episodes):
                seed = cell_config.experiment.seed + 7_919 * case_index + episode
                level = generate_fn(target, seed)
                measured = compute_intent_vector(level, catalog, cell_config)
                chunk_ids = tuple(
                    int(c) for c in level.metadata.get("chunk_ids", ())
                )
                reward, _ = terminal.compute(
                    level, target, chunk_ids=chunk_ids, measured=measured
                )
                errors.append(float(np.abs(measured - target).mean()))
                rewards.append(float(reward))
        rows_out.append(
            {
                "cell": row["cell"],
                "replicate": row["replicate"],
                **{f"level_{k}": v for k, v in row["levels"].items()},
                "intent_error_mean": round(float(np.mean(errors)), 4),
                "terminal_reward": round(float(np.mean(rewards)), 4),
                "n": len(errors),
            }
        )
        logger.info(
            "factorial evaluate: %s r%d error=%.3f reward=%.3f",
            row["cell"],
            row["replicate"],
            rows_out[-1]["intent_error_mean"],
            rows_out[-1]["terminal_reward"],
        )
    if not rows_out:
        raise RuntimeError("factorial evaluate produced no rows (no checkpoints?)")
    cells_path = out_dir / "cells.csv"
    with cells_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)


def _stage_analyze(spec: FactorialSpec, out_dir: Path) -> None:
    """Compute ranked effects per response from ``cells.csv``."""
    cells_path = out_dir / "cells.csv"
    if not cells_path.is_file():
        raise FileNotFoundError(f"{cells_path} missing; run the evaluate stage first")
    with cells_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    factor_names = [factor.name for factor in spec.factors]
    design = np.array(
        [[float(row[f"level_{name}"]) for name in factor_names] for row in rows]
    )
    effects_rows: list[dict[str, Any]] = []
    anova: dict[str, Any] = {"design": "full_2^k", "n_runs": len(rows)}
    for response in spec.responses:
        values = np.array([float(row[response]) for row in rows])
        effects = factorial_effects(design, values, factor_names)
        anova[response] = effects
        for item in effects:
            effects_rows.append({"response": response, **item})
        effects_pareto(effects, out_dir / f"effects_{response}.png")
    with (out_dir / "effects.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["response", "name", "effect", "p_value", "order"]
        )
        writer.writeheader()
        writer.writerows(effects_rows)
    (out_dir / "anova.json").write_text(json.dumps(anova, indent=2), encoding="utf-8")
    logger.info("factorial analyze: %d effect rows -> %s", len(effects_rows), out_dir)


def _evaluate_inference_cell(
    cell_config: RuntimeConfig,
    spec: FactorialSpec,
    row: dict[str, Any],
    params: dict[str, Any],
    checkpoint: Path,
) -> dict[str, Any]:
    """Score one inference-factorial cell: translate -> generate -> measure.

    Description:
        The description suite is chosen by the ``param.description_style``
        cell parameter (``templated`` band-center grids or the scored
        ``handwritten`` entries); each description is translated under the
        cell's translator config (grounding factor, record/replay store) and
        generated through the best-of-N pipeline (``generation.max_attempts``
        factor). Responses: stage-(a) translation error and end-to-end intent
        error, both against the suite reference vectors.
    """
    from pcgrl.app.evaluation.comparison import build_generate_fn
    from pcgrl.app.experiments.llm_eval import handwritten_suite, templated_suite
    from pcgrl.app.intent.factory import build_translator

    style = str(params.get("description_style", "templated"))
    if style == "templated":
        suite = templated_suite(cell_config, cases=12)
    elif style == "handwritten":
        suite = [
            (description, reference)
            for description, reference in handwritten_suite(cell_config)
            if reference is not None
        ]
    else:
        raise ValueError(f"unknown description_style {style!r}")
    translator = build_translator(cell_config)
    generate_fn = build_generate_fn(cell_config, "ppo_best_of_n", checkpoint=checkpoint)
    catalog = load_catalog(cell_config)
    translation_errors: list[float] = []
    e2e_errors: list[float] = []
    for case_index, (description, reference) in enumerate(suite):
        translation = translator.translate(description)
        requested = np.asarray(translation.vector, dtype=np.float32)
        translation_errors.append(float(np.abs(requested - reference).mean()))
        for episode in range(spec.episodes):
            seed = cell_config.experiment.seed + 7_919 * case_index + episode
            level = generate_fn(requested, seed)
            measured = compute_intent_vector(level, catalog, cell_config)
            e2e_errors.append(float(np.abs(measured - reference).mean()))
    result = {
        "cell": row["cell"],
        "replicate": row["replicate"],
        **{f"level_{k}": v for k, v in row["levels"].items()},
        "translation_error_mean": round(float(np.mean(translation_errors)), 4),
        "e2e_intent_error_mean": round(float(np.mean(e2e_errors)), 4),
        "n": len(e2e_errors),
    }
    logger.info(
        "factorial inference: %s r%d translation=%.3f e2e=%.3f",
        row["cell"],
        row["replicate"],
        result["translation_error_mean"],
        result["e2e_intent_error_mean"],
    )
    return result
