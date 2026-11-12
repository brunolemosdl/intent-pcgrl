from __future__ import annotations

import csv
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from pcgrl.app.analysis.stats import bootstrap_ci, permutation_pvalue
from pcgrl.app.data.corpus.replay import corpus_replay_level
from pcgrl.app.data.library.extract import load_raw_levels, split_levels_for_heldout
from pcgrl.app.evaluation.census import census_distance, merge_censuses, structural_census
from pcgrl.app.evaluation.controllability import (
    GenerateFn,
    composer_policy_generate_fn,
    composer_policy_polished_generate_fn,
)
from pcgrl.core.domain.level import AreaType, LevelGrid
from pcgrl.core.helpers.path import ensure_dir, resolve_path
from pcgrl.core.settings.schema import RuntimeConfig
from pcgrl.core.tiles.catalog import TileCatalog, load_catalog
from pcgrl.runtime.metrics.intent import INTENT_KEYS, compute_intent_vector, effective_target
from pcgrl.runtime.metrics.tpkl import (
    PatternReference,
    empty_pattern_key,
    tile_pattern_kl,
)
from pcgrl.runtime.reward.composer_terminal import ComposerTerminalReward

logger = logging.getLogger(__name__)

SYSTEM_NAMES = (
    "ppo",
    "ppo_best_of_n",
    "ppo_best_of_n_polished",
    "ppo_random_edits",
    "ppo_planner_edits",
    "random_legal",
    "greedy_intent",
    "beam_search",
    "markov",
    "corpus_replay",
)

_BASELINE_SYSTEMS = ("random_legal", "greedy_intent", "beam_search", "markov")
_POLICY_SYSTEMS = (
    "ppo",
    "ppo_best_of_n",
    "ppo_best_of_n_polished",
    "ppo_random_edits",
    "ppo_planner_edits",
)
_SUMMARY_METRICS = (
    "terminal_reward",
    "intent_error_mean",
    "intent_error_max",
    "component_alignment",
    "component_style",
    "component_playability",
    "component_duplicate",
    "duplicate_rate",
    "wall_time_s",
)
_PAIRWISE_METRICS = ("terminal_reward", "intent_error_mean")


def build_generate_fn(
    config: RuntimeConfig, system: str, *, checkpoint: Path | None = None
) -> GenerateFn:
    """Build the ``(target, seed) -> LevelGrid`` callable for one system.

    Args:
        config: Runtime settings.
        system: One of ``SYSTEM_NAMES``.
        checkpoint: Trained policy archive (required for the ppo systems).

    Returns:
        GenerateFn for the requested system.

    Raises:
        ValueError: For unknown systems or a missing required checkpoint.
    """
    if system in _BASELINE_SYSTEMS:
        return _baseline_generate_fn(config, system)
    if system == "corpus_replay":
        return lambda target, seed: corpus_replay_level(config, seed)
    if system in _POLICY_SYSTEMS:
        if checkpoint is None:
            raise ValueError(f"system {system!r} requires --checkpoint")
        if system == "ppo":
            return composer_policy_generate_fn(config, checkpoint)
        # The edit-selection ablations keep the policy's composition and take
        # the edit choice away from it: uniformly random legal edits, or no
        # edit phase at all with the planner run to convergence instead.
        if system == "ppo_random_edits":
            return composer_policy_generate_fn(config, checkpoint, edits="random")
        if system == "ppo_planner_edits":
            return composer_policy_polished_generate_fn(
                config, checkpoint, edits="none", bounded=False
            )
        # The plain best-of-N system measures the raw two-phase policy: the
        # polish stage is pinned OFF regardless of the product config default,
        # and ON only for the explicit ablation system.
        return _best_of_n_generate_fn(config, checkpoint, polish=system == "ppo_best_of_n_polished")
    raise ValueError(f"unknown system {system!r} (choose from {SYSTEM_NAMES})")


def _baseline_generate_fn(config: RuntimeConfig, system: str) -> GenerateFn:
    """Search-baseline GenerateFn sharing one core, terminal, and markov fit."""
    from pcgrl.core.library.types import ChunkLibrary
    from pcgrl.runtime.environment.composer_policies import (
        BigramMarkov,
        rollout_baseline,
    )
    from pcgrl.runtime.environment.composer_state import ComposerCore

    catalog = load_catalog(config)
    library = ChunkLibrary.load(resolve_path(config.library.output_root))
    core = ComposerCore(library, config, catalog)
    reference = PatternReference.load(resolve_path(config.dataset.processed_root), split="train")
    terminal = ComposerTerminalReward(config, catalog, reference)
    markov = BigramMarkov.fit(library) if system == "markov" else None

    def generate(target: np.ndarray, seed: int) -> LevelGrid:
        """Roll out the baseline policy and stitch the level."""
        rng = np.random.default_rng(seed)
        state = rollout_baseline(system, core, target, rng, terminal=terminal, markov=markov)
        return core.stitch_level(state)

    return generate


def _best_of_n_generate_fn(
    config: RuntimeConfig, checkpoint: Path, *, polish: bool = False
) -> GenerateFn:
    """GenerateFn over the full best-of-N generate pipeline (gates included).

    Description:
        Reads the selected level back from the report JSON; a gate failure
        still yields the best diagnostic candidate rather than crashing the
        comparison run. ``polish`` is passed explicitly so harness results
        never depend on the product config default.
    """
    import tempfile

    from pcgrl.learning.generate import run_generate

    catalog = load_catalog(config)

    def generate(target: np.ndarray, seed: int) -> LevelGrid:
        """Run best-of-N generation and parse the selected level back."""
        with tempfile.TemporaryDirectory(prefix="compare_bestofn_") as tmp:
            out_dir = Path(tmp)
            try:
                report = run_generate(
                    config,
                    checkpoint,
                    target=target,
                    output_dir=out_dir,
                    seed=seed,
                    polish=polish,
                )
            except RuntimeError:
                candidates = sorted(out_dir.glob("level_*.json"))
                if not candidates:
                    raise
                report = candidates[-1]
            payload = json.loads(report.read_text(encoding="utf-8"))
            chunk_ids = payload.get("diagnostics", {}).get("chunk_ids", [])
            return LevelGrid.from_ascii(
                payload["ascii"],
                catalog,
                tile_size_px=config.game.tile_size_px,
                area_type=AreaType(config.game.area_type),
                metadata={"chunk_ids": list(chunk_ids)},
            )

    return generate


def _corpus_census(config: RuntimeConfig, catalog: TileCatalog) -> dict[str, object]:
    """Aggregate structural census of the library train-split corpus levels."""
    corpus_root = resolve_path(config.dataset.corpus_root)
    levels = load_raw_levels(corpus_root, config.library.level_sets, catalog)
    train_ids, _ = split_levels_for_heldout(
        [level.level_id for level in levels],
        heldout_fraction=config.library.heldout_level_fraction,
        seed=config.experiment.seed,
    )
    censuses = []
    for raw in levels:
        if raw.level_id not in train_ids:
            continue
        grid = LevelGrid(
            tiles=raw.tiles,
            width_tiles=raw.tiles.shape[1],
            height_tiles=raw.tiles.shape[0],
            tile_size_px=config.game.tile_size_px,
            area_type=AreaType(config.game.area_type),
            metadata={"level_id": raw.level_id},
        )
        censuses.append(structural_census(grid, catalog, config))
    return merge_censuses(censuses)


def _duplicate_rate(chunk_ids: tuple[int, ...]) -> float:
    """Fraction of decisions repeating an already-used chunk."""
    if not chunk_ids:
        return 0.0
    return 1.0 - len(set(chunk_ids)) / len(chunk_ids)


def run_comparison(
    config: RuntimeConfig,
    *,
    systems: list[str],
    episodes: int = 3,
    checkpoint: Path | None = None,
    with_bridge: bool = False,
    target_source: str = "oracle",
    output_dir: Path | None = None,
) -> Path:
    """Run the consolidated system comparison over the evaluation target suite.

    Description:
        Every system runs the same protocol: target suite x shared seeds x
        episodes, scored with the composer terminal reward, per-dimension
        intent errors, heldout TPKL, census total-variation distances, and a
        duplicate rate. ``corpus_replay`` levels carry no chunk ids, so their
        duplicate component is 0 by construction (documented in the report).

    Args:
        config: Runtime settings.
        systems: System names from ``SYSTEM_NAMES``.
        episodes: Episodes per (system, case).
        checkpoint: Policy archive for the ppo systems.
        with_bridge: Whether to run the A* bridge per level (slow).
        target_source: ``oracle`` feeds suite vectors directly; ``llm`` and
            ``fake`` verbalize each case target, translate it, and condition
            the systems on the translated vector while still scoring against
            the oracle reference (per-stage error columns appear).
        output_dir: Destination directory; timestamped default otherwise.

    Returns:
        Path to the written ``comparison.csv``.

    Raises:
        ValueError: For unknown system names or missing checkpoints.
        OSError: When report fragments cannot be written.
    """
    unknown = set(systems) - set(SYSTEM_NAMES)
    if unknown:
        raise ValueError(f"unknown systems: {sorted(unknown)}")
    if target_source not in ("oracle", "llm", "fake"):
        raise ValueError(f"unknown target source {target_source!r}")
    translator: Any | None = None
    if target_source == "fake":
        from pcgrl.app.intent.fake import FakeIntentTranslator

        translator = FakeIntentTranslator(config)
    elif target_source == "llm":
        from pcgrl.app.intent.factory import build_translator

        translator = build_translator(config)
    catalog = load_catalog(config)
    heldout_reference = PatternReference.load(
        resolve_path(config.dataset.processed_root), split="heldout"
    )
    train_reference = PatternReference.load(
        resolve_path(config.dataset.processed_root), split="train"
    )
    terminal = ComposerTerminalReward(config, catalog, train_reference)
    corpus_census = _corpus_census(config, catalog)
    empty_id = catalog.id_for_name("empty")
    bridge = None
    if with_bridge:
        from pcgrl.bridge.bridge import GameplayBridge

        bridge = GameplayBridge(config)
        bridge.ensure_available()
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ensure_dir(output_dir or resolve_path("artifacts/reports/comparison") / timestamp)
    generate_fns = {
        system: build_generate_fn(config, system, checkpoint=checkpoint) for system in systems
    }
    requested_by_case: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for system in systems:
        for case_index, case in enumerate(config.evaluation.target_suite):
            target = np.array(config.build_target_vector(**case.overrides), dtype=np.float32)
            requested = target
            if translator is not None:
                if case.name not in requested_by_case:
                    from pcgrl.app.intent.verbalizer import verbalize

                    translation = translator.translate(verbalize(target, config))
                    requested_by_case[case.name] = np.asarray(translation.vector, dtype=np.float32)
                requested = requested_by_case[case.name]
            for episode in range(episodes):
                seed = config.experiment.seed + 7_919 * case_index + episode
                rows.append(
                    _score_level(
                        system=system,
                        case=case.name,
                        seed=seed,
                        episode=episode,
                        target_source=target_source,
                        target=target,
                        requested=requested,
                        generate_fn=generate_fns[system],
                        config=config,
                        catalog=catalog,
                        terminal=terminal,
                        heldout_reference=heldout_reference,
                        corpus_census=corpus_census,
                        empty_id=empty_id,
                        bridge=bridge,
                    )
                )
        logger.info("compare: system %s done (%d rows)", system, episodes)
    csv_path = _write_fragments(rows, systems, out_dir)
    return csv_path


def _score_level(
    *,
    system: str,
    case: str,
    seed: int,
    episode: int,
    target_source: str,
    target: np.ndarray,
    requested: np.ndarray,
    generate_fn: GenerateFn,
    config: RuntimeConfig,
    catalog: TileCatalog,
    terminal: ComposerTerminalReward,
    heldout_reference: PatternReference,
    corpus_census: dict[str, object],
    empty_id: int,
    bridge: Any | None,
) -> dict[str, object]:
    """Generate and score one level into a unified comparison row.

    Description:
        ``requested`` is what the system is conditioned on (the translated
        vector under an llm/fake target source); errors and the terminal
        reward are always scored against ``target`` (the oracle reference,
        read through ``effective_target`` like every generator input), so the
        oracle-vs-llm gap on the same system is exactly the translation
        stage's end-to-end cost.
    """
    target = effective_target(target)
    start = time.perf_counter()
    level = generate_fn(requested, seed)
    wall_time = time.perf_counter() - start
    chunk_ids = tuple(int(c) for c in level.metadata.get("chunk_ids", ()))
    measured = compute_intent_vector(level, catalog, config)
    reward, components = terminal.compute(level, target, chunk_ids=chunk_ids, measured=measured)
    deltas = np.abs(measured.astype(np.float64) - target.astype(np.float64))
    row: dict[str, object] = {
        "system": system,
        "case": case,
        "seed": seed,
        "episode": episode,
        "target_source": target_source,
        "wall_time_s": round(wall_time, 4),
        "terminal_reward": round(float(reward), 4),
        "component_alignment": round(components["alignment"], 4),
        "component_style": round(components["style"], 4),
        "component_playability": round(components["playability"], 4),
        "component_duplicate": round(components["dup_penalty"], 4),
        "intent_error_mean": round(float(deltas.mean()), 4),
        "intent_error_max": round(float(deltas.max()), 4),
        "duplicate_rate": round(_duplicate_rate(chunk_ids), 4),
    }
    if target_source != "oracle":
        translation_deltas = np.abs(requested.astype(np.float64) - target.astype(np.float64))
        policy_deltas = np.abs(measured.astype(np.float64) - requested.astype(np.float64))
        row["translation_error_mean"] = round(float(translation_deltas.mean()), 4)
        row["translation_error_max"] = round(float(translation_deltas.max()), 4)
        row["policy_error_mean"] = round(float(policy_deltas.mean()), 4)
        row["policy_error_max"] = round(float(policy_deltas.max()), 4)
    for index, dim in enumerate(INTENT_KEYS):
        row[f"error_{dim}"] = round(float(deltas[index]), 4)
    for size in config.evaluation.tpkl_sizes:
        divergence = tile_pattern_kl(
            level.tiles,
            heldout_reference,
            size,
            exclude=empty_pattern_key(empty_id, size),
        )
        row[f"tpkl_heldout_k{size}"] = round(float(divergence), 4)
    distances = census_distance(corpus_census, structural_census(level, catalog, config))
    for key, value in distances.items():
        row[f"census_{key}"] = value
    if bridge is not None:
        result = bridge.evaluate(
            level,
            agent=config.evaluation.agents[0],
            timer_seconds=config.evaluation.timer_seconds,
            mario_state=config.evaluation.mario_state,
            num_runs=config.generation.bridge_num_runs,
        )
        row["bridge_won"] = bool(result.won)
        row["bridge_completion_ratio"] = round(float(result.completion_ratio), 4)
    return row


def _write_fragments(rows: list[dict[str, object]], systems: list[str], out_dir: Path) -> Path:
    """Write comparison.csv, summary.json, pairwise.json, and comparison.md."""
    csv_path = out_dir / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for system in systems:
        metrics: dict[str, dict[str, float]] = {}
        for metric in _SUMMARY_METRICS:
            values = [float(row[metric]) for row in rows if row["system"] == system]  # type: ignore[arg-type]
            low, high = bootstrap_ci(values)
            metrics[metric] = {
                "mean": round(float(np.mean(values)), 4),
                "std": round(float(np.std(values)), 4),
                "n": len(values),
                "ci95_low": round(low, 4),
                "ci95_high": round(high, 4),
            }
        summary[system] = metrics
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pairwise: list[dict[str, object]] = []
    for index, system_a in enumerate(systems):
        for system_b in systems[index + 1 :]:
            for metric in _PAIRWISE_METRICS:
                a = [float(row[metric]) for row in rows if row["system"] == system_a]  # type: ignore[arg-type]
                b = [float(row[metric]) for row in rows if row["system"] == system_b]  # type: ignore[arg-type]
                pairwise.append(
                    {
                        "a": system_a,
                        "b": system_b,
                        "metric": metric,
                        "delta": round(float(np.mean(a) - np.mean(b)), 4),
                        "p_value": round(permutation_pvalue(a, b), 5),
                    }
                )
    (out_dir / "pairwise.json").write_text(json.dumps(pairwise, indent=2), encoding="utf-8")
    lines = [
        "# System comparison",
        "",
        "corpus_replay carries no chunk ids, so its duplicate component is 0 by construction.",
        "",
        "| system | terminal_reward (95% CI) | intent_error_mean | duplicate_rate | wall_time_s |",
        "|---|---|---|---|---|",
    ]
    for system in systems:
        metrics = summary[system]
        reward = metrics["terminal_reward"]
        lines.append(
            f"| {system} | {reward['mean']:.3f} "
            f"[{reward['ci95_low']:.3f}, {reward['ci95_high']:.3f}] "
            f"| {metrics['intent_error_mean']['mean']:.3f} "
            f"| {metrics['duplicate_rate']['mean']:.3f} "
            f"| {metrics['wall_time_s']['mean']:.2f} |"
        )
    (out_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("compare: fragments written to %s", out_dir)
    return csv_path
