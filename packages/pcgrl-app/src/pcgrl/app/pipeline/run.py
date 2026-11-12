from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pcgrl.core.helpers.path import ensure_dir, resolve_path
from pcgrl.core.settings.schema import RuntimeConfig


@dataclass(frozen=True)
class RunPipelineResult:
    """Paths and status produced by an end-to-end experiment run.

    Attributes:
        checkpoint: Final or reused policy checkpoint used for generation.
        level_json: Path to the generated level JSON report, if any.
        summary: Path to the Markdown generation summary, if written.
        generation_succeeded: Whether generation passed all playable gates.
    """

    checkpoint: Path
    level_json: Path | None
    summary: Path | None
    generation_succeeded: bool


def latest_checkpoint(config: RuntimeConfig) -> Path | None:
    """Return the newest final policy checkpoint for the configured experiment.

    Args:
        config: Runtime settings with ``experiment.name`` and artifacts root.

    Returns:
        Newest ``policy_final.zip`` under the experiment tree, or ``None``.
    """
    runs_root = resolve_path(config.experiment.artifacts_root)
    candidates = sorted(
        runs_root.glob(f"{config.experiment.name}/**/policy_final.zip"),
        key=lambda path: path.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def write_generation_summary(
    *,
    level_json: Path,
    summary_path: Path,
    config: RuntimeConfig,
    training_tier: str,
) -> Path:
    """Write a compact Markdown verdict from a generated level JSON report.

    Args:
        level_json: Generated level report produced by ``run_generate``.
        summary_path: Destination Markdown path.
        config: Runtime settings for gate thresholds referenced in the summary.
        training_tier: Training tier label included in the report header.

    Returns:
        Path to the written summary file.

    Raises:
        OSError: When the summary file cannot be written.
        json.JSONDecodeError: When ``level_json`` is not valid JSON.
    """
    payload = json.loads(level_json.read_text(encoding="utf-8"))
    diagnostics = payload.get("diagnostics", {})
    if not isinstance(diagnostics, dict):
        diagnostics = {}

    def as_float(name: str, default: float = 0.0) -> float:
        """Parse a diagnostic field as float, returning ``default`` on failure."""
        try:
            return float(diagnostics.get(name, default))
        except (TypeError, ValueError):
            return default

    def as_bool(name: str, default: bool = False) -> bool:
        """Parse a diagnostic field as bool with a fallback default."""
        return bool(diagnostics.get(name, default))

    status = str(payload.get("status", diagnostics.get("status", "UNKNOWN")))
    generation_status = str(diagnostics.get("generation_status", status))

    checks: list[tuple[str, bool, str]] = [
        (
            "Generation status is SUCCESS_PLAYABLE",
            generation_status == "SUCCESS_PLAYABLE",
            generation_status,
        ),
        (
            "Structurally valid",
            as_bool("structural_valid"),
            str(as_bool("structural_valid")),
        ),
        (
            "Flood reachable (spawn -> flag)",
            as_bool("heuristic_reachable"),
            str(as_bool("heuristic_reachable")),
        ),
    ]
    if as_bool("bridge_checked"):
        checks.append(
            (
                f"Completion >= {config.generation.minimum_completion_ratio:.2f}",
                as_float("bridge_completion_ratio") >= config.generation.minimum_completion_ratio,
                f"{as_float('bridge_completion_ratio'):.4f}",
            )
        )
        if config.generation.require_bridge_win:
            checks.append(("Bridge won", as_bool("bridge_won"), str(as_bool("bridge_won"))))

    failed = [name for name, ok, _ in checks if not ok]
    verdict = "PASS" if not failed else "FAIL"
    diagnostic_keys = (
        "terminal_reward",
        "reward_components",
        "reward_recomputed",
        "truncated_episode",
        "num_decisions",
        "chunk_ids",
        "structural_valid",
        "hard_violations",
        "heuristic_reachable",
        "measured_target",
        "per_dimension_error",
        "mean_target_error",
        "max_target_error",
        "intent_dimensions",
        "attempts_total",
        "attempts_judged",
        "selected_attempt",
        "failure_reasons",
        "bridge_checked",
        "bridge_skipped_reason",
        "bridge_won",
        "bridge_win_rate",
        "bridge_completion_ratio",
        "bridge_completion_std",
        "bridge_game_status",
    )

    lines = [
        "# Map Result Summary",
        "",
        f"- Verdict: **{verdict}**",
        f"- Status: `{status}`",
        f"- Generation status: `{generation_status}`",
        f"- Training tier: `{training_tier}`",
        "",
        "## Checks",
    ]
    for name, ok, value in checks:
        marker = "PASS" if ok else "FAIL"
        lines.append(f"- [{marker}] {name} (value: `{value}`)")
    if not as_bool("bridge_checked"):
        reason = diagnostics.get("bridge_skipped_reason") or "not_evaluated"
        lines.append(f"- [SKIP] Bridge checks (bridge did not run: `{reason}`)")
    lines.extend(["", "## Key Diagnostics"])
    for key in diagnostic_keys:
        if key in diagnostics:
            lines.append(f"- {key}: `{diagnostics[key]}`")

    ensure_dir(summary_path.parent)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def _corpus_has_levels(config: RuntimeConfig) -> bool:
    """Return whether the corpus root contains at least one VGLC level file."""
    corpus_root = resolve_path(config.dataset.corpus_root)
    return any(path.stem != "symbols" for path in corpus_root.rglob("*.txt"))


def _processed_corpus_exists(config: RuntimeConfig) -> bool:
    """Return whether the prepare outputs the composer needs are on disk."""
    processed_root = resolve_path(config.dataset.processed_root)
    return (processed_root / "pattern_distributions.npz").is_file() and (
        processed_root / "corpus_profile.json"
    ).is_file()


def _chunk_library_exists(config: RuntimeConfig) -> bool:
    """Return whether a chunk library built under the current config exists.

    Description:
        The library JSON sidecar records a ``library_config_hash``; a missing
        or mismatched hash marks the library stale so the pipeline rebuilds it.
    """
    from pcgrl.app.pipeline.library import library_config_hash

    library_root = resolve_path(config.library.output_root)
    json_path = library_root / "chunk_library.json"
    if not (library_root / "chunk_library.npz").is_file() or not json_path.is_file():
        return False
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    metadata = payload.get("metadata") or {}
    return metadata.get("library_config_hash") == library_config_hash(config)


def _latest_generated_json(output_dir: Path) -> Path | None:
    """Return the most recently written ``level_*.json`` in ``output_dir``."""
    candidates = sorted(output_dir.glob("level_*.json"), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def run_experiment(
    config: RuntimeConfig,
    *,
    training_tier: str,
    checkpoint: Path | None = None,
    output_dir: Path | None = None,
) -> RunPipelineResult:
    """Run prepare, library, train, generate, and summary as one workflow.

    Args:
        config: Runtime settings for the full pipeline.
        training_tier: Label recorded in the generation summary.
        checkpoint: Optional existing checkpoint to skip training.
        output_dir: Directory for generation reports.

    Returns:
        Paths and success flag for the completed experiment run.

    Raises:
        FileNotFoundError: When the corpus is empty or no checkpoint can be found.
        RuntimeError: When generation fails without a diagnostic JSON artifact.
    """
    from pcgrl.app.pipeline.library import run_library
    from pcgrl.app.pipeline.prepare import run_prepare
    from pcgrl.learning.generate import run_generate
    from pcgrl.learning.training.runner import run_train

    if not _corpus_has_levels(config):
        raise FileNotFoundError(
            f"No corpus levels found in {resolve_path(config.dataset.corpus_root)}. "
            "Run `make vglc` first."
        )
    if not _processed_corpus_exists(config):
        run_prepare(config)
    if not _chunk_library_exists(config):
        run_library(config)

    selected_checkpoint = checkpoint
    if selected_checkpoint is None:
        selected_checkpoint = run_train(config)
    if selected_checkpoint is None or not selected_checkpoint.is_file():
        fallback = latest_checkpoint(config)
        if fallback is None:
            raise FileNotFoundError("No policy_final.zip checkpoint was found.")
        selected_checkpoint = fallback

    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    report_dir = ensure_dir(
        output_dir
        or resolve_path("artifacts/reports/generated")
        / f"{config.experiment.name}_{training_tier}_{timestamp}"
    )
    generation_succeeded = True
    try:
        level_json = run_generate(config, selected_checkpoint, output_dir=report_dir)
    except RuntimeError:
        generation_succeeded = False
        latest_json = _latest_generated_json(report_dir)
        if latest_json is None:
            raise
        level_json = latest_json

    summary = write_generation_summary(
        level_json=level_json,
        summary_path=level_json.with_name("summary.md"),
        config=config,
        training_tier=training_tier,
    )
    return RunPipelineResult(
        checkpoint=selected_checkpoint,
        level_json=level_json,
        summary=summary,
        generation_succeeded=generation_succeeded,
    )
