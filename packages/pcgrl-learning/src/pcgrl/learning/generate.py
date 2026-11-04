from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from pcgrl.bridge.bridge import GameplayBridge
from pcgrl.core.domain.level import LevelGrid
from pcgrl.core.helpers.path import ensure_dir, resolve_path
from pcgrl.core.settings.schema import RuntimeConfig
from pcgrl.core.tiles.catalog import TileCatalog
from pcgrl.learning.env_factory import build_env_context, make_env
from pcgrl.rendering.ascii import level_to_ascii
from pcgrl.rendering.png import PngRenderer
from pcgrl.runtime.metrics.intent import compute_intent_vector
from pcgrl.runtime.refine import IntentRefiner
from pcgrl.runtime.validators.runner import ValidatorRunner

MaskablePPO = None

SUCCESS_PLAYABLE = "SUCCESS_PLAYABLE"
FAIL_NO_PLAYABLE_CANDIDATE = "FAIL_NO_PLAYABLE_CANDIDATE"
FAIL_STRUCTURAL_INVALID = "FAIL_STRUCTURAL_INVALID"
FAIL_HEURISTIC_UNREACHABLE = "FAIL_HEURISTIC_UNREACHABLE"
FAIL_BRIDGE_UNAVAILABLE = "FAIL_BRIDGE_UNAVAILABLE"
FAIL_BRIDGE_NOT_WON = "FAIL_BRIDGE_NOT_WON"

# Priority order for resolving a single generation_status from the gate
# failures of the selected candidate: structural breakage outranks an
# unreachable flood route, which outranks a bridge loss.
_FAILURE_PRIORITY: tuple[str, ...] = (
    FAIL_STRUCTURAL_INVALID,
    FAIL_HEURISTIC_UNREACHABLE,
    FAIL_BRIDGE_NOT_WON,
)


def _sort_failures(reasons: list[str]) -> list[str]:
    """De-duplicate and order failure reasons by functional priority."""
    order = {name: index for index, name in enumerate(_FAILURE_PRIORITY)}
    seen: list[str] = []
    for reason in reasons:
        if reason not in seen:
            seen.append(reason)
    return sorted(seen, key=lambda reason: order.get(reason, len(order)))


def _rollout(
    env: Any, model: Any, target_vector: np.ndarray, seed: int
) -> tuple[LevelGrid, float, dict[str, Any], bool]:
    """Roll one masked episode and return its level and terminal transition.

    Args:
        env: Composer environment (duck-typed: reset/step/action_masks/get_level).
        model: Policy with a ``predict(obs, action_masks=...)`` method.
        target_vector: Intent vector conditioning the episode.
        seed: Episode RNG seed.

    Returns:
        Tuple of (level, final step reward, final step info, truncated flag).
    """
    obs, _ = env.reset(seed=seed, options={"target": target_vector})
    reward = 0.0
    info: dict[str, Any] = {}
    truncated = False
    done = False
    while not done:
        action_masks = env.action_masks()
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
    return env.get_level(), float(reward), dict(info), bool(truncated)


def _bridge_stats(
    bridge: GameplayBridge, level: LevelGrid, config: RuntimeConfig
) -> dict[str, Any]:
    """Evaluate one level on the gameplay bridge and flatten the result.

    Args:
        bridge: Gameplay bridge wrapper.
        level: Level to evaluate.
        config: Runtime settings with evaluation agent and run-count knobs.

    Returns:
        Flat dict with won/completion/win-rate/run-count/status fields.
    """
    result = bridge.evaluate(
        level,
        agent=config.evaluation.agents[0],
        timer_seconds=config.evaluation.timer_seconds,
        mario_state=config.evaluation.mario_state,
        num_runs=config.generation.bridge_num_runs,
    )
    won = bool(result.won)
    return {
        "bridge_won": won,
        "bridge_completion_ratio": float(result.completion_ratio),
        "bridge_win_rate": float(getattr(result, "bridge_win_rate", 1.0 if won else 0.0)),
        "bridge_runs": int(getattr(result, "bridge_runs", config.generation.bridge_num_runs)),
        "bridge_completion_std": float(getattr(result, "bridge_completion_std", 0.0)),
        "bridge_game_status": str(getattr(result, "game_status", "UNKNOWN")),
    }


def _persist_generation_report(
    *,
    stem: Path,
    level: LevelGrid | None,
    catalog: TileCatalog,
    target_vector: np.ndarray,
    diagnostics: dict[str, object],
    save_png: bool,
    tile_size_px: int,
) -> Path:
    """Write ASCII, optional PNG, and JSON diagnostics for a generation attempt."""
    ascii_level = level_to_ascii(level, catalog) if level is not None else ""
    stem.with_suffix(".ascii").write_text(ascii_level, encoding="utf-8")
    if save_png and level is not None:
        PngRenderer(catalog, tile_size_px=tile_size_px).render(level).save(stem.with_suffix(".png"))
    payload = {
        "status": diagnostics.get("status", FAIL_NO_PLAYABLE_CANDIDATE),
        "success": diagnostics.get("status") == SUCCESS_PLAYABLE,
        "target": [float(x) for x in target_vector.tolist()],
        "ascii": ascii_level,
        "diagnostics": diagnostics,
    }
    out = stem.with_suffix(".json")
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def _candidate_diagnostics(candidate: dict[str, Any]) -> dict[str, Any]:
    """Project one candidate dict onto the frozen report-diagnostics contract."""
    return {
        key: value for key, value in candidate.items() if key not in ("level", "attempt", "seed")
    }


def _polish_selected(
    selected: dict[str, Any],
    *,
    config: RuntimeConfig,
    context: Any,
    structural: ValidatorRunner,
    terminal: Any,
    bridge: Any,
    target_vector: np.ndarray,
    polish: bool | None,
) -> dict[str, Any]:
    """Polish the selected candidate toward the target, re-gating the result.

    Description:
        Runs the deterministic ``IntentRefiner`` (the same op vocabulary the
        policy's edit phase uses) as a hard-tolerance safety net, then
        re-validates: structural chain, flood playability via the terminal
        reward, and the A* bridge when required. On success the selected
        candidate is updated IN PLACE so every downstream field (measured
        vector, errors, reward, bridge stats) describes the shipped level;
        on any gate failure the unpolished candidate stands and the reason
        is recorded.

    Returns:
        The ``polish`` diagnostics block.
    """
    polish_cfg = config.generation.polish
    enabled = polish_cfg.enabled if polish is None else bool(polish)
    block: dict[str, Any] = {"enabled": enabled, "applied": False}
    if not enabled:
        return block
    refiner = IntentRefiner(
        config,
        context.catalog,
        tolerance=polish_cfg.tolerance,
        max_passes=polish_cfg.max_passes,
    )
    result = refiner.refine(selected["level"], target_vector)
    block.update(
        {
            "converged": result.converged,
            "passes": result.passes,
            "edits_accepted": sum(1 for e in result.edits if e.accepted),
            "edits_rejected": sum(1 for e in result.edits if not e.accepted),
            "measured_before": list(result.measured_before),
            "measured_after": list(result.measured_after),
            "mean_error_before": float(
                np.mean(np.abs(np.asarray(result.measured_before) - target_vector))
            ),
            "mean_error_after": float(
                np.mean(np.abs(np.asarray(result.measured_after) - target_vector))
            ),
        }
    )
    if not result.applied:
        block["fallback_reason"] = "no_edits_applied"
        return block
    gen_cfg = config.generation
    validation = structural.validate(result.level)
    chunk_ids = tuple(int(c) for c in selected["chunk_ids"])
    measured = np.asarray(result.measured_after, dtype=np.float32)
    reward, components = terminal.compute(
        result.level, target_vector, chunk_ids=chunk_ids, measured=measured
    )
    flood_reachable = float(components.get("playability", 0.0)) >= 1.0
    if not validation.is_valid:
        block["fallback_reason"] = "structural_invalid_after_polish"
        return block
    if not flood_reachable:
        block["fallback_reason"] = "flood_unreachable_after_polish"
        return block
    bridge_fields: dict[str, Any] | None = None
    if gen_cfg.require_gameplay_bridge:
        bridge_fields = _bridge_stats(bridge, result.level, config)
        completion_ok = (
            float(bridge_fields["bridge_completion_ratio"])
            >= gen_cfg.minimum_completion_ratio
        )
        win_ok = (
            bool(bridge_fields["bridge_won"]) if gen_cfg.require_bridge_win else True
        )
        if not (completion_ok and win_ok):
            block["fallback_reason"] = "bridge_not_won_after_polish"
            return block
    per_dimension_error = [
        float(abs(m - t))
        for m, t in zip(measured.tolist(), target_vector.tolist(), strict=True)
    ]
    selected["level"] = result.level
    selected["terminal_reward"] = float(reward)
    selected["reward_components"] = {str(k): float(v) for k, v in components.items()}
    selected["measured_target"] = [float(x) for x in measured.tolist()]
    selected["per_dimension_error"] = per_dimension_error
    selected["mean_target_error"] = float(np.mean(per_dimension_error))
    selected["max_target_error"] = float(np.max(per_dimension_error))
    if bridge_fields is not None:
        selected.update(bridge_fields)
    block["applied"] = True
    block["fallback_reason"] = None
    return block


def run_generate(
    config: RuntimeConfig,
    checkpoint: Path,
    *,
    target: np.ndarray | None = None,
    output_dir: Path | None = None,
    seed: int | None = None,
    translation: Mapping[str, Any] | None = None,
    polish: bool | None = None,
) -> Path:
    """Generate a level from a trained composer policy with best-of-N selection.

    Description:
        Each attempt rolls one masked episode and is gated on the composer's
        own quality criteria: structural validity, flood reachability to the
        flag (the terminal reward's playability term at 1.0), and — when the
        bridge is required — an A* win / completion check. Candidates are
        ranked by the composer terminal reward, the training objective, so
        selection optimizes exactly what the policy was trained on. Truncated
        episodes have their reward recomputed from the stitched level because
        the final env step pays 0 in that case.

    Args:
        config: Runtime settings for environment, gates, and bridge evaluation.
        checkpoint: Path to a trained ``MaskablePPO`` policy archive.
        target: Optional intent vector; defaults to ``config.default_target_vector()``.
        output_dir: Directory for the JSON/ASCII/PNG report bundle.
        seed: Base RNG seed for rollout attempts.
        translation: Optional NL-translation provenance record (a plain
            mapping so learning never imports the app layer); lands in
            ``diagnostics["translation"]``.
        polish: Override for ``generation.polish.enabled`` — the
            deterministic hard-tolerance refinement of the selected
            candidate (evaluation harnesses pass ``False`` to measure the
            raw policy).

    Returns:
        Path to the written ``level_*.json`` report for the selected candidate.

    Raises:
        ValueError: When the configured environment lacks the composer reward.
        RuntimeError: When target feasibility fails, the bridge is required but
            unavailable, or no candidate passes the generation gates.
        FileNotFoundError: Propagated from bridge setup when the JAR is missing
            and ``require_gameplay_bridge`` is enabled.
    """
    global MaskablePPO
    if MaskablePPO is None:
        from sb3_contrib import MaskablePPO as _MaskablePPO

        MaskablePPO = _MaskablePPO

    context = build_env_context(config)
    if context.composer_terminal is None:
        raise ValueError(
            "run_generate requires a composer context with a terminal reward; "
            "run the 'prepare' and 'library' stages first"
        )
    model = MaskablePPO.load(str(checkpoint), device="cpu")
    target_vector = (
        target if target is not None else np.array(config.default_target_vector(), dtype=np.float32)
    )
    base_seed = seed if seed is not None else config.experiment.seed
    env = make_env(config, context, seed=base_seed)
    structural = ValidatorRunner(config, context.catalog)
    terminal = context.composer_terminal
    gen_cfg = config.generation
    bridge = GameplayBridge(config)
    out_dir = ensure_dir(output_dir or resolve_path("artifacts/reports/generated"))
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    stem = out_dir / f"level_{timestamp}"
    intent_dimensions = list(config.target_vector_field_names())

    if gen_cfg.require_gameplay_bridge:
        try:
            bridge.ensure_available()
        except FileNotFoundError as exc:
            out = _persist_generation_report(
                stem=stem,
                level=None,
                catalog=context.catalog,
                target_vector=target_vector,
                diagnostics={
                    "status": FAIL_BRIDGE_UNAVAILABLE,
                    "generation_status": FAIL_BRIDGE_UNAVAILABLE,
                    "bridge_available": False,
                    "error": str(exc),
                },
                save_png=False,
                tile_size_px=config.game.tile_size_px,
            )
            raise RuntimeError(
                f"Generation failed ({FAIL_BRIDGE_UNAVAILABLE}). See diagnostics at {out}"
            ) from exc

    bridge_start_attempt = max(
        0,
        int(np.floor(gen_cfg.max_attempts * gen_cfg.bridge_validation_start_fraction)),
    )
    candidates: list[dict[str, Any]] = []
    for attempt in range(gen_cfg.max_attempts):
        attempt_seed = base_seed + attempt
        level, reward, step_info, truncated = _rollout(env, model, target_vector, attempt_seed)
        chunk_ids = tuple(int(c) for c in step_info.get("chunk_ids", ()))
        components_raw = step_info.get("reward_components")
        measured = compute_intent_vector(level, context.catalog, config)
        reward_recomputed = truncated or not isinstance(components_raw, dict)
        if isinstance(components_raw, dict) and not reward_recomputed:
            components: dict[str, float] = {
                str(k): float(v) for k, v in components_raw.items()
            }
        else:
            reward, computed = terminal.compute(
                level, target_vector, chunk_ids=chunk_ids, measured=measured
            )
            components = {str(k): float(v) for k, v in computed.items()}
        validation = structural.validate(level)
        hard_violations = sum(
            1 for violation in validation.violations if violation.severity == "hard"
        )
        # Single physics source: the gate reuses the terminal reward's
        # playability term, which floods under the calibrated jump config.
        # 1.0 means the flood reached the flag column.
        flood_reachable = float(components.get("playability", 0.0)) >= 1.0
        per_dimension_error = [
            float(abs(m - t)) for m, t in zip(measured.tolist(), target_vector.tolist(), strict=True)
        ]
        bridge_checked = bool(
            gen_cfg.require_gameplay_bridge
            and attempt >= bridge_start_attempt
            and validation.is_valid
            and flood_reachable
        )
        bridge_skipped_reason: str | None = None
        if not bridge_checked:
            if not gen_cfg.require_gameplay_bridge:
                bridge_skipped_reason = "bridge_not_required"
            elif not validation.is_valid:
                bridge_skipped_reason = "structural_invalid"
            elif not flood_reachable:
                bridge_skipped_reason = "heuristic_unreachable"
            else:
                bridge_skipped_reason = "attempt_below_start_fraction"
        bridge_fields: dict[str, Any] = {
            "bridge_won": None,
            "bridge_completion_ratio": None,
            "bridge_win_rate": 0.0,
            "bridge_runs": 0,
            "bridge_completion_std": 0.0,
            "bridge_game_status": None,
        }
        bridge_ok = not gen_cfg.require_gameplay_bridge
        if bridge_checked:
            bridge_fields = _bridge_stats(bridge, level, config)
            completion_ok = (
                float(bridge_fields["bridge_completion_ratio"]) >= gen_cfg.minimum_completion_ratio
            )
            win_ok = bool(bridge_fields["bridge_won"]) if gen_cfg.require_bridge_win else True
            bridge_ok = completion_ok and win_ok
        failure_reasons: list[str] = []
        if not validation.is_valid:
            failure_reasons.append(FAIL_STRUCTURAL_INVALID)
        if not flood_reachable:
            failure_reasons.append(FAIL_HEURISTIC_UNREACHABLE)
        if bridge_checked and not bridge_ok:
            failure_reasons.append(FAIL_BRIDGE_NOT_WON)
        failure_reasons = _sort_failures(failure_reasons)
        final_playable = (
            bool(validation.is_valid)
            and flood_reachable
            and (bridge_ok if gen_cfg.require_gameplay_bridge else True)
            and (bridge_checked or not gen_cfg.require_gameplay_bridge)
        )
        candidates.append(
            {
                "attempt": attempt,
                "seed": attempt_seed,
                "level": level,
                "terminal_reward": float(reward),
                "reward_components": components,
                "reward_recomputed": bool(reward_recomputed),
                "truncated_episode": bool(truncated),
                "chunk_ids": list(chunk_ids),
                "num_decisions": len(chunk_ids),
                "structural_valid": bool(validation.is_valid),
                "hard_violations": int(hard_violations),
                "heuristic_reachable": bool(flood_reachable),
                "measured_target": [float(x) for x in measured.tolist()],
                "per_dimension_error": per_dimension_error,
                "mean_target_error": float(np.mean(per_dimension_error)),
                "max_target_error": float(np.max(per_dimension_error)),
                "bridge_checked": bridge_checked,
                "bridge_skipped_reason": bridge_skipped_reason,
                "failure_reasons": failure_reasons,
                "final_playable": bool(final_playable),
                "edit_log": list(step_info.get("edit_log", [])),
                **bridge_fields,
            }
        )
    playable_candidates = [
        candidate for candidate in candidates if bool(candidate["final_playable"])
    ]
    if not playable_candidates:
        # Prefer a structurally valid + reachable level for the diagnostic
        # report even when no candidate cleared the bridge; fall back to the
        # full pool only when nothing is reachable at all.
        reachable_pool = [
            candidate
            for candidate in candidates
            if bool(candidate["structural_valid"]) and bool(candidate["heuristic_reachable"])
        ]
        selection_pool = reachable_pool if reachable_pool else candidates
        best = max(selection_pool, key=lambda item: float(item["terminal_reward"]))
        candidate_reasons = _sort_failures(list(best.get("failure_reasons", [])))
        status = str(candidate_reasons[0]) if candidate_reasons else FAIL_NO_PLAYABLE_CANDIDATE
        diagnostics = {
            "status": status,
            "generation_status": status,
            "attempts_total": len(candidates),
            "attempts_playable": 0,
            "selection_mode": f"best_of_{gen_cfg.max_attempts}",
            "selected_attempt": int(best["attempt"]),
            "selected_seed": int(best["seed"]),
            "intent_dimensions": intent_dimensions,
            "bridge_available": bool(bridge.jar_path.is_file()),
            **_candidate_diagnostics(best),
            "final_playable": False,
            "all_attempts": [
                {key: value for key, value in candidate.items() if key != "level"}
                for candidate in candidates
            ],
        }
        if translation is not None:
            diagnostics["translation"] = dict(translation)
        out = _persist_generation_report(
            stem=stem,
            level=best["level"],
            catalog=context.catalog,
            target_vector=target_vector,
            diagnostics=diagnostics,
            save_png=True,
            tile_size_px=config.game.tile_size_px,
        )
        raise RuntimeError(f"Generation failed ({status}). See diagnostics at {out}")
    selected = max(playable_candidates, key=lambda item: float(item["terminal_reward"]))
    polish_block = _polish_selected(
        selected,
        config=config,
        context=context,
        structural=structural,
        terminal=terminal,
        bridge=bridge,
        target_vector=target_vector,
        polish=polish,
    )
    diagnostics = {
        "status": SUCCESS_PLAYABLE,
        "generation_status": SUCCESS_PLAYABLE,
        "attempts_total": len(candidates),
        "attempts_playable": len(playable_candidates),
        "selection_mode": f"best_of_{gen_cfg.max_attempts}",
        "selected_attempt": int(selected["attempt"]),
        "selected_seed": int(selected["seed"]),
        "intent_dimensions": intent_dimensions,
        "bridge_available": bool(bridge.jar_path.is_file()),
        **_candidate_diagnostics(selected),
        "polish": polish_block,
        "all_attempts": [
            {key: value for key, value in candidate.items() if key != "level"}
            for candidate in candidates
        ],
    }
    if translation is not None:
        diagnostics["translation"] = dict(translation)
    return _persist_generation_report(
        stem=stem,
        level=selected["level"],
        catalog=context.catalog,
        target_vector=target_vector,
        diagnostics=diagnostics,
        save_png=True,
        tile_size_px=config.game.tile_size_px,
    )
