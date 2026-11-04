from __future__ import annotations

import numpy as np

from pcgrl.runtime.environment.composer_state import ComposerCore, ComposeState
from pcgrl.runtime.metrics.intent import INTENT_KEYS, compute_intent_vector

_EDGE_FEATURES = 3
_CHUNK_SUMMARY_FEATURES = 4 + 1 + 3
_PROGRESS_FEATURES = 2
_PHASE_FEATURES = 2
_FAMILY_ORDER = ("start", "middle", "ending")


def composer_observation_size() -> int:
    """Return the length of the flat composer observation vector.

    Returns:
        Total number of features (target + measured + delta + seam +
        progress + edit-phase flag and budget).
    """
    return (
        3 * len(INTENT_KEYS)
        + _EDGE_FEATURES
        + _CHUNK_SUMMARY_FEATURES
        + _PROGRESS_FEATURES
        + _PHASE_FEATURES
    )


def measured_partial_intent(core: ComposerCore, state: ComposeState) -> np.ndarray:
    """Intent vector of the partially stitched composition.

    Description:
        Stitching stamps the endpoints, so the reachability flood has a player
        spawn; at 10-19 evaluations per episode this is cheap.

    Args:
        core: See signature.
        state: See signature.

    Returns:
        np.ndarray: Result of the operation.
    """
    if state.num_decisions == 0:
        return np.zeros(len(INTENT_KEYS), dtype=np.float32)
    level = core.stitch_level(state)
    return compute_intent_vector(level, core.catalog, core.config)


def _last_chunk_features(
    core: ComposerCore, state: ComposeState
) -> tuple[float, float, float, np.ndarray, float, np.ndarray]:
    """Seam profile and summary of the most recently placed chunk."""
    if not state.num_decisions:
        return 0.0, 0.0, 0.0, np.zeros(4, dtype=np.float32), 0.0, np.zeros(3, dtype=np.float32)
    config = core.config
    last_id = state.chunk_ids[-1]
    last = core.library[last_id]
    exit_height = last.exit_height / max(1, config.game.height_tiles)
    trailing_gap = core.trailing_gap(last_id) / max(1, config.validation.jump_reach)
    gap_flag = 1.0 if last.gap_at_right else 0.0
    chunk_area = max(1, last.width * last.height)
    counts = core.budget_counts(last_id) / chunk_area
    width_fraction = last.width / max(1, config.game.width_tiles)
    family = np.zeros(3, dtype=np.float32)
    family[_FAMILY_ORDER.index(last.family.value)] = 1.0
    return exit_height, min(1.0, trailing_gap), gap_flag, counts, width_fraction, family


def build_composer_observation(
    core: ComposerCore,
    state: ComposeState,
    target: np.ndarray,
    *,
    measured: np.ndarray | None = None,
    max_decisions: int,
    edit_phase: bool = False,
    edit_budget_fraction: float = 0.0,
) -> np.ndarray:
    """Flat observation for the composer environment, every feature in [-1, 1].

    Description:
        Layout: intent target (8) + measured intent (8) + delta (8) +
        seam profile (exit height, trailing gap, gap flag) + last-chunk summary
        (budget counts, width fraction, family one-hot) + progress (width used,
        decisions made) + phase (edit flag, remaining edit budget). During the
        edit phase ``measured`` is the edited level's full intent vector.

    Args:
        core: See signature.
        state: See signature.
        target: See signature.
        measured: See signature.
        max_decisions: See signature.
        edit_phase: Whether the episode is in the edit phase.
        edit_budget_fraction: Remaining edit budget in ``[0, 1]``.

    Returns:
        np.ndarray: Result of the operation.
    """
    config = core.config
    if measured is None:
        measured = measured_partial_intent(core, state)
    target = np.asarray(target, dtype=np.float32)
    delta = target - measured
    exit_height, trailing_gap, gap_flag, counts, width_fraction, family = (
        _last_chunk_features(core, state)
    )
    progress = state.used_width / max(1, config.game.width_tiles)
    decisions = state.num_decisions / max(1, max_decisions)
    observation = np.concatenate(
        [
            target,
            measured.astype(np.float32),
            delta.astype(np.float32),
            np.array([exit_height, trailing_gap, gap_flag], dtype=np.float32),
            counts.astype(np.float32),
            np.array([width_fraction], dtype=np.float32),
            family,
            np.array([progress, decisions], dtype=np.float32),
            np.array(
                [1.0 if edit_phase else 0.0, edit_budget_fraction],
                dtype=np.float32,
            ),
        ]
    )
    return np.clip(observation, -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]
