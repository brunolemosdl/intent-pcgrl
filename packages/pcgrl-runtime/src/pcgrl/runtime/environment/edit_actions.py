from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pcgrl.runtime.refine.context import RefineContext
from pcgrl.runtime.refine.ops import (
    Edit,
    gap_runs,
    ground_heights,
    pipe_clusters,
    plan_add_blocks,
    plan_add_pipe,
    plan_add_platform,
    plan_add_tiles_on_surface,
    plan_dig_gap,
    plan_fill_gaps,
    plan_flatten,
    plan_raise_hill,
    plan_remove_blocks,
    plan_remove_pipes,
    plan_remove_platforms,
    plan_remove_tiles,
    plan_roughen,
    plan_smooth,
)

N_SEGMENTS = 8

_SPECS: tuple[tuple[str, int], ...] = (
    ("remove_enemies", 1),
    ("remove_coins", 1),
    ("remove_blocks", 1),
    ("remove_pipes", 1),
    ("remove_platforms", 1),
    ("fill_gaps", 1),
    ("flatten", 1),
    ("smooth", 1),
    ("add_enemies", 2),
    ("add_coins", 2),
    ("add_blocks", 2),
    ("add_pipe", 1),
    ("dig_gap", 2),
    ("raise_hill", 3),
    ("roughen", 2),
    ("add_platform", 2),
)


@dataclass(frozen=True)
class EditAction:
    """One decoded edit-phase action.

    Attributes:
        op: Operation name from the shared refine planner vocabulary.
        segment: Segment index in ``[0, N_SEGMENTS)``.
        param: Per-op coarse parameter level.
    """

    op: str
    segment: int
    param: int


def edit_action_count() -> int:
    """Number of edit actions including the terminal STOP."""
    per_segment = sum(levels for _, levels in _SPECS)
    return per_segment * N_SEGMENTS + 1


def stop_action_index() -> int:
    """Index of the STOP action inside the edit-action block."""
    return edit_action_count() - 1


def decode_edit_action(index: int) -> EditAction | None:
    """Decode an edit-action index; ``None`` is the STOP action.

    Args:
        index: Offset inside the edit-action block, in
            ``[0, edit_action_count())``.

    Raises:
        IndexError: When the index is outside the edit block.
    """
    if index == stop_action_index():
        return None
    if not 0 <= index < stop_action_index():
        raise IndexError(f"edit action index {index} out of range")
    per_segment = sum(levels for _, levels in _SPECS)
    segment, offset = divmod(index, per_segment)
    for op, levels in _SPECS:
        if offset < levels:
            return EditAction(op=op, segment=segment, param=offset)
        offset -= levels
    raise IndexError(f"edit action index {index} out of range")


def segment_columns(ctx: RefineContext, segment: int) -> range:
    """Column range covered by one segment."""
    width = ctx.width
    per_segment = max(1, width // N_SEGMENTS)
    start = segment * per_segment
    stop = width if segment == N_SEGMENTS - 1 else start + per_segment
    return range(start, stop)


def plan_edit_action(tiles: np.ndarray, ctx: RefineContext, action: EditAction) -> list[Edit]:
    """Plan the tile edits for one decoded action.

    Description:
        Bridges the policy's coarse (op, segment, param) vocabulary onto the
        shared refine planners, so the environment edit phase and the polish
        controller perform byte-identical writes for the same intent.

    Args:
        tiles: Working tile array.
        ctx: Shared refine context.
        action: Decoded edit action.

    Returns:
        Planned edits; empty when the op has nothing to do in the segment.
    """
    columns = segment_columns(ctx, action.segment)
    if action.op == "remove_enemies":
        return plan_remove_tiles(
            tiles, ctx, ctx.enemy_ids, keep=0, columns=columns, op="remove_enemies"
        )
    if action.op == "remove_coins":
        return plan_remove_tiles(
            tiles, ctx, (ctx.coin_id,), keep=0, columns=columns, op="remove_coins"
        )
    if action.op == "remove_blocks":
        return plan_remove_blocks(tiles, ctx, keep=0, columns=columns)
    if action.op == "remove_pipes":
        return plan_remove_pipes(tiles, ctx, keep_tiles=0, columns=columns)
    if action.op == "remove_platforms":
        return plan_remove_platforms(tiles, ctx, columns=columns)
    if action.op == "fill_gaps":
        return plan_fill_gaps(tiles, ctx, target_score=0.0, columns=columns)
    if action.op == "flatten":
        return plan_flatten(tiles, ctx, base_height=2, columns=columns)
    if action.op == "smooth":
        return plan_smooth(tiles, ctx, max_step=1, columns=columns)
    if action.op == "add_enemies":
        return plan_add_tiles_on_surface(
            tiles,
            ctx,
            ctx.enemy_ids[0],
            add=2 + 2 * action.param,
            columns=columns,
            op="add_enemies",
            spacing=3,
        )
    if action.op == "add_coins":
        return plan_add_tiles_on_surface(
            tiles,
            ctx,
            ctx.coin_id,
            add=3 + 3 * action.param,
            columns=columns,
            op="add_coins",
            height_above=2,
            spacing=2,
        )
    if action.op == "add_blocks":
        return plan_add_blocks(tiles, ctx, add=3, columns=columns, mountable=action.param == 0)
    if action.op == "add_pipe":
        return plan_add_pipe(tiles, ctx, pipe_height=3, columns=columns)
    if action.op == "dig_gap":
        return plan_dig_gap(tiles, ctx, gap_width=3 + action.param, columns=columns)
    if action.op == "raise_hill":
        peak = (4, 8, 11)[action.param]
        return plan_raise_hill(tiles, ctx, peak_height=2 + peak, ramp_step=1, columns=columns)
    if action.op == "roughen":
        return plan_roughen(tiles, ctx, amplitude=1 + action.param, boundaries=4, columns=columns)
    if action.op == "add_platform":
        height_above, width = ((2, 6), (3, 4))[action.param]
        return plan_add_platform(
            tiles, ctx, width=width, height_above=height_above, columns=columns
        )
    raise ValueError(f"unknown edit op {action.op!r}")


def edit_action_masks(tiles: np.ndarray, ctx: RefineContext) -> np.ndarray:
    """Cheap availability mask over the edit-action block (STOP always legal).

    Description:
        One pass computes per-segment summaries (category counts, gaps, flat
        ground share); each op level is unmasked only when its planner could
        plausibly act, so the policy never burns budget on structural no-ops.
        The mask is intentionally optimistic — a planner may still come back
        empty — because exact planning per action would cost ~200 planner
        runs per step.

    Args:
        tiles: Current working tile array.
        ctx: Shared refine context.

    Returns:
        Boolean array of length ``edit_action_count()``.
    """
    heights = ground_heights(tiles, ctx)
    clusters = pipe_clusters(tiles, ctx)
    cluster_starts = [cluster[0][1] for cluster in clusters]
    mask = np.zeros(edit_action_count(), dtype=bool)
    mask[stop_action_index()] = True
    enemy_grid = np.isin(tiles, list(ctx.enemy_ids))
    coin_grid = tiles == ctx.coin_id
    block_grid = np.isin(tiles, list(ctx.block_ids))
    solid_ratio = float(np.isin(tiles, list(ctx.solid_like)).sum() / max(1, ctx.area))
    solid_headroom = solid_ratio < ctx.config.validation.max_solid_ratio - 0.02
    index = 0
    for segment in range(N_SEGMENTS):
        columns = segment_columns(ctx, segment)
        sl = slice(columns.start, columns.stop)
        seg_heights = heights[columns.start : columns.stop]
        grounded = [h for h in seg_heights if h > 0]
        has_gap = any(columns.start <= start < columns.stop for start, _ in gap_runs(heights))
        has_pipe = any(columns.start <= col < columns.stop for col in cluster_starts)
        relief = max(grounded) - min(grounded) if grounded else 0
        free_ground = sum(
            1 for col in columns if heights[col] > 0 and col not in ctx.protected_columns
        )
        for op, levels in _SPECS:
            available = True
            if op == "remove_enemies":
                available = bool(enemy_grid[:, sl].any())
            elif op == "remove_coins":
                available = bool(coin_grid[:, sl].any())
            elif op == "remove_blocks":
                available = bool(block_grid[:, sl].any())
            elif op in ("remove_pipes", "remove_platforms"):
                available = has_pipe if op == "remove_pipes" else True
            elif op == "fill_gaps":
                available = has_gap
            elif op in ("flatten", "smooth"):
                available = relief > 0
            elif op in ("add_enemies", "add_coins", "add_blocks", "add_platform"):
                available = free_ground >= 4
            elif op == "add_pipe":
                available = free_ground >= 4 and solid_headroom
            elif op == "dig_gap":
                available = free_ground >= 8
            elif op in ("raise_hill", "roughen"):
                available = free_ground >= 8 and solid_headroom
            for _ in range(levels):
                mask[index] = available
                index += 1
    return mask
