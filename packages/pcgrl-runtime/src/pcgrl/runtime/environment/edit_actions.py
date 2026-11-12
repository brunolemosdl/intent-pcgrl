from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pcgrl.runtime.metrics.intent import IntentMeasurement, target_value
from pcgrl.runtime.refine.context import (
    RefineContext,
    target_amplitude,
    target_count,
    target_gap_score,
)
from pcgrl.runtime.refine.corrections import (
    LOWER,
    RAISE,
    block_runs,
    carried_structures,
    correction_needed,
    elevated_spare,
    elevated_surplus,
    hill_shape,
    hill_width,
    is_literal_flat,
    peak_floor,
    plan_correction,
    reads_literally,
    relief_columns,
    runs_from,
)
from pcgrl.runtime.refine.ops import (
    Edit,
    find_bump,
    find_summit,
    gap_runs,
    gap_score,
    ground_heights,
    pipe_clusters,
    roughen_columns,
    tile_mask,
)

N_SEGMENTS = 8

# (action name, intent dimension, direction, coarse parameter levels): one
# action per dimension and direction, so the edit phase can move every
# coordinate of the intent vector both ways.
_SPECS: tuple[tuple[str, str, int, int], ...] = (
    ("remove_enemies", "enemy_density", LOWER, 1),
    ("remove_coins", "coin_density", LOWER, 1),
    ("remove_blocks", "interactive_block_density", LOWER, 1),
    ("remove_pipes", "pipe_density", LOWER, 1),
    ("remove_platforms", "platform_usage", LOWER, 1),
    ("fill_gaps", "gap_difficulty", LOWER, 1),
    ("flatten", "verticality", LOWER, 1),
    ("smooth", "path_smoothness", RAISE, 1),
    ("add_enemies", "enemy_density", RAISE, 2),
    ("add_coins", "coin_density", RAISE, 2),
    ("add_blocks", "interactive_block_density", RAISE, 2),
    ("add_pipe", "pipe_density", RAISE, 1),
    ("dig_gap", "gap_difficulty", RAISE, 2),
    ("raise_hill", "verticality", RAISE, 3),
    ("roughen", "path_smoothness", LOWER, 2),
    ("add_platform", "platform_usage", RAISE, 2),
)
_BY_OP = {op: (dim, direction) for op, dim, direction, _ in _SPECS}
_PER_SEGMENT = sum(levels for *_, levels in _SPECS)


@dataclass(frozen=True)
class EditAction:
    """One decoded edit-phase action.

    Attributes:
        op: Action name.
        segment: Segment index in ``[0, N_SEGMENTS)``.
        param: Per-op coarse parameter level.
    """

    op: str
    segment: int
    param: int

    @property
    def dim(self) -> str:
        """Intent dimension the action moves."""
        return _BY_OP[self.op][0]

    @property
    def direction(self) -> int:
        """Whether the action raises or lowers its dimension."""
        return _BY_OP[self.op][1]


def edit_action_count() -> int:
    """Number of edit actions including the terminal STOP."""
    return _PER_SEGMENT * N_SEGMENTS + 1


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
    segment, offset = divmod(index, _PER_SEGMENT)
    for op, _, _, levels in _SPECS:
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


def plan_edit_action(
    tiles: np.ndarray,
    ctx: RefineContext,
    action: EditAction,
    target: np.ndarray,
    state: IntentMeasurement,
    *,
    tolerance: float,
) -> list[Edit]:
    """Plan the tile edits for one decoded action.

    Description:
        An action names a dimension, a direction and a segment; the amount
        comes from what the target still needs (``plan_correction``), capped
        by the action's coarse level. The polish stage runs the same planner
        over the whole level, so both perform byte-identical writes for the
        same need.

    Args:
        tiles: Working tile array.
        ctx: Shared refine context.
        action: Decoded edit action.
        target: Effective intent target of the episode.
        state: Measurement of ``tiles``.
        tolerance: Per-dimension absolute error treated as satisfied.

    Returns:
        Planned edits; empty when nothing in the segment would help.
    """
    return plan_correction(
        tiles,
        ctx,
        target,
        state,
        action.dim,
        action.direction,
        columns=segment_columns(ctx, action.segment),
        level=action.param,
        tolerance=tolerance,
    )


def edit_action_masks(
    tiles: np.ndarray,
    ctx: RefineContext,
    target: np.ndarray,
    state: IntentMeasurement,
    *,
    tolerance: float,
) -> np.ndarray:
    """Availability mask over the edit-action block (STOP always legal).

    Description:
        An action is unmasked when its dimension still needs to move in its
        direction (``correction_needed``) and its segment holds what the
        planner works on (material to remove, free ground to build on), so
        the policy never spends budget on an edit that cannot help. When
        every dimension is satisfied only STOP remains. The segment check is
        a cheap summary — a planner may still come back empty.

    Args:
        tiles: Current working tile array.
        ctx: Shared refine context.
        target: Effective intent target of the episode.
        state: Measurement of ``tiles``.
        tolerance: Per-dimension absolute error treated as satisfied.

    Returns:
        Boolean array of length ``edit_action_count()``.
    """
    mask = np.zeros(edit_action_count(), dtype=bool)
    mask[stop_action_index()] = True
    needed = {
        op: correction_needed(tiles, ctx, target, state, dim, direction, tolerance=tolerance)
        for op, dim, direction, _ in _SPECS
    }
    if not any(needed.values()):
        return mask
    grid_height = tiles.shape[0]
    heights = ground_heights(tiles, ctx)
    config = ctx.config
    clusters = pipe_clusters(tiles, ctx)
    pipe_surplus = sum(len(cluster) for cluster in clusters) - target_count(
        target_value(target, "pipe_density"),
        config.metrics.target_density_ranges.pipe.max,
        ctx.area,
    )
    reach = config.validation.jump_reach
    gaps = gap_runs(heights)
    gap_surplus = gap_score(heights, reach) - target_gap_score(
        target_value(target, "gap_difficulty"), ctx.width
    )
    enemy_grid = tile_mask(tiles, ctx.enemy_ids)
    coin_grid = tiles == ctx.coin_id
    block_grid = tile_mask(tiles, ctx.block_ids)
    solid_grid = tile_mask(tiles, ctx.solid_like)
    pipe_columns: list[bool] = np.any(tile_mask(tiles, ctx.pipe_ids), axis=0).tolist()
    solid_ratio = float(solid_grid.sum() / max(1, ctx.area))
    solid_headroom = solid_ratio < config.validation.max_solid_ratio - 0.02
    # Floating structure: removable tiles above the ground-connected stack.
    rows = np.arange(grid_height)[:, None]
    in_stack = rows >= (grid_height - np.asarray(heights))[None, :]
    removable = tile_mask(tiles, (ctx.solid_like - ctx.pipe_ids) | set(ctx.block_ids))
    floating_grid = removable & ~in_stack
    platform_room = elevated_surplus(ctx, target, state)
    platform_spare = elevated_spare(ctx, target, state, tolerance)
    flat = is_literal_flat(target)
    clear_platforms = reads_literally(target, "platform_usage")
    relief = relief_columns(tiles, ctx)[0] if flat else None
    cap = state.envelope_min + target_amplitude(target_value(target, "verticality"), config)
    # Where the path stands over the height the verticality target allows.
    over_cap = [height is not None and height > cap for height in state.envelope]
    # Runs of mountable blocks, by the segment they start in.
    lifts = block_runs(tiles, ctx, state) if needed["smooth"] or needed["remove_platforms"] else []
    spawn_col, flag_col = ctx.player[1], ctx.flagpole[1]
    # Columns a hill may cover: grounded, pipe-free, away from the endpoints.
    hill_ground = [
        heights[c] > 0
        and c not in ctx.protected_columns
        and not pipe_columns[c]
        and abs(c - spawn_col) > 4
        and abs(c - flag_col) > 4
        for c in range(ctx.width)
    ]
    terrain_all = {
        c for c in range(ctx.width) if c not in ctx.protected_columns and not pipe_columns[c]
    }
    keep_height = peak_floor(ctx, target, state)
    missing_rows = target_amplitude(target_value(target, "verticality"), config) - state.amplitude
    index = 0
    for segment in range(N_SEGMENTS):
        columns = segment_columns(ctx, segment)
        sl = slice(columns.start, columns.stop)
        free = [c for c in columns if heights[c] > 0 and c not in ctx.protected_columns]
        has_floating = bool(floating_grid[:, sl].any())
        local_lifts = [carried for carried, _ in runs_from(lifts, columns)]
        removable_here = False
        if needed["remove_platforms"] and not clear_platforms and platform_room > 0:
            removable_here = any(
                carried < 2 * platform_room
                for carried, _ in carried_structures(tiles, ctx, state, columns, runs=lifts)
            )
        # Whether a ground step would show on the path, per step height.
        rough = [
            bool(
                roughen_columns(
                    heights,
                    terrain_all,
                    columns,
                    amplitude=amplitude,
                    base_height=ctx.base_height,
                    max_height=cap,
                    path=state.envelope,
                )
            )
            if needed["roughen"]
            else False
            for amplitude in (1, 2)
        ]
        local = {
            "remove_enemies": bool(enemy_grid[:, sl].any()),
            "remove_coins": bool(coin_grid[:, sl].any()),
            "remove_blocks": bool(block_grid[:, sl].any()),
            "remove_pipes": any(
                columns.start <= cluster[0][1] < columns.stop and len(cluster) < 2 * pipe_surplus
                for cluster in clusters
            ),
            "remove_platforms": has_floating if clear_platforms else removable_here,
            "fill_gaps": any(
                columns.start <= start < columns.stop
                and min(width, reach + 1) / (reach + 1) < 2 * gap_surplus
                for start, width in gaps
            ),
            "flatten": bool(relief[sl].any()) if relief is not None else any(over_cap[sl]),
            "smooth": (
                find_bump(heights, terrain_all, columns, keep_height=keep_height) is not None
                or any(carried <= platform_spare for carried in local_lifts)
            ),
            "add_enemies": len(free) >= 4,
            "add_coins": len(free) >= 4,
            "add_blocks": len(free) >= 4,
            "add_pipe": len(free) >= 4 and solid_headroom,
            "dig_gap": len(free) >= 8,
            "raise_hill": solid_headroom,
            "roughen": solid_headroom,
            "add_platform": len(free) >= 4,
        }
        for op, _, _, levels in _SPECS:
            for level in range(levels):
                available = needed[op] and local[op]
                if available and op == "raise_hill" and level == 0:
                    # Level zero raises the summit the level already has.
                    summit = find_summit(
                        heights,
                        state.envelope,
                        terrain_all,
                        columns,
                        rows=missing_rows,
                        jump_height=config.validation.jump_height,
                        grid_height=grid_height,
                    )
                    available = summit is not None
                elif available and op == "raise_hill":
                    span = hill_width(ctx, *hill_shape(ctx, target, state, level))
                    available = _hill_fits(columns, span, hill_ground)
                elif available and op == "roughen":
                    available = rough[level]
                mask[index] = available
                index += 1
    return mask


def _hill_fits(columns: range, span: int, ground: list[bool]) -> bool:
    """Whether a hill ``span`` columns wide can start inside the range."""
    run = 0
    for column in range(min(len(ground), columns.stop + span - 1) - 1, columns.start - 1, -1):
        run = run + 1 if ground[column] else 0
        if run >= span and column < columns.stop:
            return True
    return False
