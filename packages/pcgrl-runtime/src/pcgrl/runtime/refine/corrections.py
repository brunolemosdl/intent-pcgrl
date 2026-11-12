from __future__ import annotations

import math

import numpy as np

from pcgrl.runtime.metrics.intent import INTENT_KEYS, IntentMeasurement, target_value
from pcgrl.runtime.refine.context import (
    RefineContext,
    target_amplitude,
    target_count,
    target_elevated_fraction,
    target_gap_score,
    target_mean_delta,
)
from pcgrl.runtime.refine.ops import (
    Edit,
    clear_pipe,
    gap_runs,
    gap_score,
    ground_heights,
    pipe_clusters,
    plan_add_blocks,
    plan_add_pipe,
    plan_add_platform,
    plan_add_tiles_on_surface,
    plan_cap_heights,
    plan_dig_gap,
    plan_fill_gaps,
    plan_flatten,
    plan_raise_blocks,
    plan_raise_hill,
    plan_raise_summit,
    plan_remove_blocks,
    plan_remove_pipes_above,
    plan_remove_platforms,
    plan_remove_tiles,
    plan_roughen,
    plan_smooth,
    plan_trim_pipes,
    tile_mask,
)

RAISE = 1
LOWER = -1

_DIM = {key: index for index, key in enumerate(INTENT_KEYS)}
_EPS = 1e-6
# Largest batch one policy action may add, per coarse parameter level
# (``None``: everything the target still needs that fits the segment).
_ADD_STEPS: dict[str, tuple[int | None, ...]] = {
    "enemy_density": (2, None),
    "coin_density": (3, None),
    "interactive_block_density": (9, 9),
}
# Candidate placements a one-structure correction (a pit, a platform) plans:
# the caller keeps the first one its gates accept.
_ALTERNATIVES = 3
# Corrections that place one structure and plan alternatives for it.
SINGLE_PLACEMENT = frozenset({("gap_difficulty", RAISE), ("platform_usage", RAISE)})
# (height above the surface, widest span) of the platform a policy action adds.
_PLATFORM_LEVELS = ((2, 8), (3, 4))
# Height steps one policy action may add, per coarse parameter level: a few
# gentle ones, or everything the target still needs that fits the segment.
_POLICY_ROUGHEN_BOUNDARIES: tuple[int | None, ...] = (4, None)


def is_literal_flat(target: np.ndarray) -> bool:
    """Whether the request reads as literally flat ground.

    The verticality band maps every amplitude below its 5th percentile to
    zero, so a request at the bottom of the band together with a near-maximal
    smoothness asks for more than the metric can see: no relief at all.
    """
    return (
        target_value(target, "verticality") <= 0.05
        and target_value(target, "path_smoothness") >= 0.9
    )


def _count(tiles: np.ndarray, ids: tuple[int, ...], columns: range | None = None) -> int:
    """Occurrences of ``ids`` in the whole grid or inside a column range."""
    view = tiles if columns is None else tiles[:, columns.start : columns.stop]
    return int(tile_mask(view, ids).sum())


def _count_spec(ctx: RefineContext, dim: str) -> tuple[tuple[int, ...], float]:
    """Tile ids and density cap of a count dimension."""
    ranges = ctx.config.metrics.target_density_ranges
    if dim == "enemy_density":
        return ctx.enemy_ids, ranges.enemy.max
    if dim == "coin_density":
        return (ctx.coin_id,), ranges.coin.max
    return ctx.block_ids, ranges.block.max


def standing_cap(ctx: RefineContext, target: np.ndarray, state: IntentMeasurement) -> int:
    """Highest standing height the verticality target allows.

    Anchored on the envelope minimum plus the target amplitude, so structures
    added for another dimension never push verticality past its own target.
    """
    return state.envelope_min + target_amplitude(target_value(target, "verticality"), ctx.config)


def elevated_surplus(ctx: RefineContext, target: np.ndarray, state: IntentMeasurement) -> float:
    """Elevated path columns beyond what the platform target asks for."""
    wanted = target_elevated_fraction(target_value(target, "platform_usage"), ctx.config)
    return state.elevated_columns - wanted * state.path_columns


def elevated_spare(
    ctx: RefineContext, target: np.ndarray, state: IntentMeasurement, tolerance: float
) -> int:
    """Elevated columns the level can lose with platform usage still on target."""
    band = ctx.config.metrics.platform_usage_band
    slack = tolerance * (band.max - band.min) * state.path_columns
    return max(0, int(elevated_surplus(ctx, target, state) + slack))


def peak_floor(ctx: RefineContext, target: np.ndarray, state: IntentMeasurement) -> int:
    """Stack height smoothing may not cut below.

    The lower of the height the verticality target allows and the height the
    level reaches today: a peak the target stands on stays, one above it may
    come down to it.
    """
    return min(standing_cap(ctx, target, state), state.envelope_min + state.amplitude)


def _flat_relief(tiles: np.ndarray, ctx: RefineContext, columns: range) -> bool:
    """Whether a literal-flat request still sees relief in the column range."""
    heights = ground_heights(tiles, ctx)
    grid_height = tiles.shape[0]
    removable = tile_mask(tiles, (ctx.solid_like - ctx.pipe_ids) | set(ctx.block_ids))
    for column in columns:
        if column in ctx.protected_columns:
            continue
        stack = heights[column]
        if stack not in (0, ctx.base_height):
            return True
        if bool(removable[: grid_height - stack, column].any()):
            return True
    return False


def relief_columns(tiles: np.ndarray, ctx: RefineContext) -> tuple[np.ndarray, np.ndarray]:
    """Columns a literal-flat request still sees as relief.

    Returns:
        ``(relief, counted)`` boolean arrays over the columns: relief marks a
        ground stack that is neither the floor nor a pit, or floating
        structure above it; counted leaves out the protected endpoints and
        the pipe columns, which answer to the pipe target.
    """
    heights = np.asarray(ground_heights(tiles, ctx))
    grid_height = tiles.shape[0]
    rows = np.arange(grid_height)[:, None]
    above_stack = rows < (grid_height - heights)[None, :]
    removable = tile_mask(tiles, (ctx.solid_like - ctx.pipe_ids) | set(ctx.block_ids))
    relief = (heights != 0) & (heights != ctx.base_height)
    relief |= np.any(removable & above_stack, axis=0)
    counted = ~np.any(tile_mask(tiles, ctx.pipe_ids), axis=0)
    counted[sorted(ctx.protected_columns)] = False
    return relief & counted, counted


def relief_fraction(tiles: np.ndarray, ctx: RefineContext) -> float:
    """Share of the level a literal-flat request still sees as relief."""
    relief, counted = relief_columns(tiles, ctx)
    total = int(counted.sum())
    return float(relief.sum() / total) if total else 0.0


def _edge(excess: float, value: float) -> float:
    """Distance for a signed excess over the target, in the value's direction.

    A target at either end of its band is a half-line: everything past the
    band edge measures the same, so only the side inside the band counts.
    """
    if value >= 1.0 - _EPS:
        return max(0.0, -excess)
    if value <= _EPS:
        return max(0.0, excess)
    return abs(excess)


def raw_gap(
    tiles: np.ndarray,
    ctx: RefineContext,
    target: np.ndarray,
    state: IntentMeasurement,
    dim: str,
) -> float:
    """How far a dimension is from its target, in the metric's own units.

    Description:
        Objects for the count dimensions, pipe tiles, raw gap score, envelope
        rows, mean envelope step and elevated columns: progress that the
        banded value hides (below a band, or past its saturation) is visible
        here. For verticality above its target the share of the path
        standing over the allowed height is added, so cutting one of
        several peaks counts as progress.

    Args:
        tiles: Working tile array.
        ctx: Shared refine context.
        target: Effective intent target.
        state: Measurement of ``tiles``.
        dim: Intent dimension name.
    """
    config = ctx.config
    value = target_value(target, dim)
    if dim in _ADD_STEPS:
        ids, band_max = _count_spec(ctx, dim)
        return _edge(_count(tiles, ids) - target_count(value, band_max, ctx.area), value)
    if dim == "pipe_density":
        desired = target_count(value, config.metrics.target_density_ranges.pipe.max, ctx.area)
        return _edge(_count(tiles, tuple(ctx.pipe_ids)) - desired, value)
    if dim == "gap_difficulty":
        score = gap_score(ground_heights(tiles, ctx), config.validation.jump_reach)
        return _edge(score - target_gap_score(value, ctx.width), value)
    if dim == "verticality":
        amplitude = target_amplitude(value, config)
        over = state.amplitude - amplitude
        gap = _edge(over, value)
        if gap > 0 and over > 0:
            cap = state.envelope_min + amplitude
            above = sum(1 for height in state.envelope if height is not None and height > cap)
            gap += above / max(1, state.path_columns)
        return gap
    if dim == "path_smoothness":
        # A smoother path has the smaller mean step, so the sign flips.
        return _edge(target_mean_delta(value, config) - state.mean_step, value)
    wanted = target_elevated_fraction(value, config) * state.path_columns
    return _edge(state.elevated_columns - wanted, value)


def reads_literally(target: np.ndarray, dim: str) -> bool:
    """Whether lowering ``dim`` is judged by literal relief for this request."""
    if not is_literal_flat(target):
        return False
    if dim == "verticality":
        return True
    return dim == "platform_usage" and target_value(target, "platform_usage") <= 0.05


def correction_needed(
    tiles: np.ndarray,
    ctx: RefineContext,
    target: np.ndarray,
    state: IntentMeasurement,
    dim: str,
    direction: int,
    *,
    tolerance: float,
) -> bool:
    """Whether moving ``dim`` in ``direction`` would bring it closer to the target.

    Description:
        Count dimensions compare whole objects, so they are exact; pipes and
        gaps require the smallest possible step to help more than it
        overshoots; verticality compares envelope amplitudes in rows (the band
        quantises it); smoothness and platform usage use the tolerance.

    Args:
        tiles: Working tile array.
        ctx: Shared refine context.
        target: Effective intent target.
        state: Measurement of ``tiles``.
        dim: Intent dimension name.
        direction: ``RAISE`` or ``LOWER``.
        tolerance: Per-dimension absolute error treated as satisfied.
    """
    config = ctx.config
    value = target_value(target, dim)
    measured = float(state.vector[_DIM[dim]])
    # Past the end of a band everything measures the same: a target at the
    # top never needs lowering, one at the bottom never needs raising.
    saturated = value >= 1.0 - _EPS if direction == LOWER else value <= _EPS
    if dim in _ADD_STEPS:
        ids, band_max = _count_spec(ctx, dim)
        surplus = _count(tiles, ids) - target_count(value, band_max, ctx.area)
        if saturated:
            return False
        return surplus >= 1 if direction == LOWER else surplus <= -1
    if dim == "pipe_density":
        desired = target_count(value, config.metrics.target_density_ranges.pipe.max, ctx.area)
        clusters = pipe_clusters(tiles, ctx)
        surplus = sum(len(cluster) for cluster in clusters) - desired
        if saturated:
            return False
        if direction == LOWER:
            return any(len(cluster) < 2 * surplus for cluster in clusters)
        return _pipe_height(target) < -surplus
    if dim == "gap_difficulty":
        heights = ground_heights(tiles, ctx)
        reach = config.validation.jump_reach
        excess = gap_score(heights, reach) - target_gap_score(value, ctx.width)
        cap = reach + 1
        if saturated:
            return False
        if direction == LOWER:
            return any(min(width, cap) / cap < 2 * excess for _, width in gap_runs(heights))
        return 3 / cap < -2 * excess
    if dim == "verticality":
        amplitude = target_amplitude(value, config)
        if direction == LOWER:
            if is_literal_flat(target):
                return _flat_relief(tiles, ctx, range(0, ctx.width))
            return state.amplitude > amplitude and measured > value + _EPS
        if is_literal_flat(target):
            return False
        return state.amplitude < amplitude and measured < value - _EPS
    if direction == LOWER:
        if dim == "platform_usage" and is_literal_flat(target) and value <= 0.05:
            return _flat_relief(tiles, ctx, range(0, ctx.width))
        return measured - value > tolerance + _EPS
    return value - measured > tolerance + _EPS


def hill_shape(
    ctx: RefineContext, target: np.ndarray, state: IntentMeasurement, level: int | None
) -> tuple[int, int]:
    """Peak height and ramp step of the hill that reaches the target amplitude.

    Policy levels one and two pick the ramp (one or two rows per column: a
    gentler ramp keeps the path smoother, a steeper one keeps the hill
    short); the polish stage picks by the smoothness target.
    """
    amplitude = target_amplitude(target_value(target, "verticality"), ctx.config)
    peak = min(state.envelope_min + amplitude, ctx.height - 2)
    if level is None:
        ramp_step = 1 if target_value(target, "path_smoothness") >= 0.5 else 2
    else:
        ramp_step = max(1, level)
    return peak, min(ramp_step, ctx.config.validation.jump_height)


def hill_width(ctx: RefineContext, peak: int, ramp_step: int) -> int:
    """Columns a hill of the given shape occupies."""
    climb = max(1, (peak - ctx.base_height + ramp_step - 1) // ramp_step)
    return 2 * climb + 3


def _pipe_height(target: np.ndarray) -> int:
    """Rows of a newly added pipe.

    Shorter pipes inject fewer envelope steps when the target also asks for a
    smooth path (each pipe top is two unsmoothable edges).
    """
    return 2 if target_value(target, "path_smoothness") >= 0.8 else 3


def plan_correction(
    tiles: np.ndarray,
    ctx: RefineContext,
    target: np.ndarray,
    state: IntentMeasurement,
    dim: str,
    direction: int,
    *,
    columns: range,
    level: int | None = None,
    tolerance: float = 0.05,
) -> list[Edit]:
    """Plan the edits that move one dimension toward its target.

    Description:
        Inverts the metric of ``dim`` into tile writes sized by what the
        target still needs, so a correction never overshoots on purpose:
        removals keep the count the target asks for, additions stop at the
        deficit, pipes and gaps are touched only when the step helps more
        than it overshoots, and terrain is cut or raised to the envelope
        amplitude the verticality target maps to. The policy's edit actions
        call it on one segment with a coarse ``level``; the polish stage
        calls it on the whole level with ``level=None`` (amounts chosen from
        the remaining error).

    Args:
        tiles: Working tile array.
        ctx: Shared refine context.
        target: Effective intent target.
        state: Measurement of ``tiles``.
        dim: Intent dimension name.
        direction: ``RAISE`` or ``LOWER`` the measured value.
        columns: Column range the edits may touch.
        level: Coarse parameter of a policy action; ``None`` sizes by error.
        tolerance: Per-dimension absolute error treated as satisfied.

    Returns:
        Planned edits; empty when nothing in the range would help.
    """
    if dim in _ADD_STEPS:
        return _plan_count(tiles, ctx, target, state, dim, direction, columns, level, tolerance)
    if dim == "pipe_density":
        return _plan_pipes(tiles, ctx, target, state, direction, columns)
    if dim == "gap_difficulty":
        return _plan_gaps(tiles, ctx, target, direction, columns, level)
    if dim == "verticality":
        return _plan_verticality(tiles, ctx, target, state, direction, columns, level)
    if dim == "path_smoothness":
        return _plan_smoothness(tiles, ctx, target, state, direction, columns, level, tolerance)
    if dim == "platform_usage":
        return _plan_platforms(tiles, ctx, target, state, direction, columns, level)
    raise ValueError(f"unknown intent dimension {dim!r}")


def _plan_count(
    tiles: np.ndarray,
    ctx: RefineContext,
    target: np.ndarray,
    state: IntentMeasurement,
    dim: str,
    direction: int,
    columns: range,
    level: int | None,
    tolerance: float,
) -> list[Edit]:
    """Enemies, coins and blocks: remove the surplus or add the deficit."""
    ids, band_max = _count_spec(ctx, dim)
    desired = target_count(target_value(target, dim), band_max, ctx.area)
    surplus = _count(tiles, ids) - desired
    if direction == LOWER:
        if surplus <= 0:
            return []
        keep = max(0, _count(tiles, ids, columns) - surplus)
        if dim == "interactive_block_density":
            return plan_remove_blocks(tiles, ctx, keep=keep, columns=columns)
        op = "remove_enemies" if dim == "enemy_density" else "remove_coins"
        return plan_remove_tiles(tiles, ctx, ids, keep=keep, columns=columns, op=op)
    if surplus >= 0:
        return []
    step = None if level is None else _ADD_STEPS[dim][level]
    amount = -surplus if step is None else min(-surplus, step)
    if dim == "enemy_density":
        return plan_add_tiles_on_surface(
            tiles, ctx, ctx.enemy_ids[0], add=amount, columns=columns, op="add_enemies", spacing=3
        )
    if dim == "coin_density":
        return plan_add_tiles_on_surface(
            tiles,
            ctx,
            ctx.coin_id,
            add=amount,
            columns=columns,
            op="add_coins",
            height_above=2,
            spacing=2,
        )
    if level is None:
        # Mountable blocks become platforms: only when platform usage has room.
        platform = _DIM["platform_usage"]
        mountable = float(target[platform] - state.vector[platform]) > tolerance
    else:
        mountable = level == 0
    return plan_add_blocks(
        tiles,
        ctx,
        add=amount,
        columns=columns,
        mountable=mountable,
        max_standing_height=standing_cap(ctx, target, state),
    )


def _plan_pipes(
    tiles: np.ndarray,
    ctx: RefineContext,
    target: np.ndarray,
    state: IntentMeasurement,
    direction: int,
    columns: range,
) -> list[Edit]:
    """Remove whole pipe clusters, smallest first, or add one grounded pipe.

    Removing part of a cluster would trip the pipe-shape validator, so the
    cluster is the atomic unit.
    """
    value = target_value(target, "pipe_density")
    desired = target_count(value, ctx.config.metrics.target_density_ranges.pipe.max, ctx.area)
    clusters = pipe_clusters(tiles, ctx)
    surplus = sum(len(cluster) for cluster in clusters) - desired
    if direction == LOWER:
        edits: list[Edit] = []
        local = [c for c in clusters if columns.start <= c[0][1] < columns.stop]
        for cluster in sorted(local, key=len):
            if len(cluster) >= 2 * surplus:
                break
            edits.append(Edit(op="remove_pipes", cells=clear_pipe(tiles, ctx, cluster)))
            surplus -= len(cluster)
        return edits
    pipe_height = _pipe_height(target)
    if pipe_height >= -surplus:
        return []
    spacing = 3 if value >= 0.8 else None
    # The pipe top must stay under the height the verticality target allows.
    surface = min(4, standing_cap(ctx, target, state) - pipe_height)
    return plan_add_pipe(
        tiles,
        ctx,
        pipe_height=pipe_height,
        columns=columns,
        max_surface_height=surface,
        spacing=spacing,
    )


def _plan_gaps(
    tiles: np.ndarray,
    ctx: RefineContext,
    target: np.ndarray,
    direction: int,
    columns: range,
    level: int | None,
) -> list[Edit]:
    """Fill the hardest gaps down to the target score, or dig one gap."""
    reach = ctx.config.validation.jump_reach
    cap = reach + 1
    desired = target_gap_score(target_value(target, "gap_difficulty"), ctx.width)
    if direction == LOWER:
        return plan_fill_gaps(tiles, ctx, target_score=desired, columns=columns)
    deficit = desired - gap_score(ground_heights(tiles, ctx), reach)
    # A policy level names the widest pit it digs; narrower ones follow when
    # that would overshoot or does not fit.
    widths = (4, 3, 5) if level is None else tuple(range(3 + level, 2, -1))
    edits: list[Edit] = []
    for width in widths:
        width = min(width, ctx.config.validation.max_gap_width)
        if min(width, cap) / cap >= 2 * deficit:
            continue
        edits += plan_dig_gap(
            tiles, ctx, gap_width=width, columns=columns, alternatives=_ALTERNATIVES
        )
    return edits


def _plan_verticality(
    tiles: np.ndarray,
    ctx: RefineContext,
    target: np.ndarray,
    state: IntentMeasurement,
    direction: int,
    columns: range,
    level: int | None,
) -> list[Edit]:
    """Cut everything standing above the target amplitude, or raise the relief."""
    amplitude = target_amplitude(target_value(target, "verticality"), ctx.config)
    base = state.envelope_min
    if direction == LOWER:
        flat = is_literal_flat(target)
        cap = base + amplitude
        edits = plan_trim_pipes(tiles, ctx, max_height=3, columns=columns)
        if amplitude <= 1 or flat:
            edits += plan_flatten(tiles, ctx, base_height=ctx.base_height, columns=columns)
        else:
            edits += plan_cap_heights(tiles, ctx, cap=cap, columns=columns)
        if flat:
            # Literal caps: the banded amplitude is not visible to the metric
            # but its mounds and floaters are visible to the player. Pipes
            # and platforms survive only when their own targets ask for them.
            pipe_cap = base if target_value(target, "pipe_density") <= 0.05 else base + 2
            platform_cap = base if target_value(target, "platform_usage") <= 0.05 else cap
            edits += plan_remove_pipes_above(tiles, ctx, cap=pipe_cap, columns=columns)
            edits += plan_remove_platforms(tiles, ctx, columns=columns, above_height=platform_cap)
        else:
            edits += plan_remove_pipes_above(tiles, ctx, cap=cap, columns=columns)
            edits += plan_remove_platforms(tiles, ctx, columns=columns, above_height=cap)
        return edits
    if not level:
        # A few tiles on the summit the level already has. Policy level zero
        # is exactly that; the polish stage falls back to a new hill.
        summit = plan_raise_summit(
            tiles, ctx, envelope=state.envelope, rows=amplitude - state.amplitude, columns=columns
        )
        if summit or level == 0:
            return summit
    peak, ramp_step = hill_shape(ctx, target, state, level)
    # A hill starts inside the range but may run past its right edge: a tall
    # gentle hill is wider than one segment.
    reach = range(
        columns.start, min(ctx.width, columns.stop + hill_width(ctx, peak, ramp_step) - 1)
    )
    return plan_raise_hill(tiles, ctx, peak_height=peak, ramp_step=ramp_step, columns=reach)


def _plan_smoothness(
    tiles: np.ndarray,
    ctx: RefineContext,
    target: np.ndarray,
    state: IntentMeasurement,
    direction: int,
    columns: range,
    level: int | None,
    tolerance: float,
) -> list[Edit]:
    """Level bumps and dips, or modulate the ground into a sawtooth."""
    smooth = _DIM["path_smoothness"]
    needed = target_mean_delta(target_value(target, "path_smoothness"), ctx.config)
    present = target_mean_delta(float(state.vector[smooth]), ctx.config)
    if direction == RAISE:
        rows = max(1, math.ceil((present - needed) * max(1, state.path_columns - 1)))
        edits = plan_smooth(
            tiles, ctx, columns=columns, rows=rows, keep_height=peak_floor(ctx, target, state)
        )
        # Mountable block tops are envelope steps no ground op can smooth;
        # lifting them out of jump reach removes the spikes without touching
        # the block count, as far as platform usage can spare the columns.
        # Narrowest runs first: a lone block costs two steps for one column.
        spare = elevated_spare(ctx, target, state, tolerance)
        for carried, lifts in sorted(block_runs(tiles, ctx, state, columns), key=lambda r: r[0]):
            if carried > spare:
                break
            edits += lifts
            spare -= carried
        return edits
    if level is None:
        amplitude = 1 if target_value(target, "verticality") <= 0.3 else 2
    else:
        amplitude = 1 + level
    if needed <= present:
        return []
    boundaries = max(1, math.ceil((needed - present) * ctx.width / amplitude))
    cap = None if level is None else _POLICY_ROUGHEN_BOUNDARIES[level]
    if cap is not None:
        boundaries = min(boundaries, cap)
    return plan_roughen(
        tiles,
        ctx,
        amplitude=amplitude,
        boundaries=boundaries,
        columns=columns,
        max_height=standing_cap(ctx, target, state),
        path=state.envelope,
    )


def block_runs(
    tiles: np.ndarray,
    ctx: RefineContext,
    state: IntentMeasurement,
    columns: range | None = None,
) -> list[tuple[int, list[Edit]]]:
    """Runs of mountable blocks, with the edits that lift them out of reach.

    Description:
        A run is lifted as a whole, wherever it ends: the lifted part of a
        walkway stays reachable from the rest of it, so half a run frees no
        column. Each entry pairs the elevated path columns the run carries
        with its per-column lifts. With ``columns`` only the runs that start
        inside the range are returned.
    """
    grid_height = tiles.shape[0]
    lifts = plan_raise_blocks(tiles, ctx, min_height_above=6, columns=range(0, ctx.width))
    runs: list[list[Edit]] = []
    for lift in lifts:
        column = lift.cells[0][1]
        if runs and runs[-1][-1].cells[0][1] == column - 1:
            runs[-1].append(lift)
        else:
            runs.append([lift])
    found: list[tuple[int, list[Edit]]] = []
    for run in runs:
        carried = sum(
            1
            for lift in run
            if any(
                state.envelope[col] == grid_height - row
                for row, col, tile in lift.cells
                if tile == ctx.empty_id
            )
        )
        if carried:
            found.append((carried, run))
    return found if columns is None else runs_from(found, columns)


def runs_from(runs: list[tuple[int, list[Edit]]], columns: range) -> list[tuple[int, list[Edit]]]:
    """The block runs that start inside ``columns``."""
    return [
        (carried, run)
        for carried, run in runs
        if columns.start <= run[0].cells[0][1] < columns.stop
    ]


def carried_structures(
    tiles: np.ndarray,
    ctx: RefineContext,
    state: IntentMeasurement,
    columns: range,
    *,
    runs: list[tuple[int, list[Edit]]] | None = None,
) -> list[tuple[int, list[Edit]]]:
    """Floating structures the path stands on, with the edits that take them away.

    Plain platforms are removed (the stretch inside ``columns``); mountable
    blocks are lifted out of reach, which keeps their count
    (``block_runs``; ``runs`` passes the level's runs when the caller
    already has them). Structures the path does not stand on are left out:
    removing them frees no elevated column.
    """
    grid_height = tiles.shape[0]
    found: list[tuple[int, list[Edit]]] = []
    for edit in plan_remove_platforms(tiles, ctx, columns=columns, keep_blocks=True):
        carried = sum(1 for row, col, _ in edit.cells if state.envelope[col] == grid_height - row)
        if carried:
            found.append((carried, [edit]))
    if runs is None:
        runs = block_runs(tiles, ctx, state)
    return found + runs_from(runs, columns)


def _best_fit(candidates: list[tuple[int, list[Edit]]], room: float) -> list[Edit]:
    """Order the structures so that their carried columns add up to the surplus.

    Widest first while they fit, then the remaining ones that help more than
    they overshoot.
    """
    chosen: list[Edit] = []
    rest: list[tuple[int, list[Edit]]] = []
    for carried, edits in sorted(candidates, key=lambda item: -item[0]):
        if carried <= room:
            chosen += edits
            room -= carried
        else:
            rest.append((carried, edits))
    # Narrowest first, each one only if it helps more than it overshoots:
    # the first closes what is left, the others stand in when a gate refuses
    # an earlier pick (the caller stops once the dimension is served).
    for _, edits in sorted((item for item in rest if item[0] < 2 * room), key=lambda i: i[0]):
        chosen += edits
    return chosen


def _plan_platforms(
    tiles: np.ndarray,
    ctx: RefineContext,
    target: np.ndarray,
    state: IntentMeasurement,
    direction: int,
    columns: range,
    level: int | None,
) -> list[Edit]:
    """Remove floating structure, or add one mountable platform."""
    if direction == LOWER:
        lifts = plan_raise_blocks(tiles, ctx, min_height_above=6, columns=columns)
        if is_literal_flat(target) and target_value(target, "platform_usage") <= 0.05:
            return plan_remove_platforms(tiles, ctx, columns=columns) + lifts
        return _best_fit(
            carried_structures(tiles, ctx, state, columns), elevated_surplus(ctx, target, state)
        )
    platform = _DIM["platform_usage"]
    delta = float(target[platform] - state.vector[platform])
    smooth_target = target_value(target, "path_smoothness")
    cap = standing_cap(ctx, target, state)
    band = ctx.config.metrics.platform_usage_band
    # Columns of platform the remaining error is worth: a platform moves the
    # elevated fraction by its width over the path length.
    needed = max(3, round(delta * (band.max - band.min) * max(1, state.path_columns)))
    if level is None:
        height_above = 2 if smooth_target >= 0.8 else 3
        # A wide platform under a smooth-path target amortises its two edge
        # steps; near the target a narrow one avoids overshooting.
        if smooth_target >= 0.8 and delta > 0.15:
            preferred = 8
        elif delta > 0.15:
            preferred = 6
        else:
            preferred = 3
        widths = tuple(dict.fromkeys((preferred, 4, 3)))
    else:
        height_above, width = _PLATFORM_LEVELS[level]
        # The widest platform the level allows and the target needs; a
        # narrower one follows when the segment has no room for it.
        widths = tuple(range(min(width, needed), 2, -1))
    edits: list[Edit] = []
    for width in widths:
        edits += plan_add_platform(
            tiles,
            ctx,
            width=width,
            height_above=height_above,
            columns=columns,
            max_standing_height=cap,
            alternatives=_ALTERNATIVES,
        )
    return edits
