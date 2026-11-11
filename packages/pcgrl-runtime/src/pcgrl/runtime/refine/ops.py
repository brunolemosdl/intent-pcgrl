from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pcgrl.runtime.refine.context import RefineContext


@dataclass(frozen=True)
class Edit:
    """One atomic, individually gated batch of tile writes.

    Attributes:
        op: Planner name that produced the edit.
        cells: ``(row, col, new_tile_id)`` writes applied together.
    """

    op: str
    cells: tuple[tuple[int, int, int], ...]


def apply_edit(tiles: np.ndarray, edit: Edit) -> tuple[tuple[int, int, int], ...]:
    """Write an edit in place and return the snapshot needed to revert it.

    Args:
        tiles: Working tile array (mutated).
        edit: Edit to apply.

    Returns:
        ``(row, col, previous_tile_id)`` tuples in write order.
    """
    snapshot = tuple((row, col, int(tiles[row, col])) for row, col, _ in edit.cells)
    for row, col, new_id in edit.cells:
        tiles[row, col] = new_id
    return snapshot


def revert_edit(tiles: np.ndarray, snapshot: tuple[tuple[int, int, int], ...]) -> None:
    """Undo an applied edit from its snapshot.

    Args:
        tiles: Working tile array (mutated).
        snapshot: Return value of ``apply_edit``.
    """
    for row, col, old_id in snapshot:
        tiles[row, col] = old_id


def ground_heights(tiles: np.ndarray, ctx: RefineContext) -> list[int]:
    """Bottom-contiguous solid-stack height per column (0 = floor gap).

    Mirrors ``metrics.intent.ground_height_profile`` on a raw array.
    """
    heights: list[int] = []
    height = tiles.shape[0]
    for column in range(tiles.shape[1]):
        stack = 0
        for row in range(height - 1, -1, -1):
            if int(tiles[row, column]) in ctx.solid_like:
                stack += 1
            else:
                break
        heights.append(stack)
    return heights


def _cells_of(tiles: np.ndarray, ids: tuple[int, ...], columns: range) -> list[tuple[int, int]]:
    """All ``(row, col)`` holding any of ``ids`` inside the column range, sorted."""
    found: list[tuple[int, int]] = []
    rows, cols = np.nonzero(np.isin(tiles, list(ids)))
    for row, col in zip(rows.tolist(), cols.tolist(), strict=False):
        if columns.start <= col < columns.stop:
            found.append((row, col))
    found.sort(key=lambda cell: (cell[1], cell[0]))
    return found


def _keep_indices(total: int, keep: int) -> set[int]:
    """Evenly spaced indices to keep out of ``total`` items."""
    if keep <= 0:
        return set()
    if keep >= total:
        return set(range(total))
    positions = np.linspace(0, total - 1, num=keep)
    return {round(p) for p in positions}


def _column_has_pipe(tiles: np.ndarray, ctx: RefineContext, column: int) -> bool:
    """Whether any pipe tile occupies the column."""
    return bool(np.isin(tiles[:, column], list(ctx.pipe_ids)).any())


def terrain_columns(tiles: np.ndarray, ctx: RefineContext, columns: range) -> list[int]:
    """Columns terrain ops may rewrite: unprotected and pipe-free."""
    return [
        column
        for column in columns
        if column not in ctx.protected_columns and not _column_has_pipe(tiles, ctx, column)
    ]


def rewrite_stack(
    tiles: np.ndarray, ctx: RefineContext, column: int, new_height: int
) -> list[tuple[int, int, int]]:
    """Cell writes that move one column's ground stack to ``new_height``.

    Description:
        Lowering empties the removed stack rows and drops a decoration
        (enemy or coin) sitting on the old surface onto the new one; raising
        writes solid over whatever occupied the added rows. Cells outside the
        changed band are untouched, so platforms above the new surface
        survive.
    """
    height = tiles.shape[0]
    current = 0
    for row in range(height - 1, -1, -1):
        if int(tiles[row, column]) in ctx.solid_like:
            current += 1
        else:
            break
    cells: list[tuple[int, int, int]] = []
    if new_height < current:
        for row in range(height - current, height - new_height):
            cells.append((row, column, ctx.empty_id))
        old_surface = height - current - 1
        new_surface = height - new_height - 1
        if old_surface >= 0:
            occupant = int(tiles[old_surface, column])
            decoration_ids = (*ctx.enemy_ids, ctx.coin_id)
            if occupant in decoration_ids and (new_surface, column) not in ctx.protected:
                cells.append((old_surface, column, ctx.empty_id))
                cells.append((new_surface, column, occupant))
    elif new_height > current:
        for row in range(height - new_height, height - current):
            cells.append((row, column, ctx.solid_id))
    return cells


def plan_remove_tiles(
    tiles: np.ndarray,
    ctx: RefineContext,
    ids: tuple[int, ...],
    *,
    keep: int,
    columns: range,
    op: str,
) -> list[Edit]:
    """Remove evenly thinned occurrences of ``ids`` down to ``keep``.

    Args:
        tiles: Working tile array.
        ctx: Shared refine context.
        ids: Tile ids to thin.
        keep: Number of occurrences to leave in place.
        columns: Column range searched.
        op: Edit label.

    Returns:
        At most one edit batching every removal.
    """
    cells = _cells_of(tiles, ids, columns)
    if len(cells) <= keep:
        return []
    kept = _keep_indices(len(cells), keep)
    writes = tuple(
        (row, col, ctx.empty_id) for index, (row, col) in enumerate(cells) if index not in kept
    )
    return [Edit(op=op, cells=writes)] if writes else []


def plan_remove_blocks(
    tiles: np.ndarray, ctx: RefineContext, *, keep: int, columns: range
) -> list[Edit]:
    """Thin interactive blocks; stack-embedded blocks become plain solid.

    Description:
        A block inside the bottom-contiguous stack is load-bearing for the
        ground profile, so it is swapped to ``solid`` (profile and envelope
        unchanged); floating blocks are simply emptied.
    """
    cells = _cells_of(tiles, ctx.block_ids, columns)
    if len(cells) <= keep:
        return []
    heights = ground_heights(tiles, ctx)
    height = tiles.shape[0]
    kept = _keep_indices(len(cells), keep)
    writes: list[tuple[int, int, int]] = []
    for index, (row, col) in enumerate(cells):
        if index in kept:
            continue
        in_stack = row >= height - heights[col]
        writes.append((row, col, ctx.solid_id if in_stack else ctx.empty_id))
    return [Edit(op="remove_blocks", cells=tuple(writes))] if writes else []


def pipe_clusters(
    tiles: np.ndarray, ctx: RefineContext, columns: range | None = None
) -> list[list[tuple[int, int]]]:
    """Orthogonally connected pipe-tile clusters, sorted by leftmost column.

    One vectorised scan finds the few pipe cells; the BFS walks only that
    set (this helper runs inside the per-step edit-action mask).
    """
    pipe_cells = {
        (int(row), int(col)) for row, col in np.argwhere(np.isin(tiles, list(ctx.pipe_ids)))
    }
    seen: set[tuple[int, int]] = set()
    clusters: list[list[tuple[int, int]]] = []
    for row, col in sorted(pipe_cells):
        if (row, col) in seen:
            continue
        stack = [(row, col)]
        cluster: list[tuple[int, int]] = []
        seen.add((row, col))
        while stack:
            r, c = stack.pop()
            cluster.append((r, c))
            for neighbour in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if neighbour in pipe_cells and neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        cluster.sort(key=lambda cell: (cell[1], cell[0]))
        clusters.append(cluster)
    clusters.sort(key=lambda cluster: cluster[0][1])
    if columns is not None:
        clusters = [
            cluster for cluster in clusters if columns.start <= cluster[0][1] < columns.stop
        ]
    return clusters


def plan_repair_pipes(tiles: np.ndarray, ctx: RefineContext) -> list[Edit]:
    """Remove malformed pipe clusters (wrong width, too short, ungrounded).

    Chunk seams can merge or clip corpus pipes into shapes the pipe
    validator hard-rejects; composed levels arrive at the refiner with these
    pre-existing violations, so repair runs before any target-driven edit.
    """
    validation = ctx.config.validation
    height = tiles.shape[0]
    edits: list[Edit] = []
    for cluster in pipe_clusters(tiles, ctx):
        cols = sorted({col for _, col in cluster})
        rows = sorted({row for row, _ in cluster})
        malformed = (validation.require_pipe_width_two and len(cols) != 2) or len(
            rows
        ) < validation.min_pipe_height
        if not malformed and validation.require_grounded_pipes:
            bottom = max(row for row, _ in cluster)
            support = all(
                bottom + 1 < height and int(tiles[bottom + 1, col]) in ctx.solid_like
                for col in cols
            )
            malformed = not support
        if malformed:
            edits.append(
                Edit(
                    op="repair_pipes",
                    cells=tuple((row, col, ctx.empty_id) for row, col in cluster),
                )
            )
    return edits


def plan_remove_pipes(
    tiles: np.ndarray, ctx: RefineContext, *, keep_tiles: int, columns: range
) -> list[Edit]:
    """Remove whole pipe clusters until at most ``keep_tiles`` pipe tiles remain.

    Removing partial clusters would trip the pipe-shape validator, so the
    cluster is the atomic unit; smallest clusters are dropped first to land
    as close to the budget as possible.
    """
    clusters = pipe_clusters(tiles, ctx, columns)
    total = sum(len(cluster) for cluster in clusters)
    if total <= keep_tiles:
        return []
    edits: list[Edit] = []
    for cluster in sorted(clusters, key=len):
        if total <= keep_tiles:
            break
        edits.append(
            Edit(
                op="remove_pipes",
                cells=tuple((row, col, ctx.empty_id) for row, col in cluster),
            )
        )
        total -= len(cluster)
    return edits


def gap_runs(heights: list[int]) -> list[tuple[int, int]]:
    """Maximal runs of floor-gap columns as ``(start, width)`` tuples."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for column, height in enumerate(heights):
        if height == 0:
            if start is None:
                start = column
        elif start is not None:
            runs.append((start, column - start))
            start = None
    if start is not None:
        runs.append((start, len(heights) - start))
    return runs


def gap_score(heights: list[int], jump_reach: int) -> float:
    """Raw (un-normalised) gap-difficulty score of a ground profile."""
    cap = jump_reach + 1
    return sum(min(width, cap) / cap for _, width in gap_runs(heights))


def plan_fill_gaps(
    tiles: np.ndarray, ctx: RefineContext, *, target_score: float, columns: range
) -> list[Edit]:
    """Fill floor gaps, hardest first, until the raw score drops to target.

    Each filled column rises to the lower of its two grounded shoulders
    (minimum 2), keeping the seam walkable from both sides.
    """
    heights = ground_heights(tiles, ctx)
    reach = ctx.config.validation.jump_reach
    edits: list[Edit] = []
    score = gap_score(heights, reach)
    cap = reach + 1
    runs = [run for run in gap_runs(heights) if columns.start <= run[0] < columns.stop]
    for start, width in sorted(runs, key=lambda run: -min(run[1], cap)):
        if score <= target_score:
            break
        left = heights[start - 1] if start > 0 else 0
        right = heights[start + width] if start + width < len(heights) else 0
        fill = max(2, min(h for h in (left, right) if h > 0) if (left or right) else 2)
        cells: list[tuple[int, int, int]] = []
        for column in range(start, start + width):
            cells.extend(rewrite_stack(tiles, ctx, column, fill))
            heights[column] = fill
        if cells:
            edits.append(Edit(op="fill_gaps", cells=tuple(cells)))
            score -= min(width, cap) / cap
    return edits


def plan_flatten(
    tiles: np.ndarray, ctx: RefineContext, *, base_height: int, columns: range
) -> list[Edit]:
    """Cut every eligible ground stack to one uniform height.

    Gap columns stay gaps (the gap dims own them); pipe and protected columns
    are skipped, so flattening can never unground a pipe or touch endpoints.
    """
    heights = ground_heights(tiles, ctx)
    cells: list[tuple[int, int, int]] = []
    for column in terrain_columns(tiles, ctx, columns):
        if heights[column] == 0 or heights[column] == base_height:
            continue
        cells.extend(rewrite_stack(tiles, ctx, column, base_height))
    return [Edit(op="flatten", cells=tuple(cells))] if cells else []


def plan_cap_heights(
    tiles: np.ndarray, ctx: RefineContext, *, cap: int, columns: range
) -> list[Edit]:
    """Cut only the ground stacks taller than ``cap`` down to it.

    Unlike ``plan_flatten`` this never raises low columns, so intermediate
    verticality targets keep their existing relief below the cap.
    """
    heights = ground_heights(tiles, ctx)
    cells: list[tuple[int, int, int]] = []
    for column in terrain_columns(tiles, ctx, columns):
        if heights[column] > cap:
            cells.extend(rewrite_stack(tiles, ctx, column, cap))
    return [Edit(op="cap_heights", cells=tuple(cells))] if cells else []


def plan_trim_pipes(
    tiles: np.ndarray, ctx: RefineContext, *, max_height: int, columns: range
) -> list[Edit]:
    """Shorten pipe clusters taller than ``max_height``, keeping valid tops.

    The trimmed cluster keeps its bottom rows and gets a fresh top pair, so
    the width-2 / grounded invariants hold by construction.
    """
    edits: list[Edit] = []
    for cluster in pipe_clusters(tiles, ctx, columns):
        rows = sorted({row for row, _ in cluster})
        cols = sorted({col for _, col in cluster})
        if len(cols) != 2 or len(rows) <= max_height:
            continue
        cells: list[tuple[int, int, int]] = []
        keep_rows = rows[-max_height:]
        for row, col in cluster:
            if row not in keep_rows:
                cells.append((row, col, ctx.empty_id))
        top_row = keep_rows[0]
        cells.append((top_row, cols[0], ctx.pipe_top_left))
        cells.append((top_row, cols[1], ctx.pipe_top_right))
        edits.append(Edit(op="trim_pipes", cells=tuple(cells)))
    return edits


def plan_smooth(
    tiles: np.ndarray, ctx: RefineContext, *, max_step: int, columns: range
) -> list[Edit]:
    """Cut large height steps, then level narrow bumps and dips.

    First pass clamps steps to ``max_step`` toward the lower neighbour;
    second pass removes one- and two-column bumps/dips whose both sides sit
    at the same height — the high-frequency relief that dominates the mean
    envelope delta on corpus chunks.
    """
    heights = ground_heights(tiles, ctx)
    eligible = set(terrain_columns(tiles, ctx, columns))
    cells: list[tuple[int, int, int]] = []
    changed = dict(enumerate(heights))
    for column in range(1, len(heights)):
        left, right = changed[column - 1], changed[column]
        if left == 0 or right == 0:
            continue
        if right - left > max_step and column in eligible:
            new_height = left + max_step
            cells.extend(rewrite_stack(tiles, ctx, column, new_height))
            changed[column] = new_height
        elif left - right > max_step and (column - 1) in eligible:
            new_height = right + max_step
            cells.extend(rewrite_stack(tiles, ctx, column - 1, new_height))
            changed[column - 1] = new_height
    for width in (1, 2):
        for column in range(1, len(heights) - width):
            left = changed[column - 1]
            inner = [changed[column + offset] for offset in range(width)]
            right = changed[column + width]
            if left == 0 or right == 0 or left != right:
                continue
            if all(h != left and h > 0 for h in inner) and all(
                column + offset in eligible for offset in range(width)
            ):
                for offset in range(width):
                    cells.extend(rewrite_stack(tiles, ctx, column + offset, left))
                    changed[column + offset] = left
    return [Edit(op="smooth", cells=tuple(cells))] if cells else []


def plan_remove_platforms(
    tiles: np.ndarray,
    ctx: RefineContext,
    *,
    columns: range,
    above_height: int = 0,
) -> list[Edit]:
    """Delete floating solid-like structures sitting above the ground stack.

    Bricks and question blocks count too — corpus chunks build towers and
    walkways out of them; the block-density loop restores the block count at
    envelope-safe heights on the next phase.

    Args:
        tiles: Working tile array.
        ctx: Shared refine context.
        columns: Column range searched.
        above_height: Only remove platforms whose standing height exceeds
            this (0 removes every floating platform); lets the verticality
            handler clear amplitude spikes while sparing low walkways.
    """
    heights = ground_heights(tiles, ctx)
    height = tiles.shape[0]
    removable = (ctx.solid_like - ctx.pipe_ids) | set(ctx.block_ids)
    edits: list[Edit] = []
    for row in range(height):
        if height - row <= above_height:
            continue
        run: list[tuple[int, int, int]] = []
        for col in columns:
            keep = (
                (row, col) in ctx.protected
                or int(tiles[row, col]) not in removable
                or row >= height - heights[col]
            )
            if keep:
                if run:
                    edits.append(Edit(op="remove_platforms", cells=tuple(run)))
                    run = []
            else:
                run.append((row, col, ctx.empty_id))
        if run:
            edits.append(Edit(op="remove_platforms", cells=tuple(run)))
    edits.sort(key=lambda edit: -len(edit.cells))
    return edits


def plan_raise_blocks(
    tiles: np.ndarray, ctx: RefineContext, *, min_height_above: int, columns: range
) -> list[Edit]:
    """Lift mountable floating blocks out of the envelope's jump reach.

    Blocks within jump height of the local surface are standable platforms
    and inflate platform usage; when the platform target is low they move up
    to ``min_height_above`` rows over the surface, keeping the block count
    (their own metric) intact.
    """
    heights = ground_heights(tiles, ctx)
    height = tiles.shape[0]
    jump = ctx.config.validation.jump_height
    per_column: dict[int, list[tuple[int, int, int]]] = {}
    for row in range(height):
        for col in columns:
            tile = int(tiles[row, col])
            if tile not in ctx.block_ids or (row, col) in ctx.protected:
                continue
            if row >= height - heights[col]:
                continue
            surface_row = height - heights[col] - 1
            above = surface_row - row
            if above > jump:
                continue
            target_row = surface_row - min_height_above
            writes = per_column.setdefault(col, [])
            if (
                target_row < 1
                or int(tiles[target_row, col]) != ctx.empty_id
                or (target_row, col) in ctx.protected
            ):
                writes.append((row, col, ctx.empty_id))
            else:
                writes.append((row, col, ctx.empty_id))
                writes.append((target_row, col, tile))
    # One edit per column: lifting a whole walkway at once can sever the
    # flood path, so each column is gated (and revertible) independently;
    # blocks with no head-room are dropped and re-added by the block loop.
    return [
        Edit(op="raise_blocks", cells=tuple(writes)) for _, writes in sorted(per_column.items())
    ]


def surface_cells(tiles: np.ndarray, ctx: RefineContext, columns: range) -> list[tuple[int, int]]:
    """Empty cells directly on top of the ground stack, spawn-safe.

    Excludes protected columns, pipe columns, and columns within three tiles
    of the player spawn.
    """
    heights = ground_heights(tiles, ctx)
    height = tiles.shape[0]
    spawn_col = ctx.player[1]
    found: list[tuple[int, int]] = []
    for column in terrain_columns(tiles, ctx, columns):
        if heights[column] == 0 or abs(column - spawn_col) <= 3:
            continue
        row = height - heights[column] - 1
        if row < 0 or (row, column) in ctx.protected:
            continue
        if int(tiles[row, column]) == ctx.empty_id:
            found.append((row, column))
    return found


def plan_add_tiles_on_surface(
    tiles: np.ndarray,
    ctx: RefineContext,
    tile_id: int,
    *,
    add: int,
    columns: range,
    op: str,
    height_above: int = 0,
    spacing: int = 2,
) -> list[Edit]:
    """Place ``add`` tiles evenly over surface positions.

    Args:
        tiles: Working tile array.
        ctx: Shared refine context.
        tile_id: Tile to write.
        add: Number of tiles to place.
        columns: Column range searched.
        op: Edit label.
        height_above: Rows above the surface cell (0 = on the surface).
        spacing: Minimum column spacing between placements.

    Returns:
        At most one edit with every placement.
    """
    if add <= 0:
        return []
    candidates = surface_cells(tiles, ctx, columns)
    placed: list[tuple[int, int, int]] = []
    last_col = -(10**9)
    chosen = _keep_indices(len(candidates), min(add * spacing, len(candidates)))
    for index in sorted(chosen):
        if len(placed) >= add:
            break
        row, col = candidates[index]
        target_row = row - height_above
        if target_row < 0 or col - last_col < spacing:
            continue
        if (target_row, col) in ctx.protected:
            continue
        if int(tiles[target_row, col]) != ctx.empty_id:
            continue
        placed.append((target_row, col, tile_id))
        last_col = col
    return [Edit(op=op, cells=tuple(placed))] if placed else []


def plan_add_blocks(
    tiles: np.ndarray,
    ctx: RefineContext,
    *,
    add: int,
    columns: range,
    mountable: bool,
    max_standing_height: int | None = None,
) -> list[Edit]:
    """Place interactive blocks at jump height or out of envelope reach.

    Mountable rows (3 above the surface) become standable platforms and raise
    platform usage; non-mountable rows (6 above) are visible content that the
    envelope cannot reach, for when the platform target is low.
    """
    if add <= 0:
        return []
    height_above = 3 if mountable else 6
    grid_height = tiles.shape[0]
    pattern = (
        ctx.block_ids[0],
        ctx.block_ids[1 % len(ctx.block_ids)],
        ctx.block_ids[0],
    )
    candidates = surface_cells(tiles, ctx, columns)
    placed: list[tuple[int, int, int]] = []
    index = 0
    while len(placed) < add and index < len(candidates):
        row, col = candidates[index]
        target_row = row - height_above
        if target_row < 1:
            index += 1
            continue
        if (
            mountable
            and max_standing_height is not None
            and grid_height - target_row > max_standing_height
        ):
            index += 1
            continue
        run = min(3, add - len(placed))
        cells_ok = all(
            col + offset < ctx.width
            and (target_row, col + offset) not in ctx.protected
            and int(tiles[target_row, col + offset]) == ctx.empty_id
            for offset in range(run)
        )
        if cells_ok:
            for offset in range(run):
                placed.append((target_row, col + offset, pattern[offset % 3]))
            index += 6
        else:
            index += 1
    return [Edit(op="add_blocks", cells=tuple(placed))] if placed else []


def plan_remove_pipes_above(
    tiles: np.ndarray, ctx: RefineContext, *, cap: int, columns: range
) -> list[Edit]:
    """Remove pipe clusters whose top stands above the verticality cap.

    The ground cap skips pipe pedestals (cutting them would unground the
    pipe), so towering pipes are cleared whole; the pipe-density loop re-adds
    them on low terrain in a later phase.
    """
    height = tiles.shape[0]
    edits: list[Edit] = []
    for cluster in pipe_clusters(tiles, ctx, columns):
        top_row = min(row for row, _ in cluster)
        if height - top_row > cap:
            edits.append(
                Edit(
                    op="remove_pipes_above",
                    cells=tuple((row, col, ctx.empty_id) for row, col in cluster),
                )
            )
    return edits


def plan_add_pipe(
    tiles: np.ndarray,
    ctx: RefineContext,
    *,
    pipe_height: int,
    columns: range,
    max_surface_height: int = 4,
    spacing: int | None = None,
) -> list[Edit]:
    """Add one grounded width-2 pipe respecting the corpus spacing rule.

    Args:
        tiles: Working tile array.
        ctx: Shared refine context.
        pipe_height: Pipe height in rows (top included).
        columns: Column range searched.
        max_surface_height: Skip pads higher than this so pipes never sit on
            hills and blow the verticality amplitude.
        spacing: Minimum columns between cluster starts; defaults to the
            corpus-derived ``validation.pipe_min_gap``, relaxed by callers
            chasing extreme pipe-density targets.
    """
    heights = ground_heights(tiles, ctx)
    min_gap = spacing if spacing is not None else ctx.config.validation.pipe_min_gap
    existing = [cluster[0][1] for cluster in pipe_clusters(tiles, ctx)]
    height = tiles.shape[0]
    for col in terrain_columns(tiles, ctx, columns):
        if col + 1 >= ctx.width or (col + 1) in ctx.protected_columns:
            continue
        if _column_has_pipe(tiles, ctx, col + 1):
            continue
        left_h, right_h = heights[col], heights[col + 1]
        if min(left_h, right_h) < 2 or abs(left_h - right_h) > 2:
            continue
        pad = min(left_h, right_h)
        if pad > max_surface_height:
            continue
        if any(abs(col - other) < min_gap for other in existing):
            continue
        surface = height - pad - 1
        top_row = surface - pipe_height + 1
        if top_row < 1:
            continue
        cells: list[tuple[int, int, int]] = []
        for offset, current in ((0, left_h), (1, right_h)):
            if current != pad:
                cells.extend(rewrite_stack(tiles, ctx, col + offset, pad))
        span_blocked = any(
            (row, col + offset) in ctx.protected
            for row in range(top_row, surface + 1)
            for offset in (0, 1)
        )
        if span_blocked:
            continue
        cells.append((top_row, col, ctx.pipe_top_left))
        cells.append((top_row, col + 1, ctx.pipe_top_right))
        for row in range(top_row + 1, surface + 1):
            cells.append((row, col, ctx.pipe_body_left))
            cells.append((row, col + 1, ctx.pipe_body_right))
        return [Edit(op="add_pipe", cells=tuple(cells))]
    return []


def plan_dig_gap(
    tiles: np.ndarray, ctx: RefineContext, *, gap_width: int, columns: range
) -> list[Edit]:
    """Carve one flood-safe floor gap of ``gap_width`` columns.

    The span must be grounded on both shoulders, pipe-free, away from the
    endpoints and existing gaps, and within the configured maximum width, so
    the jump physics can always cross it.
    """
    heights = ground_heights(tiles, ctx)
    height = tiles.shape[0]
    gap_width = min(gap_width, ctx.config.validation.max_gap_width)
    eligible = set(terrain_columns(tiles, ctx, columns))
    spawn_col, flag_col = ctx.player[1], ctx.flagpole[1]
    for start in range(columns.start, columns.stop - gap_width):
        span = range(start, start + gap_width)
        if any(col not in eligible for col in span):
            continue
        if any(abs(col - spawn_col) <= 6 or abs(col - flag_col) <= 6 for col in span):
            continue
        left_ok = start >= 2 and heights[start - 1] > 0 and heights[start - 2] > 0
        right_ok = (
            start + gap_width + 1 < len(heights)
            and heights[start + gap_width] > 0
            and heights[start + gap_width + 1] > 0
        )
        if not (left_ok and right_ok):
            continue
        if any(heights[col] == 0 for col in span):
            continue
        cells: list[tuple[int, int, int]] = []
        for col in span:
            for row in range(height):
                if int(tiles[row, col]) != ctx.empty_id:
                    cells.append((row, col, ctx.empty_id))
        return [Edit(op="dig_gap", cells=tuple(cells))]
    return []


def plan_raise_hill(
    tiles: np.ndarray,
    ctx: RefineContext,
    *,
    peak_height: int,
    ramp_step: int,
    columns: range,
) -> list[Edit]:
    """Build one ramped ground hill reaching ``peak_height``.

    The ramp climbs by ``ramp_step`` per column (within jump height), holds a
    plateau of three columns, and descends symmetrically, so the flood walks
    over it in both directions.
    """
    heights = ground_heights(tiles, ctx)
    base = 2
    climb = max(1, (peak_height - base + ramp_step - 1) // ramp_step)
    span_width = 2 * climb + 3
    eligible = set(terrain_columns(tiles, ctx, columns))
    spawn_col, flag_col = ctx.player[1], ctx.flagpole[1]
    best_start = -1
    for start in range(columns.start, columns.stop - span_width):
        span = range(start, start + span_width)
        if any(
            col not in eligible
            or heights[col] == 0
            or abs(col - spawn_col) <= 4
            or abs(col - flag_col) <= 4
            for col in span
        ):
            continue
        best_start = start
        break
    if best_start < 0:
        return []
    profile: list[int] = []
    for index in range(span_width):
        if index < climb:
            level_height = min(peak_height, base + ramp_step * (index + 1))
        elif index < climb + 3:
            level_height = peak_height
        else:
            level_height = min(peak_height, base + ramp_step * (span_width - index))
        profile.append(max(base, level_height))
    cells: list[tuple[int, int, int]] = []
    for offset, level_height in enumerate(profile):
        cells.extend(rewrite_stack(tiles, ctx, best_start + offset, level_height))
    return [Edit(op="raise_hill", cells=tuple(cells))] if cells else []


def plan_roughen(
    tiles: np.ndarray,
    ctx: RefineContext,
    *,
    amplitude: int,
    boundaries: int,
    columns: range,
) -> list[Edit]:
    """Modulate stack heights into a sawtooth to lower path smoothness.

    Args:
        tiles: Working tile array.
        ctx: Shared refine context.
        amplitude: Half-height of the sawtooth in rows.
        boundaries: Number of height steps to introduce.
        columns: Column range searched.
    """
    if boundaries <= 0:
        return []
    heights = ground_heights(tiles, ctx)
    eligible = [col for col in terrain_columns(tiles, ctx, columns) if heights[col] >= 2]
    if len(eligible) < 4:
        return []
    edits: list[Edit] = []
    made = 0
    period = max(2, len(eligible) // max(1, boundaries))
    block: list[tuple[int, int, int]] = []
    for index, col in enumerate(eligible):
        if made >= boundaries:
            break
        if (index // period) % 2 == 1:
            base = heights[col]
            block.extend(rewrite_stack(tiles, ctx, col, base + amplitude))
            if index % period == 0:
                made += 2
        elif block:
            edits.append(Edit(op="roughen", cells=tuple(block)))
            block = []
    if block:
        edits.append(Edit(op="roughen", cells=tuple(block)))
    return edits


def plan_add_platform(
    tiles: np.ndarray,
    ctx: RefineContext,
    *,
    width: int,
    height_above: int,
    columns: range,
    max_standing_height: int | None = None,
) -> list[Edit]:
    """Add one mountable floating platform of solid tiles.

    The platform sits ``height_above`` rows over the local surface (within
    jump height so the envelope adopts it) with two clear rows above for
    standing room. Gap columns may sit under the span — the SMB1 idiom of
    platforms bridging pits — as long as one grounded anchor column remains,
    so platform usage keeps climbing after gap difficulty has consumed the
    floor.
    """
    heights = ground_heights(tiles, ctx)
    height = tiles.shape[0]
    eligible = set(terrain_columns(tiles, ctx, columns))
    for start in range(columns.start, columns.stop - width):
        span = range(start, start + width)
        if any(col not in eligible for col in span):
            continue
        grounded = [col for col in span if heights[col] > 0]
        if not grounded:
            continue
        surface_rows = {height - heights[col] - 1 for col in grounded}
        if max(surface_rows) - min(surface_rows) > 1:
            continue
        platform_row = min(surface_rows) - height_above
        if platform_row < 2:
            continue
        if max_standing_height is not None and height - platform_row > max_standing_height:
            continue
        clear = all(
            int(tiles[row, col]) == ctx.empty_id and (row, col) not in ctx.protected
            for col in span
            for row in (platform_row, platform_row - 1, platform_row - 2)
        )
        if not clear:
            continue
        cells = tuple((platform_row, col, ctx.solid_id) for col in span)
        return [Edit(op="add_platform", cells=cells)]
    return []
