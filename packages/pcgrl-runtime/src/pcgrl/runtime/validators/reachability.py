from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from pcgrl.core.tiles.catalog import TileCatalog

# Module-level defaults; the single source of truth is
# config.validation.jump_height/jump_reach, calibrated against the bridge A* by
# the `calibrate` command. New code should pass them explicitly.
_JUMP_HEIGHT = 4
_JUMP_REACH = 5


@dataclass(frozen=True)
class ReachabilityMap:
    """Result of a single reachability flood from the player spawn.

    Attributes:
        standable: Cells where Mario can stand (solid below, head clear).
    """

    standable: frozenset[tuple[int, int]]


def _passable_grid(tiles: np.ndarray, catalog: TileCatalog) -> np.ndarray:
    """Vectorised passability mask via a per-id lookup table.

    The LUT costs one ``tile_for_id`` call per catalog entry (~14) instead of
    one per cell (~2128), and the gather is a single numpy indexing op.
    """
    lut = np.zeros(catalog.size, dtype=bool)
    for tile_id in range(catalog.size):
        lut[tile_id] = catalog.tile_for_id(tile_id).passable
    return lut[tiles]  # type: ignore[no-any-return]


def _support_grid(tiles: np.ndarray, catalog: TileCatalog) -> np.ndarray:
    """Vectorised solid-top mask: cells Mario can stand or land on.

    Distinct from impassability — enemies block movement but are not floors
    (they walk away or die), so standing and fall-landing require an actual
    ``is_solid_top`` tile; floating enemy spawns therefore never inflate the
    path envelope.
    """
    lut = np.zeros(catalog.size, dtype=bool)
    for tile_id in range(catalog.size):
        lut[tile_id] = catalog.tile_for_id(tile_id).is_solid_top
    return lut[tiles]  # type: ignore[no-any-return]


def _flood(
    tiles: np.ndarray,
    catalog: TileCatalog,
    *,
    jump_height: int = _JUMP_HEIGHT,
    jump_reach: int = _JUMP_REACH,
    start: tuple[int, int] | None = None,
) -> set[tuple[int, int]]:
    """Flood from the player spawn, returning the standable cells.

    Single source of truth for reachability: the intent path envelope, the
    terminal reward's playability term, and the physics calibration all run
    this exact BFS. ``start`` overrides the player-tile lookup (used by the
    intent metrics to flood corpus levels that carry no player tile).

    Description:
        This loop dominated training time (~97%), so the per-move costs are
        precomputed: ``head``/``stand``/``fall`` tables replace the per-cell
        predicates and the row-by-row descent, then everything is converted to
        Python lists because scalar indexing into numpy arrays is ~5x slower
        than list indexing inside a tight BFS. Movement semantics are pinned
        bit-exactly against the scalar reference implementation by
        ``tests/unit/runtime/validators/reachability.py``:

        - walk +-1 column onto a standable cell;
        - walk off an edge and fall straight down (safe-landing rule: a fall
          into a pit visits nothing; falls pass through enemies — stomping —
          and stop only on solid tops);
        - jump up to ``jump_height`` rows with up to ``jump_reach`` columns of
          horizontal reach through an open apex cell, then fall to landing;
        - standing requires an ``is_solid_top`` tile below: enemies block
          movement but are never floors.
    """
    height, width = tiles.shape
    passable = _passable_grid(tiles, catalog)
    if start is None:
        player_id = catalog.id_for_name("player")
        hits = np.argwhere(tiles == player_id)
        if hits.size:
            start = (int(hits[0][0]), int(hits[0][1]))
    if start is None:
        return set()

    support = _support_grid(tiles, catalog)
    # head[r, c]: cell above is passable or off the grid top.
    head = np.empty((height, width), dtype=bool)
    head[0, :] = True
    head[1:, :] = passable[:-1, :]
    # stand[r, c]: passable + head clear + solid top below (never the last row).
    stand = np.zeros((height, width), dtype=bool)
    stand[: height - 1, :] = (
        passable[: height - 1, :] & head[: height - 1, :] & support[1:, :]
    )
    # fall[r, c]: landing row when falling down column c from row r; falls
    # stop only on solid tops (an enemy underneath is not a floor).
    fall = np.empty((height, width), dtype=np.int64)
    fall[height - 1, :] = height - 1
    for r in range(height - 2, -1, -1):
        fall[r, :] = np.where(support[r + 1, :], r, fall[r + 1, :])
    # open[r, c]: a jump apex / walk-through cell (passable with head room).
    open_l = (passable & head).tolist()
    stand_l = stand.tolist()
    fall_l = fall.tolist()

    visited: set[tuple[int, int]] = {start}
    queue: deque[tuple[int, int]] = deque([start])
    while queue:
        r, c = queue.popleft()
        for dc in (-1, 1):
            nc = c + dc
            if nc < 0 or nc >= width:
                continue
            if stand_l[r][nc]:
                if (r, nc) not in visited:
                    visited.add((r, nc))
                    queue.append((r, nc))
            elif open_l[r][nc]:
                nr = fall_l[r][nc]
                if stand_l[nr][nc] and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        lo = c - jump_reach
        if lo < 0:
            lo = 0
        hi = c + jump_reach
        if hi >= width:
            hi = width - 1
        for dy in range(1, jump_height + 1):
            nr_apex = r - dy
            if nr_apex < 0:
                break
            apex_open = open_l[nr_apex]
            for nc_j in range(lo, hi + 1):
                if not apex_open[nc_j]:
                    continue
                nr = fall_l[nr_apex][nc_j]
                if stand_l[nr][nc_j] and (nr, nc_j) not in visited:
                    visited.add((nr, nc_j))
                    queue.append((nr, nc_j))
    return visited


def reachable_cells(
    level_or_tiles: object,
    catalog: TileCatalog,
    *,
    jump_height: int = _JUMP_HEIGHT,
    jump_reach: int = _JUMP_REACH,
    start: tuple[int, int] | None = None,
) -> ReachabilityMap:
    """Flood from spawn and return the standable cells.

    Args:
        level_or_tiles: ``LevelGrid`` or raw tile array.
        catalog: Tile catalog.
        jump_height: Maximum jump height in rows.
        jump_reach: Maximum horizontal jump reach in columns.
        start: Optional spawn override (for corpus levels without player).

    Returns:
        ``ReachabilityMap`` with the frozen standable set.
    """
    tiles = getattr(level_or_tiles, "tiles", level_or_tiles)
    standable = _flood(
        tiles,  # type: ignore[arg-type]
        catalog,
        jump_height=jump_height,
        jump_reach=jump_reach,
        start=start,
    )
    return ReachabilityMap(standable=frozenset(standable))
