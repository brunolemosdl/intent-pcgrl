from __future__ import annotations

from collections import deque

import numpy as np
import pytest
from pcgrl.core.helpers.path import resolve_path
from pcgrl.runtime.validators.reachability import _flood, reachable_cells


def _flood_reference(tiles, catalog, *, jump_height=4, jump_reach=5, start=None):
    """Scalar movement oracle (solid-top support semantics).

    Enemies block movement but are never floors: standing and fall landings
    require an ``is_solid_top`` tile below, and falls pass through enemies."""
    height, width = tiles.shape
    passable = np.zeros((height, width), dtype=bool)
    support = np.zeros((height, width), dtype=bool)
    for r in range(height):
        for c in range(width):
            tile = catalog.tile_for_id(int(tiles[r, c]))
            passable[r, c] = tile.passable
            support[r, c] = tile.is_solid_top
    if start is None:
        player_id = catalog.id_for_name("player")
        for r in range(height):
            for c in range(width):
                if int(tiles[r, c]) == player_id:
                    start = (r, c)
                    break
            if start is not None:
                break
    if start is None:
        return set()

    def head_clear(r, c):
        """Cell above is passable or off the grid top."""
        return r == 0 or passable[r - 1, c]

    def can_stand(r, c):
        """Passable + head clear + solid top below, never the last row."""
        if r < 0 or r >= height - 1 or c < 0 or c >= width:
            return False
        return bool(passable[r, c] and head_clear(r, c) and support[r + 1, c])

    def fall_to_ground(r, c):
        """Row reached after falling down column ``c`` onto a solid top."""
        while r + 1 < height and not support[r + 1, c]:
            r += 1
        return r

    visited = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for dc in (-1, 1):
            nc = c + dc
            if nc < 0 or nc >= width:
                continue
            if can_stand(r, nc):
                if (r, nc) not in visited:
                    visited.add((r, nc))
                    queue.append((r, nc))
            elif passable[r, nc] and head_clear(r, nc):
                nr = fall_to_ground(r, nc)
                if can_stand(nr, nc) and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        for dy in range(1, jump_height + 1):
            nr_apex = r - dy
            if nr_apex < 0:
                break
            for dx in range(-jump_reach, jump_reach + 1):
                nc_j = c + dx
                if nc_j < 0 or nc_j >= width:
                    continue
                if not passable[nr_apex, nc_j] or not head_clear(nr_apex, nc_j):
                    continue
                nr = fall_to_ground(nr_apex, nc_j)
                if can_stand(nr, nc_j) and (nr, nc_j) not in visited:
                    visited.add((nr, nc_j))
                    queue.append((nr, nc_j))
    return visited


def test_flood_matches_scalar_reference_bit_exactly(config, catalog) -> None:
    """The optimised flood visits exactly the reference set on real levels."""
    library_root = resolve_path(config.library.output_root)
    if not (library_root / "chunk_library.npz").is_file():
        pytest.skip("Chunk library not built")
    from pcgrl.core.library.types import ChunkLibrary
    from pcgrl.runtime.environment.composer_policies import rollout_random_legal
    from pcgrl.runtime.environment.composer_state import ComposerCore

    library = ChunkLibrary.load(library_root)
    core = ComposerCore(library, config, catalog)
    rng = np.random.default_rng(7)
    for episode in range(12):
        target = rng.uniform(0.0, 1.0, size=8).astype(np.float32)
        tiles = core.stitch_level(rollout_random_legal(core, target, rng)).tiles
        for start in (None, (tiles.shape[0] - 2, 1)):
            assert _flood(tiles, catalog, start=start) == _flood_reference(
                tiles, catalog, start=start
            ), f"flood differs on episode {episode} (start={start})"


def _grid(catalog, rows: list[str]) -> np.ndarray:
    """Tiny ASCII fixture: X solid, M player, - empty."""
    mapping = {
        "X": catalog.id_for_name("solid"),
        "M": catalog.id_for_name("player"),
        "-": catalog.id_for_name("empty"),
    }
    return np.array([[mapping[ch] for ch in row] for row in rows], dtype=np.int16)


def test_flood_walks_and_jumps(catalog) -> None:
    """Flat walking plus a jumpable step are both visited."""
    tiles = _grid(
        catalog,
        [
            "----------",
            "------XX--",
            "M---------",
            "XXXXXXXXXX",
        ],
    )
    standable = reachable_cells(tiles, catalog).standable
    assert (2, 0) in standable
    assert (2, 5) in standable
    assert (0, 6) in standable


def test_flood_respects_jump_height(catalog) -> None:
    """A platform requiring a 5-row jump is unreachable at jump_height=4.

    The jump model is apex-teleport (no arc collision — the known generous
    physics measured against the A* in E0), so the limit under test is the
    vertical rise to the apex, not wall blocking.
    """
    tiles = _grid(
        catalog,
        [
            "----------",
            "------XX--",
            "----------",
            "----------",
            "----------",
            "M---------",
            "XXXXXXXXXX",
        ],
    )
    low = reachable_cells(tiles, catalog, jump_height=4, jump_reach=5).standable
    assert (0, 6) not in low  # apex row 0 would need dy=5
    high = reachable_cells(tiles, catalog, jump_height=5, jump_reach=5).standable
    assert (0, 6) in high


def test_flood_safe_landing_rule(catalog) -> None:
    """A fall into a bottomless pit visits nothing beyond the edge."""
    tiles = _grid(
        catalog,
        [
            "M---------",
            "XXX-------",
        ],
    )
    standable = reachable_cells(tiles, catalog).standable
    assert all(col <= 2 for _, col in standable)


def test_flood_without_player_or_start_is_empty(catalog) -> None:
    """No player tile and no start override returns an empty map."""
    tiles = _grid(catalog, ["----", "XXXX"])
    assert reachable_cells(tiles, catalog).standable == frozenset()
    assert reachable_cells(tiles, catalog, start=(0, 0)).standable != frozenset()
