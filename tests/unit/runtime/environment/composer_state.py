from __future__ import annotations

import numpy as np
import pytest
from pcgrl.core.library.types import Chunk, ChunkFamily, ChunkLibrary
from pcgrl.runtime.environment.composer_obs import (
    build_composer_observation,
    composer_observation_size,
)
from pcgrl.runtime.environment.composer_state import ComposerCore, ComposeState


def _chunk_tiles(catalog, *, width: int = 16, floor_height: int = 2, gap_cols=()) -> np.ndarray:
    """chunk tiles."""
    tiles = np.full((14, width), catalog.id_for_name("empty"), dtype=np.int16)
    solid = catalog.id_for_name("solid")
    for column in range(width):
        if column in gap_cols:
            continue
        tiles[14 - floor_height :, column] = solid
    return tiles


def _chunk(
    chunk_id: int,
    catalog,
    *,
    family: ChunkFamily = ChunkFamily.MIDDLE,
    width: int = 16,
    floor_height: int = 2,
    gap_cols=(),
    enemies: int = 0,
) -> Chunk:
    """Chunk."""
    tiles = _chunk_tiles(catalog, width=width, floor_height=floor_height, gap_cols=gap_cols)
    if enemies:
        goomba = catalog.id_for_name("goomba")
        for index in range(enemies):
            tiles[14 - floor_height - 1, 1 + index] = goomba
    entry = 0 if 0 in gap_cols else floor_height
    exit_ = 0 if width - 1 in gap_cols else floor_height
    counts: dict[str, int] = {}
    if enemies:
        counts["enemy"] = enemies
    return Chunk(
        chunk_id=chunk_id,
        tiles=tiles,
        family=family,
        level_id="1-1",
        column_offset=chunk_id * 8,
        entry_height=entry,
        exit_height=exit_,
        gap_at_left=entry == 0,
        gap_at_right=exit_ == 0,
        category_counts=counts,
    )


@pytest.fixture
def core(config, catalog):
    """Pytest fixture providing ``core`` for tests."""
    chunks = [
        _chunk(0, catalog, family=ChunkFamily.START),
        _chunk(1, catalog),  # flat middle, height 2
        _chunk(2, catalog, floor_height=7),  # high middle (|7-2|=5 > jump 4)
        _chunk(3, catalog, gap_cols=range(10, 16)),  # exits into a 6-wide gap
        _chunk(4, catalog, gap_cols=range(0, 3)),  # enters from a 3-wide gap
        _chunk(5, catalog, family=ChunkFamily.ENDING),
        _chunk(6, catalog, enemies=12),
        _chunk(7, catalog, floor_height=5),  # reachable step (|5-2|=3)
    ]
    library = ChunkLibrary(
        chunks,
        train_level_ids=["1-1"],
        heldout_level_ids=["8-1"],
        tile_names=[t.name for t in catalog.tiles],
    )
    return ComposerCore(library, config, catalog)


def test_first_decision_restricted_to_start(core) -> None:
    """Verify first decision restricted to start."""
    mask = core.legal_chunk_mask(ComposeState())
    assert mask[0]
    assert not mask[1]
    assert not mask[5]


def test_seam_height_rule(core) -> None:
    """Verify seam height rule."""
    state = core.apply_chunk(ComposeState(), 0)
    mask = core.legal_chunk_mask(state)
    assert mask[1]  # flat -> flat
    assert mask[7]  # step of 3 <= jump_height 4
    assert not mask[2]  # step of 5 > jump_height
    assert not mask[0]  # START never again


def test_seam_gap_rule(core, config) -> None:
    """Verify seam gap rule."""
    state = core.apply_chunk(ComposeState(), 0)
    state = core.apply_chunk(state, 3)  # trailing gap of 6
    mask = core.legal_chunk_mask(state)
    # 6 (trailing) + 3 (leading of chunk 4) = 9 > jump_reach 5 -> illegal;
    # grounded entries are also illegal because the seam gap is uncrossable
    # only when the entry is grounded... the grounded rule applies instead:
    # entry 2 vs exit 0 is not grounded-jumpable, so only gap seams <= 5 pass.
    assert not mask[4]
    assert not mask[1] or config.validation.jump_reach >= 6


def test_gap_seam_within_reach_is_legal(core, catalog, config) -> None:
    """Verify gap seam within reach is legal."""
    # Chunk exiting with a 2-wide gap; next grounded chunk crosses a 2-gap seam.
    chunks = [
        _chunk(0, catalog, family=ChunkFamily.START),
        _chunk(1, catalog, gap_cols=range(14, 16)),  # trailing gap 2
        _chunk(2, catalog),
        _chunk(3, catalog, family=ChunkFamily.ENDING),
    ]
    library = ChunkLibrary(
        chunks,
        train_level_ids=["1-1"],
        heldout_level_ids=["8-1"],
        tile_names=[t.name for t in catalog.tiles],
    )
    core2 = ComposerCore(library, config, catalog)
    state = core2.apply_chunk(ComposeState(), 0)
    state = core2.apply_chunk(state, 1)
    mask = core2.legal_chunk_mask(state)
    assert mask[2]  # 2 + 0 <= jump_reach


def test_width_arithmetic_forces_ending(core, config) -> None:
    """Verify width arithmetic forces ending."""
    state = ComposeState()
    state = core.apply_chunk(state, 0)
    for _ in range(7):
        mask = core.legal_chunk_mask(state)
        assert mask.any()
        state = core.apply_chunk(state, 1)
    # used = 8*16 = 128, remaining 24: middles would leave 8 (< min ending 16).
    mask = core.legal_chunk_mask(state)
    assert mask[5]
    assert not mask[1]
    state = core.apply_chunk(state, 5)
    assert state.closed
    assert core.is_terminal(state)
    assert not core.legal_chunk_mask(state).any()


def test_duplicate_window_relaxes_rather_than_deadlocks(core) -> None:
    """Verify duplicate window relaxes rather than deadlocks."""
    state = core.apply_chunk(ComposeState(), 0)
    state = core.apply_chunk(state, 1)
    mask = core.legal_chunk_mask(state)
    assert not mask[1]  # inside the duplicate window
    assert mask[7]
    # When every alternative is illegal the duplicate rule must yield: compose
    # until only chunk 1 fits structurally.
    for _ in range(6):
        state = core.apply_chunk(state, 1)
    mask = core.legal_chunk_mask(state)
    assert mask[5]


def test_budget_mask_blocks_enemy_overshoot(core, config) -> None:
    """Verify stitch level layout and provenance."""
    state = core.apply_chunk(ComposeState(), 0)
    target = np.zeros(8, dtype=np.float32)  # zero enemy budget
    mask = core.legal_chunk_mask(state, target)
    assert not mask[6]  # 12 enemies >> budget 0 + slack 4
    assert mask[1]
    high = np.ones(8, dtype=np.float32)
    assert core.legal_chunk_mask(state, high)[6]


def test_stitch_level_layout_and_provenance(core, config, catalog) -> None:
    """Verify stitch level layout and provenance."""
    state = core.apply_chunk(ComposeState(), 0)
    for _ in range(7):
        state = core.apply_chunk(state, 1)
    state = core.apply_chunk(state, 5)
    level = core.stitch_level(state)
    assert level.width_tiles == config.game.width_tiles
    assert level.height_tiles == config.game.height_tiles
    player = catalog.id_for_name("player")
    flag = catalog.id_for_name("flagpole")
    assert (level.tiles == player).sum() == 1
    assert (level.tiles == flag).sum() == 1
    solid = catalog.id_for_name("solid")
    # Chunks are flat-floored here, so the whole bottom row is ground; gaps in
    # chunk floors must NOT be filled (fill_full_ground is composer-ignored).
    assert (level.tiles[-1] == solid).all()
    assert len(level.metadata["provenance"]) == 9
    assert level.metadata["chunk_ids"][0] == 0


def test_observation_shape_and_bounds(core) -> None:
    """Verify observation shape and bounds."""
    target = np.full(8, 0.5, dtype=np.float32)
    state = ComposeState()
    obs = build_composer_observation(core, state, target, max_decisions=19)
    assert obs.shape == (composer_observation_size(),)
    state = core.apply_chunk(state, 0)
    obs = build_composer_observation(core, state, target, max_decisions=19)
    assert obs.shape == (composer_observation_size(),)
    assert ((obs >= -1.0) & (obs <= 1.0)).all()
    assert obs.dtype == np.float32
