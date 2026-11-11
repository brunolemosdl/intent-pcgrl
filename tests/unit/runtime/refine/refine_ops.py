from __future__ import annotations

import itertools

import numpy as np
import pytest
from pcgrl.core.domain.level import AreaType, LevelGrid
from pcgrl.runtime.metrics.intent import INTENT_KEYS, compute_intent_vector
from pcgrl.runtime.refine.context import build_context
from pcgrl.runtime.refine.gates import flood_reaches_flag, grid_from_tiles
from pcgrl.runtime.refine.ops import (
    apply_edit,
    ground_heights,
    pipe_clusters,
    plan_add_pipe,
    plan_add_platform,
    plan_add_tiles_on_surface,
    plan_dig_gap,
    plan_fill_gaps,
    plan_flatten,
    plan_raise_hill,
    plan_remove_pipes,
    plan_remove_tiles,
    plan_repair_pipes,
    revert_edit,
)

_DIM = {key: index for index, key in enumerate(INTENT_KEYS)}


@pytest.fixture
def ctx(config, catalog):
    """Shared refine context for the session config."""
    return build_context(config, catalog)


@pytest.fixture
def flat_tiles(config, catalog, ctx):
    """Playable flat level: 2-row floor, endpoints stamped, some decorations."""
    height, width = config.game.height_tiles, config.game.width_tiles
    tiles = np.full((height, width), ctx.empty_id, dtype=np.int16)
    tiles[height - 2 :, :] = ctx.solid_id
    goomba = ctx.enemy_ids[0]
    for col in (30, 60, 90):
        tiles[height - 3, col] = goomba
    for col in (40, 41, 42):
        tiles[height - 4, col] = ctx.coin_id
    tiles[ctx.player] = catalog.id_for_name("player")
    tiles[ctx.flagpole] = catalog.id_for_name("flagpole")
    return tiles


def _measure(tiles, config, catalog):
    """Intent vector of a raw tile array."""
    return compute_intent_vector(grid_from_tiles(tiles, config), catalog, config)


def test_apply_and_revert_roundtrip(flat_tiles, ctx) -> None:
    """Reverting an applied edit restores the array bit-exactly."""
    before = flat_tiles.copy()
    edits = plan_remove_tiles(
        flat_tiles,
        ctx,
        ctx.enemy_ids,
        keep=0,
        columns=range(ctx.width),
        op="remove_enemies",
    )
    snapshot = apply_edit(flat_tiles, edits[0])
    assert not np.array_equal(before, flat_tiles)
    revert_edit(flat_tiles, snapshot)
    assert np.array_equal(before, flat_tiles)


def test_remove_tiles_reaches_exact_zero(flat_tiles, ctx, config, catalog) -> None:
    """Removing every enemy measures exactly 0 on the density band."""
    edits = plan_remove_tiles(
        flat_tiles,
        ctx,
        ctx.enemy_ids,
        keep=0,
        columns=range(ctx.width),
        op="remove_enemies",
    )
    apply_edit(flat_tiles, edits[0])
    assert _measure(flat_tiles, config, catalog)[_DIM["enemy_density"]] == 0.0


def test_add_tiles_hits_target_count(flat_tiles, ctx) -> None:
    """Surface placement adds the requested number of tiles, spaced out."""
    before = set(map(tuple, np.argwhere(np.isin(flat_tiles, list(ctx.enemy_ids))).tolist()))
    edits = plan_add_tiles_on_surface(
        flat_tiles,
        ctx,
        ctx.enemy_ids[0],
        add=10,
        columns=range(ctx.width),
        op="add_enemies",
        spacing=3,
    )
    apply_edit(flat_tiles, edits[0])
    after = set(map(tuple, np.argwhere(np.isin(flat_tiles, list(ctx.enemy_ids))).tolist()))
    new_columns = sorted(col for _, col in after - before)
    assert len(new_columns) == 10
    assert all(b - a >= 3 for a, b in itertools.pairwise(new_columns))


def test_dig_and_fill_gaps_are_inverse_on_the_metric(flat_tiles, ctx, config, catalog) -> None:
    """Digging raises gap difficulty; filling back to 0 restores it."""
    edits = plan_dig_gap(flat_tiles, ctx, gap_width=4, columns=range(ctx.width))
    apply_edit(flat_tiles, edits[0])
    assert _measure(flat_tiles, config, catalog)[_DIM["gap_difficulty"]] > 0.0
    assert flood_reaches_flag(flat_tiles, catalog, config)
    fills = plan_fill_gaps(flat_tiles, ctx, target_score=0.0, columns=range(ctx.width))
    for edit in fills:
        apply_edit(flat_tiles, edit)
    assert _measure(flat_tiles, config, catalog)[_DIM["gap_difficulty"]] == 0.0


def test_raise_hill_and_flatten_move_verticality(flat_tiles, ctx, config, catalog) -> None:
    """A ramped hill raises verticality; flattening removes it again."""
    edits = plan_raise_hill(flat_tiles, ctx, peak_height=11, ramp_step=1, columns=range(ctx.width))
    apply_edit(flat_tiles, edits[0])
    assert _measure(flat_tiles, config, catalog)[_DIM["verticality"]] > 0.3
    assert flood_reaches_flag(flat_tiles, catalog, config)
    flat = plan_flatten(flat_tiles, ctx, base_height=2, columns=range(ctx.width))
    apply_edit(flat_tiles, flat[0])
    assert _measure(flat_tiles, config, catalog)[_DIM["verticality"]] == 0.0


def test_add_pipe_is_validator_shaped_and_grounded(flat_tiles, ctx) -> None:
    """An added pipe is one width-2, height>=2 grounded cluster."""
    edits = plan_add_pipe(flat_tiles, ctx, pipe_height=3, columns=range(ctx.width))
    apply_edit(flat_tiles, edits[0])
    clusters = pipe_clusters(flat_tiles, ctx)
    assert len(clusters) == 1
    cols = sorted({col for _, col in clusters[0]})
    rows = sorted({row for row, _ in clusters[0]})
    assert len(cols) == 2
    assert len(rows) == 3
    bottom = max(rows)
    assert all(int(flat_tiles[bottom + 1, col]) in ctx.solid_like for col in cols)


def test_repair_pipes_removes_malformed_clusters(flat_tiles, ctx) -> None:
    """A width-3 pipe blob (seam merge artefact) is removed whole."""
    height = flat_tiles.shape[0]
    for col in (70, 71, 72):
        flat_tiles[height - 3, col] = ctx.pipe_top_left
        flat_tiles[height - 4, col] = ctx.pipe_body_left
    edits = plan_repair_pipes(flat_tiles, ctx)
    assert edits
    for edit in edits:
        apply_edit(flat_tiles, edit)
    assert not pipe_clusters(flat_tiles, ctx)


def test_remove_pipes_keeps_whole_clusters(flat_tiles, ctx) -> None:
    """Pipe removal never leaves partial clusters behind."""
    edits = plan_add_pipe(flat_tiles, ctx, pipe_height=3, columns=range(ctx.width))
    apply_edit(flat_tiles, edits[0])
    removals = plan_remove_pipes(flat_tiles, ctx, keep_tiles=0, columns=range(ctx.width))
    for edit in removals:
        apply_edit(flat_tiles, edit)
    assert not np.isin(flat_tiles, list(ctx.pipe_ids)).any()


def test_add_platform_is_mountable_and_raises_platform_usage(
    flat_tiles, ctx, config, catalog
) -> None:
    """A platform within jump height enters the envelope as elevated."""
    before = _measure(flat_tiles, config, catalog)[_DIM["platform_usage"]]
    for _ in range(4):
        edits = plan_add_platform(
            flat_tiles, ctx, width=6, height_above=3, columns=range(ctx.width)
        )
        if not edits:
            break
        apply_edit(flat_tiles, edits[0])
    after = _measure(flat_tiles, config, catalog)[_DIM["platform_usage"]]
    assert after > before


def test_protected_cells_never_written(flat_tiles, ctx) -> None:
    """No planner output touches endpoints or reserved cells."""
    planners = [
        plan_dig_gap(flat_tiles, ctx, gap_width=4, columns=range(ctx.width)),
        plan_flatten(flat_tiles, ctx, base_height=2, columns=range(ctx.width)),
        plan_raise_hill(flat_tiles, ctx, peak_height=8, ramp_step=1, columns=range(ctx.width)),
        plan_add_pipe(flat_tiles, ctx, pipe_height=3, columns=range(ctx.width)),
    ]
    for edits in planners:
        for edit in edits:
            for row, col, _ in edit.cells:
                assert (row, col) not in ctx.protected


def test_ground_heights_matches_metric_profile(flat_tiles, ctx, config, catalog) -> None:
    """The ops-side ground profile mirrors the metric helper."""
    from pcgrl.runtime.metrics.intent import ground_height_profile

    level = LevelGrid(
        tiles=flat_tiles.copy(),
        width_tiles=ctx.width,
        height_tiles=ctx.height,
        tile_size_px=16,
        area_type=AreaType.OVERWORLD,
    )
    assert ground_heights(flat_tiles, ctx) == ground_height_profile(level, catalog, config)
