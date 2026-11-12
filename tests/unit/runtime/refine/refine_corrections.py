from __future__ import annotations

import itertools

import numpy as np
import pytest
from pcgrl.runtime.metrics.intent import INTENT_KEYS, measure_intent
from pcgrl.runtime.refine.context import build_context, target_amplitude
from pcgrl.runtime.refine.corrections import (
    LOWER,
    RAISE,
    block_runs,
    correction_needed,
    is_literal_flat,
    plan_correction,
    raw_gap,
)
from pcgrl.runtime.refine.gates import RefineGate, grid_from_tiles
from pcgrl.runtime.refine.ops import (
    Edit,
    apply_edit,
    ground_heights,
    pipe_clusters,
    plan_dig_gap,
    plan_remove_blocks,
    plan_smooth,
    rewrite_stack,
)

_DIM = {key: index for index, key in enumerate(INTENT_KEYS)}


@pytest.fixture
def ctx(config, catalog):
    """Shared refine context for the session config."""
    return build_context(config, catalog)


@pytest.fixture
def floor(config, catalog, ctx):
    """Playable corpus-style level: one-row floor with the endpoints stamped."""
    height, width = config.game.height_tiles, config.game.width_tiles
    tiles = np.full((height, width), ctx.empty_id, dtype=np.int16)
    tiles[height - ctx.base_height :, :] = ctx.solid_id
    tiles[ctx.player] = catalog.id_for_name("player")
    tiles[ctx.flagpole] = catalog.id_for_name("flagpole")
    return tiles


def _state(tiles, config, catalog):
    return measure_intent(grid_from_tiles(tiles, config), catalog, config)


def _target(tiles, config, catalog, **overrides: float) -> np.ndarray:
    """The level's own measurement with some coordinates overridden."""
    target = _state(tiles, config, catalog).vector.astype(np.float32).copy()
    for key, value in overrides.items():
        target[_DIM[key]] = value
    return target


def _apply(tiles, edits) -> None:
    for edit in edits:
        apply_edit(tiles, edit)


def _density(count: int, band_max: float, ctx) -> float:
    return count / (band_max * ctx.area)


def test_corpus_floor_is_one_row(ctx) -> None:
    """Terrain ops treat the layout's ground row as flat ground."""
    assert ctx.base_height == 1


def test_removal_keeps_what_the_target_asks_for(floor, config, catalog, ctx) -> None:
    """Six enemies, four wanted: one segment gives up exactly the surplus."""
    row = floor.shape[0] - ctx.base_height - 1
    for column in (30, 32, 34, 36, 60, 90):
        floor[row, column] = ctx.enemy_ids[0]
    band = config.metrics.target_density_ranges.enemy.max
    target = _target(floor, config, catalog, enemy_density=_density(4, band, ctx))
    state = _state(floor, config, catalog)
    assert correction_needed(floor, ctx, target, state, "enemy_density", LOWER, tolerance=0.05)
    edits = plan_correction(
        floor, ctx, target, state, "enemy_density", LOWER, columns=range(19, 38), level=0
    )
    _apply(floor, edits)
    assert int(np.isin(floor, ctx.enemy_ids).sum()) == 4
    state = _state(floor, config, catalog)
    assert not correction_needed(floor, ctx, target, state, "enemy_density", LOWER, tolerance=0.05)
    assert (
        plan_correction(
            floor, ctx, target, state, "enemy_density", LOWER, columns=range(19, 38), level=0
        )
        == []
    )


@pytest.mark.parametrize(("level", "expected"), [(0, 3), (1, 5)])
def test_addition_is_capped_by_level_and_deficit(
    floor, config, catalog, ctx, level, expected
) -> None:
    """Level 0 adds a small batch; level 1 everything the target still needs."""
    band = config.metrics.target_density_ranges.coin.max
    target = _target(floor, config, catalog, coin_density=_density(5, band, ctx))
    state = _state(floor, config, catalog)
    edits = plan_correction(
        floor, ctx, target, state, "coin_density", RAISE, columns=range(38, 76), level=level
    )
    _apply(floor, edits)
    assert int((floor == ctx.coin_id).sum()) == expected


def test_pipe_is_added_on_the_one_row_floor(floor, config, catalog, ctx) -> None:
    """Pipes stand directly on the corpus floor, as in the source levels."""
    band = config.metrics.target_density_ranges.pipe.max
    target = _target(floor, config, catalog, pipe_density=_density(12, band, ctx))
    state = _state(floor, config, catalog)
    edits = plan_correction(
        floor, ctx, target, state, "pipe_density", RAISE, columns=range(38, 57), level=0
    )
    assert edits
    _apply(floor, edits)
    clusters = pipe_clusters(floor, ctx)
    assert len(clusters) == 1
    bottom = max(row for row, _ in clusters[0])
    assert bottom == floor.shape[0] - ctx.base_height - 1
    ok, reason = RefineGate(config, catalog).check(floor)
    assert ok, reason


def test_pipe_step_must_help_more_than_it_overshoots(floor, config, catalog, ctx) -> None:
    """A six-tile pipe is not removed for a two-tile surplus, but is for four."""
    band = config.metrics.target_density_ranges.pipe.max
    seed = _target(floor, config, catalog, pipe_density=_density(12, band, ctx))
    state = _state(floor, config, catalog)
    _apply(
        floor,
        plan_correction(
            floor, ctx, seed, state, "pipe_density", RAISE, columns=range(38, 57), level=0
        ),
    )
    tiles_in_pipe = len(pipe_clusters(floor, ctx)[0])
    state = _state(floor, config, catalog)
    full = range(0, ctx.width)
    keep = _target(floor, config, catalog, pipe_density=_density(tiles_in_pipe - 2, band, ctx))
    assert plan_correction(floor, ctx, keep, state, "pipe_density", LOWER, columns=full) == []
    drop = _target(floor, config, catalog, pipe_density=_density(tiles_in_pipe - 4, band, ctx))
    assert plan_correction(floor, ctx, drop, state, "pipe_density", LOWER, columns=full)


def test_filled_gap_rises_to_the_floor_level(floor, config, catalog, ctx) -> None:
    """A filled pit matches its shoulders instead of leaving a bump."""
    floor[:, 60:63] = ctx.empty_id
    target = _target(floor, config, catalog, gap_difficulty=0.0)
    state = _state(floor, config, catalog)
    edits = plan_correction(
        floor, ctx, target, state, "gap_difficulty", LOWER, columns=range(57, 76), level=0
    )
    _apply(floor, edits)
    assert ground_heights(floor, ctx)[58:65] == [ctx.base_height] * 7


def test_roughen_works_on_the_one_row_floor(floor, config, catalog, ctx) -> None:
    """Lowering smoothness finds ground to modulate on plain corpus floor."""
    target = _target(floor, config, catalog, path_smoothness=0.2, verticality=0.65)
    state = _state(floor, config, catalog)
    edits = plan_correction(
        floor, ctx, target, state, "path_smoothness", LOWER, columns=range(38, 57), level=0
    )
    assert edits
    _apply(floor, edits)
    heights = ground_heights(floor, ctx)[38:57]
    steps = sum(1 for left, right in itertools.pairwise(heights) if left != right)
    assert steps == 4
    assert max(ground_heights(floor, ctx)[38:57]) == ctx.base_height + 1


@pytest.mark.parametrize("level", [1, 2])
def test_hill_reaches_the_target_amplitude(floor, config, catalog, ctx, level) -> None:
    """One hill brings the envelope to the amplitude the target maps to.

    The hill starts inside the segment and may run past its right edge, so
    even the gentle ramp fits."""
    target = _target(floor, config, catalog, verticality=0.9)
    state = _state(floor, config, catalog)
    columns = range(38, 57)
    assert correction_needed(floor, ctx, target, state, "verticality", RAISE, tolerance=0.05)
    edits = plan_correction(
        floor, ctx, target, state, "verticality", RAISE, columns=columns, level=level
    )
    assert edits
    touched = sorted({col for _, col, _ in edits[0].cells})
    assert columns.start <= touched[0] < columns.stop
    _apply(floor, edits)
    state = _state(floor, config, catalog)
    assert state.amplitude == target_amplitude(0.9, config)
    assert not correction_needed(floor, ctx, target, state, "verticality", RAISE, tolerance=0.05)
    ok, reason = RefineGate(config, catalog).check(floor)
    assert ok, reason


def test_summit_rises_by_the_missing_rows(floor, config, catalog, ctx) -> None:
    """Level zero adds amplitude on top of the summit the level already has.

    A staircase with a two-column top gains two rows: the first top column
    stays as a step and only the second one grows, two tiles in all. On a
    plain floor there is no summit and level zero plans nothing."""
    plain = _state(floor, config, catalog)
    wanted = _target(floor, config, catalog, verticality=1.0)
    assert (
        plan_correction(
            floor, ctx, wanted, plain, "verticality", RAISE, columns=range(38, 57), level=0
        )
        == []
    )
    height = floor.shape[0]
    for offset, stack in enumerate((2, 3, 4, 5, 6, 7, 8, 9, 9)):
        floor[height - stack :, 40 + offset] = ctx.solid_id
    state = _state(floor, config, catalog)
    assert state.amplitude == 8
    target = _target(floor, config, catalog, verticality=0.83)
    assert target_amplitude(0.83, config) == 10
    edits = plan_correction(
        floor, ctx, target, state, "verticality", RAISE, columns=range(38, 57), level=0
    )
    assert [edit.op for edit in edits] == ["raise_summit"]
    assert len(edits[0].cells) == 2
    _apply(floor, edits)
    assert ground_heights(floor, ctx)[47:49] == [9, 11]
    state = _state(floor, config, catalog)
    assert state.amplitude == 10
    ok, reason = RefineGate(config, catalog).check(floor)
    assert ok, reason


def _stack(tiles: np.ndarray, ctx, columns: range, height: int) -> None:
    """Raise the ground of ``columns`` to ``height`` rows."""
    tiles[tiles.shape[0] - height :, columns.start : columns.stop] = ctx.solid_id


def test_smoothing_takes_the_narrowest_bump_and_only_what_is_needed(floor, ctx) -> None:
    """A one-column spike comes down before a wide plateau is touched.

    Two rows of steps are asked for: the spike drops one row (two steps
    gone) and the plateau stays. With no limit every bump is levelled."""
    _stack(floor, ctx, range(45, 46), 3)
    _stack(floor, ctx, range(50, 54), 2)
    edits = plan_smooth(floor, ctx, columns=range(38, 57), rows=2)
    assert len(edits) == 1
    assert {col for _, col, _ in edits[0].cells} == {45}
    _apply(floor, edits)
    heights = ground_heights(floor, ctx)
    assert heights[45] == 2
    assert heights[50:54] == [2, 2, 2, 2]
    _apply(floor, plan_smooth(floor, ctx, columns=range(38, 57)))
    assert ground_heights(floor, ctx)[38:57] == [ctx.base_height] * 19


def test_smoothing_fills_a_dip_to_its_shoulders(floor, ctx) -> None:
    """A narrow dip rises to the ground around it; the wide terrace stays."""
    _stack(floor, ctx, range(60, 70), 3)
    floor[: floor.shape[0] - ctx.base_height, 64:66] = ctx.empty_id
    assert ground_heights(floor, ctx)[63:67] == [3, 1, 1, 3]
    _apply(floor, plan_smooth(floor, ctx, columns=range(57, 76)))
    assert ground_heights(floor, ctx)[60:70] == [3] * 10


def test_smoothing_spares_the_peak_the_target_stands_on(floor, ctx) -> None:
    """The summit that carries the verticality target is not cut."""
    for offset, stack in enumerate((2, 3, 4, 5, 6, 7, 8, 9, 9)):
        _stack(floor, ctx, range(40 + offset, 41 + offset), stack)
    assert plan_smooth(floor, ctx, columns=range(38, 57), keep_height=9) == []
    edits = plan_smooth(floor, ctx, columns=range(38, 57), rows=2)
    assert {col for _, col, _ in edits[0].cells} == {47, 48}


def test_platform_removal_fits_the_surplus(floor, config, catalog, ctx) -> None:
    """Four elevated columns too many: the four-column platform goes, not a wide one."""
    row = floor.shape[0] - ctx.base_height - 4
    spans = (range(20, 24), range(40, 48), range(70, 78), range(100, 108))
    for span in spans:
        floor[row, span.start : span.stop] = ctx.solid_id
    state = _state(floor, config, catalog)
    assert state.elevated_columns == 28
    band = config.metrics.platform_usage_band
    wanted = (24 / state.path_columns - band.min) / (band.max - band.min)
    target = _target(floor, config, catalog, platform_usage=wanted)
    full = range(0, ctx.width)
    assert correction_needed(floor, ctx, target, state, "platform_usage", LOWER, tolerance=0.05)
    edits = plan_correction(floor, ctx, target, state, "platform_usage", LOWER, columns=full)
    assert len(edits) == 1
    assert sorted(col for _, col, _ in edits[0].cells) == list(spans[0])
    _apply(floor, edits)
    assert _state(floor, config, catalog).elevated_columns == 24


def test_lowering_platform_usage_keeps_the_block_count(floor, config, catalog, ctx) -> None:
    """Mountable blocks are lifted out of reach, not deleted, one walkway at a time.

    Six elevated columns too many: the six-brick walkway goes up as a whole,
    the long one stays, and every brick is still in the level. A walkway
    wider than twice the surplus is left alone."""
    row = floor.shape[0] - ctx.base_height - 4
    floor[row, 20:26] = ctx.block_ids[0]
    floor[row, 60:78] = ctx.block_ids[0]
    state = _state(floor, config, catalog)
    assert state.elevated_columns == 24
    blocks = int(np.isin(floor, ctx.block_ids).sum())
    band = config.metrics.platform_usage_band
    wanted = (18 / state.path_columns - band.min) / (band.max - band.min)
    target = _target(floor, config, catalog, platform_usage=wanted, verticality=0.3)
    full = range(0, ctx.width)
    edits = plan_correction(floor, ctx, target, state, "platform_usage", LOWER, columns=full)
    assert [edit.op for edit in edits] == ["raise_blocks"] * 6
    assert {col for edit in edits for _, col, _ in edit.cells} == set(range(20, 26))
    _apply(floor, edits)
    assert int(np.isin(floor, ctx.block_ids).sum()) == blocks
    state = _state(floor, config, catalog)
    assert state.elevated_columns == 18
    wanted = (12 / state.path_columns - band.min) / (band.max - band.min)
    target = _target(floor, config, catalog, platform_usage=wanted, verticality=0.3)
    assert plan_correction(floor, ctx, target, state, "platform_usage", LOWER, columns=full) == []


def test_block_runs_are_lifted_whole(floor, config, catalog, ctx) -> None:
    """A walkway counts as one structure, wherever the segment ends.

    The lone block and the walkway both start in the segment; the walkway
    runs past its edge and is still returned whole, with the elevated
    columns each one carries."""
    row = floor.shape[0] - ctx.base_height - 4
    floor[row, 45] = ctx.block_ids[0]
    floor[row, 52:62] = ctx.block_ids[0]
    state = _state(floor, config, catalog)
    runs = block_runs(floor, ctx, state, range(38, 57))
    assert [(carried, len(lifts)) for carried, lifts in runs] == [(1, 1), (10, 10)]
    assert block_runs(floor, ctx, state, range(57, 76)) == []


def test_smoothing_lifts_only_the_runs_platform_usage_can_spare(
    floor, config, catalog, ctx
) -> None:
    """With one column to spare the lone block goes up and the walkway stays."""
    row = floor.shape[0] - ctx.base_height - 4
    for start in (10, 80, 100, 120):
        floor[row, start : start + 8] = ctx.solid_id
    floor[row, 45] = ctx.block_ids[0]
    floor[row, 50:53] = ctx.block_ids[0]
    for column in range(134, 146, 2):
        _stack(floor, ctx, range(column, column + 1), 4)
    state = _state(floor, config, catalog)
    target = _target(floor, config, catalog, path_smoothness=1.0)
    assert correction_needed(floor, ctx, target, state, "path_smoothness", RAISE, tolerance=0.05)
    edits = plan_correction(
        floor, ctx, target, state, "path_smoothness", RAISE, columns=range(38, 57), level=0
    )
    assert [edit.op for edit in edits] == ["raise_blocks"]
    assert {col for _, col, _ in edits[0].cells} == {45}


def test_removed_pipe_leaves_its_enemy_on_the_ground(floor, config, catalog, ctx) -> None:
    """Lowering pipe density moves no other count: the pipe's rider lands on the floor."""
    band = config.metrics.target_density_ranges.pipe.max
    seed = _target(floor, config, catalog, pipe_density=_density(12, band, ctx))
    state = _state(floor, config, catalog)
    _apply(
        floor,
        plan_correction(
            floor, ctx, seed, state, "pipe_density", RAISE, columns=range(38, 57), level=0
        ),
    )
    (cluster,) = pipe_clusters(floor, ctx)
    top, right = min(row for row, _ in cluster), max(col for _, col in cluster)
    floor[top - 1, right] = ctx.enemy_ids[0]
    state = _state(floor, config, catalog)
    none = _target(floor, config, catalog, pipe_density=0.0)
    _apply(
        floor,
        plan_correction(
            floor, ctx, none, state, "pipe_density", LOWER, columns=range(0, ctx.width)
        ),
    )
    assert not pipe_clusters(floor, ctx)
    surface = floor.shape[0] - ctx.base_height - 1
    assert int(floor[surface, right]) == ctx.enemy_ids[0]
    assert int(np.isin(floor, ctx.enemy_ids).sum()) == 1


def test_raised_ground_carries_its_enemy(floor, ctx) -> None:
    """An enemy standing on a column that rises ends up on the new surface."""
    surface = floor.shape[0] - ctx.base_height - 1
    floor[surface, 45] = ctx.enemy_ids[0]
    apply_edit(floor, Edit(op="raise", cells=tuple(rewrite_stack(floor, ctx, 45, 3))))
    assert ground_heights(floor, ctx)[45] == 3
    assert int(floor[surface - 2, 45]) == ctx.enemy_ids[0]
    assert int(np.isin(floor, ctx.enemy_ids).sum()) == 1


def test_pit_is_dug_where_the_least_content_stands(floor, ctx) -> None:
    """A gap avoids the columns that carry coins when a clear span exists."""
    surface = floor.shape[0] - ctx.base_height - 1
    floor[surface - 2, 38:48] = ctx.coin_id
    edits = plan_dig_gap(floor, ctx, gap_width=3, columns=range(38, 57))
    assert sorted({col for _, col, _ in edits[0].cells}) == [48, 49, 50]
    _apply(floor, edits)
    assert int((floor == ctx.coin_id).sum()) == 10


def test_raw_gap_counts_objects_and_sees_progress_under_a_band(floor, config, catalog, ctx) -> None:
    """The distance to the target is in the metric's own units.

    Three enemies against a target of five are two objects away. Four
    elevated columns measure the same platform usage as none (both sit
    under the band), but they are four columns closer to a mid-band target."""
    surface = floor.shape[0] - ctx.base_height - 1
    for column in (30, 60, 90):
        floor[surface, column] = ctx.enemy_ids[0]
    band = config.metrics.target_density_ranges.enemy.max
    target = _target(floor, config, catalog, enemy_density=_density(5, band, ctx))
    state = _state(floor, config, catalog)
    assert raw_gap(floor, ctx, target, state, "enemy_density") == 2
    target = _target(floor, config, catalog, platform_usage=0.5)
    before = raw_gap(floor, ctx, target, state, "platform_usage")
    floor[surface - 3, 40:44] = ctx.solid_id
    after_state = _state(floor, config, catalog)
    assert after_state.vector[_DIM["platform_usage"]] == state.vector[_DIM["platform_usage"]]
    assert raw_gap(floor, ctx, target, after_state, "platform_usage") == pytest.approx(before - 4)


def test_targets_at_a_band_edge_are_half_lines(floor, config, catalog, ctx) -> None:
    """Past the top of a band nothing needs lowering, and nothing is off.

    More coins than the band can show still measure 1.0: with the target at
    the top there is no distance and no correction either way."""
    surface = floor.shape[0] - ctx.base_height - 1
    band = config.metrics.target_density_ranges.coin.max
    cap = round(band * ctx.area)
    columns = [c for c in range(10, 140) if c not in ctx.protected_columns][: cap + 4]
    for column in columns:
        floor[surface - 2, column] = ctx.coin_id
    state = _state(floor, config, catalog)
    target = _target(floor, config, catalog, coin_density=1.0)
    assert state.vector[_DIM["coin_density"]] == 1.0
    assert raw_gap(floor, ctx, target, state, "coin_density") == 0
    for direction in (LOWER, RAISE):
        assert not correction_needed(
            floor, ctx, target, state, "coin_density", direction, tolerance=0.05
        )
    lower = _target(floor, config, catalog, coin_density=_density(cap - 2, band, ctx))
    assert raw_gap(floor, ctx, lower, state, "coin_density") == 6


def test_cutting_one_of_two_peaks_counts_as_progress(floor, config, catalog, ctx) -> None:
    """Verticality above its target also measures how much path stands too high."""
    _stack(floor, ctx, range(40, 44), 9)
    for offset, stack in enumerate((2, 3, 4, 5, 6, 7, 8)):
        _stack(floor, ctx, range(33 + offset, 34 + offset), stack)
    _stack(floor, ctx, range(100, 104), 9)
    for offset, stack in enumerate((2, 3, 4, 5, 6, 7, 8)):
        _stack(floor, ctx, range(93 + offset, 94 + offset), stack)
    state = _state(floor, config, catalog)
    assert state.amplitude == 8
    target = _target(floor, config, catalog, verticality=0.0)
    before = raw_gap(floor, ctx, target, state, "verticality")
    _apply(
        floor,
        plan_correction(
            floor, ctx, target, state, "verticality", LOWER, columns=range(38, 57), level=0
        ),
    )
    after = _state(floor, config, catalog)
    assert after.amplitude == 8
    assert raw_gap(floor, ctx, target, after, "verticality") < before


@pytest.mark.parametrize(("keep", "left"), [(8, list(range(40, 48))), (6, list(range(40, 46)))])
def test_blocks_are_removed_by_whole_rows(floor, ctx, keep, left) -> None:
    """Surplus blocks go row by row, never thinned out of the middle of one.

    A row of three and a row of eight: keeping eight drops the short row,
    keeping six also trims the long one from its right end."""
    row = floor.shape[0] - ctx.base_height - 4
    floor[row, 20:23] = ctx.block_ids[0]
    floor[row, 40:48] = ctx.block_ids[0]
    edits = plan_remove_blocks(floor, ctx, keep=keep, columns=range(0, ctx.width))
    _apply(floor, edits)
    columns = sorted(np.argwhere(np.isin(floor, ctx.block_ids))[:, 1].tolist())
    assert columns == left


def test_satisfied_level_needs_no_correction(floor, config, catalog, ctx) -> None:
    """With the target equal to the measurement nothing is needed either way."""
    state = _state(floor, config, catalog)
    target = state.vector.astype(np.float32)
    for key in INTENT_KEYS:
        for direction in (LOWER, RAISE):
            assert not correction_needed(floor, ctx, target, state, key, direction, tolerance=0.05)


def test_literal_flat_reads_float32_targets_as_written() -> None:
    """0.05 and 0.9 survive the float32 round trip in the threshold checks."""
    flat = np.array([0.5, 0.4, 0.5, 0.3, 0.4, 0.05, 0.9, 0.55], dtype=np.float32)
    assert is_literal_flat(flat)
    hilly = np.array([0.5, 0.4, 0.5, 0.3, 0.4, 0.06, 0.9, 0.55], dtype=np.float32)
    assert not is_literal_flat(hilly)


def test_gate_accepts_edits_that_do_not_shorten_the_reach(floor, config, catalog, ctx) -> None:
    """A level the flood cannot finish can still be repaired.

    With a pit too wide to jump the flag is out of reach, so the plain gate
    rejects everything; gated against the reach before the edit, a harmless
    change passes and a change that shortens the reach still fails."""
    floor[:, 60:70] = ctx.empty_id
    gate = RefineGate(config, catalog)
    reach = gate.reach(floor)
    assert reach < ctx.flagpole[1]
    row = floor.shape[0] - ctx.base_height - 1
    floor[row - 2, 20] = ctx.coin_id
    assert gate.check(floor) == (False, "flood_unreachable")
    assert gate.check(floor, min_reach=reach) == (True, None)
    floor[:, 30:40] = ctx.empty_id
    assert gate.check(floor, min_reach=reach) == (False, "flood_unreachable")
    floor[row + 1 :, 30:40] = ctx.solid_id
    floor[row + 1 :, 60:70] = ctx.solid_id
    assert gate.reach(floor) == ctx.flagpole[1]
