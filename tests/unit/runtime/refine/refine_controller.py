from __future__ import annotations

import numpy as np
import pytest
from pcgrl.core.helpers.path import resolve_path
from pcgrl.runtime.metrics.intent import INTENT_KEYS
from pcgrl.runtime.refine import IntentRefiner

_CORNERS = {
    "user_flat_empty": [0, 0, 0, 0, 0, 0, 1.0, 0],
    "gaps_platforms": [0, 0, 0, 0, 1.0, 0, 0.6, 1.0],
    "tall_smooth": [0.2, 0.3, 0.2, 0.1, 0.2, 1.0, 1.0, 0.2],
    "corpus_mid": [0.5, 0.4, 0.5, 0.3, 0.4, 0.65, 0.40, 0.55],
}


@pytest.fixture
def composed_level(config, catalog):
    """One random-legal composition (skipped when the library is absent)."""
    library_root = resolve_path(config.library.output_root)
    if not (library_root / "chunk_library.npz").is_file():
        pytest.skip("Chunk library not built")
    from pcgrl.core.library.types import ChunkLibrary
    from pcgrl.runtime.environment.composer_policies import rollout_baseline
    from pcgrl.runtime.environment.composer_state import ComposerCore

    library = ChunkLibrary.load(library_root)
    core = ComposerCore(library, config, catalog)
    rng = np.random.default_rng(11)
    target = np.full(8, 0.5, dtype=np.float32)
    return core.stitch_level(rollout_baseline("random_legal", core, target, rng))


def test_touch_up_closes_count_errors_within_the_tile_budget(config, catalog) -> None:
    """Small count errors are fixed with a handful of tile writes.

    The level matches its target except for three surplus enemies and two
    missing coins: the kind of residue the edit phase leaves."""
    level, ctx = _flat_level(config, catalog, enemies=(40, 70, 100))
    target = _measured(level, config, catalog)
    target[INTENT_KEYS.index("enemy_density")] = 0.0
    target[INTENT_KEYS.index("coin_density")] = 2 / (
        config.metrics.target_density_ranges.coin.max * ctx.area
    )
    result = IntentRefiner(config, catalog, max_tiles=32).refine(level, target)
    assert result.converged
    assert 0 < result.tiles_changed <= 32
    assert int(np.isin(result.level.tiles, ctx.enemy_ids).sum()) == 0
    assert int((result.level.tiles == ctx.coin_id).sum()) == 2


def test_touch_up_spends_a_short_budget_on_what_fits(config, catalog) -> None:
    """A budget smaller than the residue is spent tile by tile, not skipped.

    Six coins are missing and four tile writes are allowed: four coins land
    and the edit that asked for more is cut to the budget."""
    level, ctx = _flat_level(config, catalog)
    target = _measured(level, config, catalog)
    target[INTENT_KEYS.index("coin_density")] = 6 / (
        config.metrics.target_density_ranges.coin.max * ctx.area
    )
    result = IntentRefiner(config, catalog, max_tiles=4).refine(level, target)
    assert int((result.level.tiles == ctx.coin_id).sum()) == 4
    assert result.tiles_changed == 4


def test_touch_up_takes_the_cheapest_effective_edit_first(config, catalog) -> None:
    """With room for one fix only, the one that buys more error per tile wins.

    The level is one enemy over its target and one pipe short. Removing the
    enemy costs one tile, the short pipe four: a budget of three can afford
    only the enemy, a budget of five fixes both."""
    level, ctx = _flat_level(config, catalog, enemies=(40,))
    target = _measured(level, config, catalog)
    target[INTENT_KEYS.index("enemy_density")] = 0.0
    target[INTENT_KEYS.index("pipe_density")] = 6 / (
        config.metrics.target_density_ranges.pipe.max * ctx.area
    )
    short = IntentRefiner(config, catalog, max_tiles=3).refine(level, target)
    assert int(np.isin(short.level.tiles, ctx.enemy_ids).sum()) == 0
    assert not np.isin(short.level.tiles, list(ctx.pipe_ids)).any()
    assert short.tiles_changed == 1
    enough = IntentRefiner(config, catalog, max_tiles=5).refine(level, target)
    assert int(np.isin(enough.level.tiles, ctx.enemy_ids).sum()) == 0
    assert int(np.isin(enough.level.tiles, list(ctx.pipe_ids)).sum()) == 4
    assert enough.tiles_changed == 5
    assert enough.converged


def test_touch_up_raises_the_summit_instead_of_a_new_hill(config, catalog) -> None:
    """One row of missing amplitude costs one tile on the existing summit."""
    from pcgrl.runtime.refine.context import target_amplitude

    level, ctx = _flat_level(config, catalog)
    tiles = level.tiles.copy()
    height = tiles.shape[0]
    for offset, stack in enumerate((2, 3, 4, 5, 6, 7, 8, 9, 9)):
        tiles[height - stack :, 100 + offset] = ctx.solid_id
    level = level.copy_with_tiles(tiles)
    target = _measured(level, config, catalog)
    target[INTENT_KEYS.index("verticality")] = 0.667
    assert target_amplitude(0.667, config) == 9
    result = IntentRefiner(config, catalog, max_tiles=32).refine(level, target)
    accepted = [e.op for e in result.edits if e.accepted]
    assert accepted == ["raise_summit"]
    assert result.tiles_changed == 1
    assert result.converged


@pytest.mark.parametrize("budget", [8, 32])
@pytest.mark.parametrize("name", sorted(_CORNERS))
def test_touch_up_never_exceeds_the_tile_budget(
    composed_level, config, catalog, name, budget
) -> None:
    """However far the target is, the touch-up keeps at most ``max_tiles`` writes.

    A random composition is far from every corner; reshaping it is the edit
    phase's job, so the bounded touch-up must stop instead of rebuilding the
    level (pipe repairs are structural fixes and are not counted)."""
    refiner = IntentRefiner(config, catalog, max_tiles=budget)
    result = refiner.refine(composed_level, np.array(_CORNERS[name], dtype=np.float32))
    kept = sum(e.cell_count for e in result.edits if e.accepted and e.dim != "repair")
    assert kept <= budget
    assert not any(e.accepted and e.cell_count > budget for e in result.edits if e.dim != "repair")


def test_refiner_is_deterministic(composed_level, config, catalog) -> None:
    """Same level + target produce bit-identical tiles and edit logs."""
    refiner = IntentRefiner(config, catalog)
    target = np.array(_CORNERS["user_flat_empty"], dtype=np.float32)
    first = refiner.refine(composed_level, target)
    second = refiner.refine(composed_level, target)
    assert np.array_equal(first.level.tiles, second.level.tiles)
    assert first.edits == second.edits


def test_refiner_never_breaks_gates(composed_level, config, catalog) -> None:
    """The refined level passes the structural chain and reaches the flag."""
    from pcgrl.runtime.refine.gates import RefineGate

    refiner = IntentRefiner(config, catalog)
    gate = RefineGate(config, catalog)
    baseline = gate.hard_rules(composed_level.tiles)
    for name, raw in _CORNERS.items():
        result = refiner.refine(composed_level, np.array(raw, dtype=np.float32))
        ok, reason = gate.check(result.level.tiles.copy(), baseline=baseline)
        assert ok, (name, reason)


def test_rejected_edits_are_reverted(composed_level, config, catalog, monkeypatch) -> None:
    """With every gate rejecting, no gated edit lands and writes are undone.

    Coin ops are gateless (inert tiles) and may still apply; everything else
    must report ``accepted=False`` and leave its cells reverted."""
    from pcgrl.runtime.refine import gates

    monkeypatch.setattr(
        gates.RefineGate,
        "check",
        lambda self, tiles, baseline=frozenset(), min_reach=None: (False, "forced"),
    )
    refiner = IntentRefiner(config, catalog)
    target = np.array(_CORNERS["user_flat_empty"], dtype=np.float32)
    result = refiner.refine(composed_level, target)
    gated = [e for e in result.edits if e.op not in ("remove_coins", "add_coins")]
    assert gated
    assert all(not e.accepted for e in gated)
    coin_id = catalog.id_for_name("coin")
    coin_involved = (composed_level.tiles == coin_id) | (result.level.tiles == coin_id)
    assert np.array_equal(composed_level.tiles[~coin_involved], result.level.tiles[~coin_involved])


def test_literal_flat_strips_sub_band_structure(config, catalog) -> None:
    """A literal-flat request removes structure the bands cannot see.

    Mounds and floating platforms below the verticality band's 5th
    percentile measure as zero, so the error-driven loop skips them; the
    target-driven cleanup must strip them anyway: a converged policy
    delivers banded zeros itself, leaving the polish nothing to react to
    while the level still shows relief."""
    level, ctx = _flat_level(config, catalog)
    height = config.game.height_tiles
    floor = height - ctx.base_height
    tiles = level.tiles.copy()
    tiles[floor - 1, 50:56] = ctx.solid_id
    tiles[floor - 4, 80:84] = ctx.solid_id
    level = level.copy_with_tiles(tiles)
    target = np.array(_CORNERS["user_flat_empty"], dtype=np.float32)
    refiner = IntentRefiner(config, catalog)
    result = refiner.refine(level, target)
    assert abs(result.measured_before[INTENT_KEYS.index("verticality")]) <= 0.05, (
        "fixture must start sub-band, otherwise it does not exercise the cleanup"
    )
    above_ground = result.level.tiles[:floor, :]
    visible = np.isin(above_ground, list(ctx.solid_like | set(ctx.pipe_ids)))
    assert not visible.any(), "sub-band mound or platform survived the cleanup"
    assert result.converged
    assert result.tiles_changed == 10


def _flat_level(config, catalog, enemies: tuple[int, ...] = ()):
    """Flat corpus-style floor from spawn to flag, with goombas at the given columns."""
    from pcgrl.core.domain.level import AreaType, LevelGrid
    from pcgrl.runtime.refine.context import build_context

    ctx = build_context(config, catalog)
    height, width = config.game.height_tiles, config.game.width_tiles
    tiles = np.full((height, width), ctx.empty_id, dtype=np.int16)
    floor = height - ctx.base_height
    tiles[floor:, :] = ctx.solid_id
    for column in enemies:
        tiles[floor - 1, column] = ctx.enemy_ids[0]
    tiles[ctx.player] = catalog.id_for_name("player")
    tiles[ctx.flagpole] = catalog.id_for_name("flagpole")
    level = LevelGrid(
        tiles=tiles,
        width_tiles=width,
        height_tiles=height,
        tile_size_px=16,
        area_type=AreaType.OVERWORLD,
    )
    return level, ctx


def _measured(level, config, catalog) -> np.ndarray:
    from pcgrl.runtime.metrics.intent import compute_intent_vector

    return np.asarray(compute_intent_vector(level, catalog, config), dtype=np.float32)


@pytest.mark.parametrize("seed", range(6))
def test_refiner_never_returns_a_worse_level(composed_level, config, catalog, seed) -> None:
    """Polish never raises the overall intent error of its input.

    Pipe repairs are structural fixes kept regardless of the error, so the
    comparison starts from the level with its seam-clipped pipes removed."""
    from pcgrl.runtime.refine import intent_objective
    from pcgrl.runtime.refine.context import build_context
    from pcgrl.runtime.refine.ops import apply_edit, plan_repair_pipes

    tiles = composed_level.tiles.astype(np.int16).copy()
    for edit in plan_repair_pipes(tiles, build_context(config, catalog)):
        apply_edit(tiles, edit)
    level = composed_level.copy_with_tiles(tiles)
    rng = np.random.default_rng(seed)
    target = rng.random(8).astype(np.float32)
    refiner = IntentRefiner(config, catalog)
    result = refiner.refine(level, target)
    effective = refiner.effective_target(target)
    before = intent_objective(np.asarray(result.measured_before), effective)
    after = intent_objective(np.asarray(result.measured_after), effective)
    assert after <= before + 1e-6


def test_none_request_removes_every_enemy(config, catalog) -> None:
    """A "no enemies" target (0.05, as the language bridge writes it) means zero.

    0.05 of the enemy cap rounds to one goomba, which also measures inside
    the generic tolerance, so "none" must be read literally."""
    level, ctx = _flat_level(config, catalog, enemies=(40, 70, 100))
    target = _measured(level, config, catalog)
    target[INTENT_KEYS.index("enemy_density")] = 0.05
    result = IntentRefiner(config, catalog).refine(level, target)
    assert int(np.isin(result.level.tiles, ctx.enemy_ids).sum()) == 0


def test_flat_request_keeps_a_flat_level_flat(config, catalog) -> None:
    """A flat target written as verticality 0.05 adds no relief to flat ground.

    In float32 0.05 is 0.0500000007, just above the tolerance against a
    measured 0.0, which must not read as a request for a hill."""
    level, ctx = _flat_level(config, catalog)
    target = _measured(level, config, catalog)
    target[INTENT_KEYS.index("verticality")] = 0.05
    target[INTENT_KEYS.index("path_smoothness")] = 0.95
    floor = config.game.height_tiles - ctx.base_height
    result = IntentRefiner(config, catalog).refine(level, target)
    above_ground = np.isin(result.level.tiles[:floor, :], list(ctx.solid_like))
    assert not above_ground.any()
    assert not any(e.op in ("raise_hill", "raise_summit") and e.accepted for e in result.edits)


def test_effective_target_reads_none_literally(config, catalog) -> None:
    """Only count dimensions at or below 0.05 are zeroed."""
    target = np.array([0.05, 0.04, 0.06, 0.05, 0.05, 0.05, 0.95, 0.05], dtype=np.float32)
    effective = IntentRefiner(config, catalog).effective_target(target)
    assert effective[:6].tolist() == pytest.approx([0.0, 0.0, 0.06, 0.0, 0.0, 0.05])
    assert effective[6:].tolist() == pytest.approx([0.95, 0.05])
