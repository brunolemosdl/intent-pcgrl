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


@pytest.mark.parametrize("name", sorted(_CORNERS))
def test_refiner_converges_on_corner_targets(composed_level, config, catalog, name) -> None:
    """Every benchmark corner ends within tolerance on all 8 dimensions.

    These four targets span the regions the chunk material alone cannot
    reach (exact zeros, max gaps with platforms, the tall+smooth combo, and
    the corpus-typical mid point)."""
    refiner = IntentRefiner(config, catalog)
    target = np.array(_CORNERS[name], dtype=np.float32)
    result = refiner.refine(composed_level, target)
    errors = {
        key: abs(after - want)
        for key, after, want in zip(
            INTENT_KEYS, result.measured_after, target.tolist(), strict=True
        )
    }
    assert result.converged, errors


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
        lambda self, tiles, baseline=frozenset(): (False, "forced"),
    )
    refiner = IntentRefiner(config, catalog, max_passes=1)
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
    from pcgrl.runtime.refine.context import build_context

    ctx = build_context(config, catalog)
    height, width = config.game.height_tiles, config.game.width_tiles
    tiles = np.full((height, width), ctx.empty_id, dtype=np.int16)
    tiles[height - 2 :, :] = ctx.solid_id
    tiles[height - 3, 50:56] = ctx.solid_id
    tiles[height - 6, 80:84] = ctx.solid_id
    tiles[ctx.player] = catalog.id_for_name("player")
    tiles[ctx.flagpole] = catalog.id_for_name("flagpole")
    from pcgrl.core.domain.level import AreaType, LevelGrid

    level = LevelGrid(
        tiles=tiles,
        width_tiles=width,
        height_tiles=height,
        tile_size_px=16,
        area_type=AreaType.OVERWORLD,
    )
    target = np.array(_CORNERS["user_flat_empty"], dtype=np.float32)
    refiner = IntentRefiner(config, catalog)
    before = refiner.refine(level, target).measured_before
    assert abs(before[INTENT_KEYS.index("verticality")]) <= 0.05, (
        "fixture must start sub-band, otherwise it does not exercise the cleanup"
    )
    result = refiner.refine(level, target)
    refined = result.level.tiles
    above_ground = refined[: height - 2, :]
    visible = np.isin(above_ground, list(ctx.solid_like | set(ctx.pipe_ids)))
    assert not visible.any(), "sub-band mound or platform survived the cleanup"
    assert result.converged


def _flat_level(config, catalog, enemies: tuple[int, ...] = ()):
    """Flat ground from spawn to flag, with goombas at the given columns."""
    from pcgrl.core.domain.level import AreaType, LevelGrid
    from pcgrl.runtime.refine.context import build_context

    ctx = build_context(config, catalog)
    height, width = config.game.height_tiles, config.game.width_tiles
    tiles = np.full((height, width), ctx.empty_id, dtype=np.int16)
    tiles[height - 2 :, :] = ctx.solid_id
    for column in enemies:
        tiles[height - 3, column] = ctx.enemy_ids[0]
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

    Pipe repairs are structural fixes kept regardless of the error, so runs
    that needed one are not compared."""
    from pcgrl.runtime.refine import intent_objective

    rng = np.random.default_rng(seed)
    target = rng.random(8).astype(np.float32)
    refiner = IntentRefiner(config, catalog)
    result = refiner.refine(composed_level, target)
    if any(e.dim == "repair" and e.accepted for e in result.edits):
        pytest.skip("a pipe repair was required")
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
    height = config.game.height_tiles
    result = IntentRefiner(config, catalog).refine(level, target)
    above_ground = np.isin(result.level.tiles[: height - 2, :], list(ctx.solid_like))
    assert not above_ground.any()
    assert not any(e.op == "raise_hill" and e.accepted for e in result.edits)


def test_literal_flat_request_settles_on_the_corpus_floor(config, catalog) -> None:
    """A flat, empty, smooth request ends on the one-row floor of the corpus."""
    from pcgrl.runtime.refine.ops import ground_heights

    level, ctx = _flat_level(config, catalog)
    target = np.array(_CORNERS["user_flat_empty"], dtype=np.float32)
    result = IntentRefiner(config, catalog).refine(level, target)
    heights = ground_heights(result.level.tiles.astype(np.int16), ctx)
    settled = {h for column, h in enumerate(heights) if column not in ctx.protected_columns}
    assert ctx.base_height == 1
    assert settled == {ctx.base_height}
    assert result.converged


def test_clipped_pipe_the_path_climbs_on_becomes_ground(config, catalog) -> None:
    """A half pipe that is the only step up a wall is repaired without removing it."""
    from pcgrl.runtime.refine.gates import RefineGate

    level, ctx = _flat_level(config, catalog)
    tiles = level.tiles.astype(np.int16).copy()
    floor = tiles.shape[0] - 2
    tiles[floor - 8 : floor, 60:65] = ctx.solid_id
    tiles[floor - 4, 59] = ctx.pipe_top_left
    tiles[floor - 3 : floor, 59] = ctx.pipe_body_left
    walled = level.copy_with_tiles(tiles)
    gate = RefineGate(config, catalog)
    assert "pipe.width" in gate.hard_rules(tiles)
    target = _measured(walled, config, catalog)
    result = IntentRefiner(config, catalog).refine(walled, target)
    refined = result.level.tiles.astype(np.int16)
    assert "pipe.width" not in gate.hard_rules(refined)
    assert (refined[floor - 4 : floor, 59] == ctx.solid_id).all()
    ok, reason = gate.check(refined)
    assert ok, reason


def test_effective_target_reads_none_literally(config, catalog) -> None:
    """Only count dimensions at or below 0.05 are zeroed."""
    target = np.array([0.05, 0.04, 0.06, 0.05, 0.05, 0.05, 0.95, 0.05], dtype=np.float32)
    effective = IntentRefiner(config, catalog).effective_target(target)
    assert effective[:6].tolist() == pytest.approx([0.0, 0.0, 0.06, 0.0, 0.0, 0.05])
    assert effective[6:].tolist() == pytest.approx([0.95, 0.05])
