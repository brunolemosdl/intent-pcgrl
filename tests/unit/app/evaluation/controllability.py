from __future__ import annotations

import numpy as np
import pytest
from pcgrl.app.evaluation.controllability import (
    evaluate_controllability,
    spearman_rho,
    write_controllability_report,
)
from pcgrl.core.domain.level import AreaType, LevelGrid
from pcgrl.runtime.metrics.intent import INTENT_KEYS


def test_spearman_perfect_monotone() -> None:
    """Verify spearman perfect monotone."""
    x = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    assert spearman_rho(x, x * 2.0) == pytest.approx(1.0)
    assert spearman_rho(x, -x) == pytest.approx(-1.0)


def test_spearman_handles_ties_and_constants() -> None:
    """Verify spearman handles ties and constants."""
    x = np.array([1.0, 1.0, 2.0, 3.0])
    y = np.array([1.0, 1.0, 2.0, 3.0])
    assert spearman_rho(x, y) == pytest.approx(1.0)
    assert spearman_rho(x, np.zeros(4)) == 0.0


def _enemy_responsive_generate_fn(config, catalog):
    """Fake generator whose enemy count tracks the requested enemy_density."""
    h, w = config.game.height_tiles, config.game.width_tiles
    empty = catalog.id_for_name("empty")
    solid = catalog.id_for_name("solid")
    goomba = catalog.id_for_name("goomba")
    cap = config.metrics.target_density_ranges.enemy.max * h * w

    def generate(target: np.ndarray, seed: int) -> LevelGrid:
        """Generate."""
        tiles = np.full((h, w), empty, dtype=np.int16)
        tiles[-2:, :] = solid
        count = round(float(target[0]) * cap)
        columns = np.random.default_rng(seed).choice(w, size=min(count, w), replace=False)
        for column in columns:
            tiles[-3, column] = goomba
        return LevelGrid(
            tiles=tiles,
            width_tiles=w,
            height_tiles=h,
            tile_size_px=16,
            area_type=AreaType.OVERWORLD,
            metadata={},
        )

    return generate


def test_evaluate_controllability_detects_responsive_dim(config, catalog, tmp_path) -> None:
    """Verify evaluate controllability detects responsive dim."""
    generate_fn = _enemy_responsive_generate_fn(config, catalog)
    results = evaluate_controllability(
        config, generate_fn, catalog=catalog, dims=("enemy_density",)
    )
    stats = results["enemy_density"]
    assert stats["spearman_rho"] > 0.9
    assert stats["mae"] < 0.25
    expected_n = len(config.evaluation.controllability.sweep_values) * (
        config.evaluation.controllability.levels_per_value
    )
    assert stats["n"] == expected_n
    report = write_controllability_report(results, tmp_path)
    assert report.is_file()


def test_evaluate_controllability_flat_generator_fails_rho(config, catalog) -> None:
    """Verify flat generator fails Spearman rho threshold."""
    h, w = config.game.height_tiles, config.game.width_tiles
    empty = catalog.id_for_name("empty")
    solid = catalog.id_for_name("solid")

    def generate(target: np.ndarray, seed: int) -> LevelGrid:
        """Generate."""
        tiles = np.full((h, w), empty, dtype=np.int16)
        tiles[-2:, :] = solid
        return LevelGrid(
            tiles=tiles,
            width_tiles=w,
            height_tiles=h,
            tile_size_px=16,
            area_type=AreaType.OVERWORLD,
            metadata={},
        )

    results = evaluate_controllability(config, generate, catalog=catalog, dims=("coin_density",))
    assert not results["coin_density"]["passes_rho"]


def test_intent_keys_match_target_fields(config) -> None:
    """Verify intent keys match target fields."""
    assert tuple(config.target_vector_field_names()) == INTENT_KEYS


def test_scripted_edit_takes_an_offered_edit_before_stop() -> None:
    """Random edits never stop while the masks still offer one."""
    from pcgrl.app.evaluation.controllability import _scripted_edit

    mask = np.zeros(12, dtype=bool)
    mask[[3, 7, 11]] = True
    rng = np.random.default_rng(0)
    assert {_scripted_edit(mask, rng, stop=False) for _ in range(40)} == {3, 7}
    assert _scripted_edit(mask, rng, stop=True) == 11
    mask[[3, 7]] = False
    assert _scripted_edit(mask, rng, stop=False) == 11


def test_policy_generator_rejects_unknown_edit_source(config, tmp_path) -> None:
    """The edit source is validated before the checkpoint is opened."""
    from pcgrl.app.evaluation.controllability import composer_policy_generate_fn

    with pytest.raises(ValueError, match="edit source"):
        composer_policy_generate_fn(config, tmp_path / "missing.zip", edits="greedy")


def test_scripted_edit_sources_share_the_composition(config, catalog) -> None:
    """Without the policy's edit choice the composition is the same level.

    Stopping at once returns the composed level; random edits under the
    masks then take its pipes away for a request that asks for none.
    """
    from pcgrl.app.evaluation.controllability import composer_policy_generate_fn
    from pcgrl.core.helpers.path import resolve_path

    checkpoint = resolve_path("artifacts/models/medium_seed42/policy_final.zip")
    library = resolve_path(config.library.output_root) / "chunk_library.npz"
    if not checkpoint.is_file() or not library.is_file():
        pytest.skip("Released policy or chunk library not on disk")
    target = np.array([0.1, 0.4, 0.5, 0.0, 0.4, 0.65, 0.4, 0.55], dtype=np.float32)
    pipe_ids = [catalog.id_for_name(name) for name in config.categories.get("pipes", [])]
    pipes = {}
    for edits in ("none", "random"):
        generate = composer_policy_generate_fn(config, checkpoint, edits=edits)
        level = generate(target, 7)
        assert level.tiles.shape == (config.game.height_tiles, config.game.width_tiles)
        assert np.array_equal(level.tiles, generate(target, 7).tiles)
        pipes[edits] = int(np.isin(level.tiles, pipe_ids).sum())
    assert pipes["random"] < pipes["none"]
