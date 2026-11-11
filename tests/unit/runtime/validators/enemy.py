from __future__ import annotations

import numpy as np
from pcgrl.core.domain.level import AreaType, LevelGrid
from pcgrl.runtime.validators.enemy import EnemyValidator


def test_no_enemies_passes(config, catalog, blank_level) -> None:
    """Verify no enemies passes."""
    result = EnemyValidator(config, catalog).validate(blank_level)
    assert result.is_valid
    assert result.penalty == 0.0


def test_grounded_enemy_passes(config, catalog) -> None:
    """Verify grounded enemy passes."""
    h, w = (config.game.height_tiles, config.game.width_tiles)
    solid_id = catalog.id_for_name("solid")
    goomba_id = catalog.id_for_name("goomba")
    empty_id = catalog.id_for_name("empty")
    tiles = np.full((h, w), empty_id, dtype=np.int16)
    tiles[-1, :] = solid_id
    tiles[-2, 5] = goomba_id
    level = LevelGrid(
        tiles=tiles, width_tiles=w, height_tiles=h, tile_size_px=16, area_type=AreaType.OVERWORLD
    )
    result = EnemyValidator(config, catalog).validate(level)
    assert result.is_valid


def test_spawn_above_ground_passes(config, catalog) -> None:
    """An enemy with empty cells below but ground in the column is a valid
    spawn (the game drops it), matching the raw corpus placement style."""
    h, w = (config.game.height_tiles, config.game.width_tiles)
    solid_id = catalog.id_for_name("solid")
    goomba_id = catalog.id_for_name("goomba")
    empty_id = catalog.id_for_name("empty")
    tiles = np.full((h, w), empty_id, dtype=np.int16)
    tiles[-1, :] = solid_id
    tiles[5, 10] = goomba_id
    level = LevelGrid(
        tiles=tiles, width_tiles=w, height_tiles=h, tile_size_px=16, area_type=AreaType.OVERWORLD
    )
    result = EnemyValidator(config, catalog).validate(level)
    assert result.is_valid
    assert not result.violations


def test_library_chunks_never_hard_fail_enemy_rule(config, catalog) -> None:
    """Every corpus-extracted chunk must pass the enemy validator — the
    corpus is the calibration target, so a rule stricter than the corpus
    rejects faithful compositions at random."""
    import pytest
    from pcgrl.core.helpers.path import resolve_path
    from pcgrl.core.library.types import ChunkLibrary

    library_root = resolve_path(config.library.output_root)
    if not (library_root / "chunk_library.npz").is_file():
        pytest.skip("Chunk library not built")
    library = ChunkLibrary.load(library_root)
    validator = EnemyValidator(config, catalog)
    for chunk in library.chunks:
        h, w = chunk.tiles.shape
        level = LevelGrid(
            tiles=chunk.tiles.astype(np.int16),
            width_tiles=w,
            height_tiles=h,
            tile_size_px=16,
            area_type=AreaType.OVERWORLD,
        )
        assert validator.validate(level).is_valid


def test_orphan_enemy_is_soft_violation(config, catalog) -> None:
    """An enemy with no ground anywhere below penalises but never invalidates
    (the raw corpus contains such spawns over pits)."""
    h, w = (config.game.height_tiles, config.game.width_tiles)
    goomba_id = catalog.id_for_name("goomba")
    empty_id = catalog.id_for_name("empty")
    tiles = np.full((h, w), empty_id, dtype=np.int16)
    tiles[5, 10] = goomba_id
    level = LevelGrid(
        tiles=tiles, width_tiles=w, height_tiles=h, tile_size_px=16, area_type=AreaType.OVERWORLD
    )
    result = EnemyValidator(config, catalog).validate(level)
    assert result.is_valid
    assert any(v.rule == "enemy.orphaned" for v in result.violations)
    assert all(v.severity == "soft" for v in result.violations)
    assert result.penalty > 0.0
