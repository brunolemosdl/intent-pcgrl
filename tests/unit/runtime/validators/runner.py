from __future__ import annotations

import numpy as np
from pcgrl.core.domain.level import AreaType, LevelGrid
from pcgrl.runtime.validators.runner import ValidatorRunner


def test_runner_blank_level_aggregates(config, catalog, blank_level) -> None:
    """Verify runner blank level aggregates."""
    runner = ValidatorRunner(config, catalog)
    result = runner.validate(blank_level)
    assert isinstance(result.violations, tuple)
    assert not result.is_valid


def test_runner_short_circuit_stops_early(config, catalog) -> None:
    """Verify runner short circuit stops early."""
    tiles = np.zeros((4, 4), dtype=np.int16)
    level = LevelGrid(tiles=tiles, width_tiles=4, height_tiles=4, tile_size_px=16)
    runner = ValidatorRunner(config, catalog)
    result_full = runner.validate(level, short_circuit_hard=False)
    result_sc = runner.validate(level, short_circuit_hard=True)
    assert len(result_sc.violations) <= len(result_full.violations)


def test_runner_penalty_accumulates(config, catalog, blank_level) -> None:
    """Verify runner penalty accumulates."""
    runner = ValidatorRunner(config, catalog)
    result = runner.validate(blank_level)
    assert result.penalty > 0.0


def test_runner_valid_level(config, catalog) -> None:
    """Verify runner valid level."""
    h, w = (config.game.height_tiles, config.game.width_tiles)
    solid_id = catalog.id_for_name("solid")
    empty_id = catalog.id_for_name("empty")
    player_id = catalog.id_for_name("player")
    flag_id = catalog.id_for_name("flagpole")
    tiles = np.full((h, w), empty_id, dtype=np.int16)
    tiles[-1, :] = solid_id
    tiles[h - 2, 1] = player_id
    tiles[h - 2, w - 2] = flag_id
    level = LevelGrid(
        tiles=tiles, width_tiles=w, height_tiles=h, tile_size_px=16, area_type=AreaType.OVERWORLD
    )
    runner = ValidatorRunner(config, catalog)
    result = runner.validate(level)
    assert result.is_valid
