from __future__ import annotations

import numpy as np
from pcgrl.core.domain.level import AreaType, LevelGrid
from pcgrl.runtime.validators.density import DensityValidator


def test_empty_level_below_min_solid(config, catalog, blank_level) -> None:
    """Verify empty level below min solid."""
    result = DensityValidator(config, catalog).validate(blank_level)
    assert result.is_valid
    assert result.penalty > 0.0


def test_all_solid_level_above_max(config, catalog) -> None:
    """Verify all solid level above max."""
    h, w = (config.game.height_tiles, config.game.width_tiles)
    solid_id = catalog.id_for_name("solid")
    tiles = np.full((h, w), solid_id, dtype=np.int16)
    level = LevelGrid(
        tiles=tiles, width_tiles=w, height_tiles=h, tile_size_px=16, area_type=AreaType.OVERWORLD
    )
    result = DensityValidator(config, catalog).validate(level)
    assert result.is_valid
    assert result.penalty > 0.0
    assert any(v.rule == "density.solid_ratio" for v in result.violations)


def test_valid_density_passes(config, catalog) -> None:
    """Verify valid density passes."""
    h, w = (config.game.height_tiles, config.game.width_tiles)
    solid_id = catalog.id_for_name("solid")
    empty_id = catalog.id_for_name("empty")
    tiles = np.full((h, w), empty_id, dtype=np.int16)
    tiles[-2:, :] = solid_id
    level = LevelGrid(
        tiles=tiles, width_tiles=w, height_tiles=h, tile_size_px=16, area_type=AreaType.OVERWORLD
    )
    result = DensityValidator(config, catalog).validate(level)
    assert result.is_valid
    assert result.penalty == 0.0


def test_soft_violation_severity(config, catalog, blank_level) -> None:
    """Verify soft violation severity."""
    result = DensityValidator(config, catalog).validate(blank_level)
    assert all(v.severity == "soft" for v in result.violations)
