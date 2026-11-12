from __future__ import annotations

import numpy as np
from pcgrl.core.domain.level import LevelGrid
from pcgrl.runtime.validators.geometry import GeometryValidator


def test_passes_matching_shape(config, catalog, blank_level) -> None:
    """Verify passes matching shape."""
    result = GeometryValidator(config).validate(blank_level)
    assert result.is_valid
    assert result.penalty == 0.0


def test_fails_wrong_height(config, catalog) -> None:
    """Verify fails wrong height."""
    tiles = np.zeros((8, config.game.width_tiles), dtype=np.int16)
    level = LevelGrid(
        tiles=tiles, width_tiles=config.game.width_tiles, height_tiles=8, tile_size_px=16
    )
    result = GeometryValidator(config).validate(level)
    assert not result.is_valid
    assert result.penalty > 0.0


def test_fails_wrong_width(config, catalog) -> None:
    """Verify fails wrong width."""
    tiles = np.zeros((config.game.height_tiles, 8), dtype=np.int16)
    level = LevelGrid(
        tiles=tiles, width_tiles=8, height_tiles=config.game.height_tiles, tile_size_px=16
    )
    result = GeometryValidator(config).validate(level)
    assert not result.is_valid


def test_violation_rule_name(config, catalog) -> None:
    """Verify violation rule name."""
    tiles = np.zeros((4, 4), dtype=np.int16)
    level = LevelGrid(tiles=tiles, width_tiles=4, height_tiles=4, tile_size_px=16)
    result = GeometryValidator(config).validate(level)
    assert not result.is_valid
    assert any(v.rule == "geometry.shape" for v in result.violations)


def test_violation_severity_hard(config, catalog) -> None:
    """Verify violation severity hard."""
    tiles = np.zeros((4, 4), dtype=np.int16)
    level = LevelGrid(tiles=tiles, width_tiles=4, height_tiles=4, tile_size_px=16)
    result = GeometryValidator(config).validate(level)
    assert all(v.severity == "hard" for v in result.violations)
