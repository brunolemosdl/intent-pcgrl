from __future__ import annotations

import numpy as np
from pcgrl.core.domain.level import LevelGrid
from pcgrl.runtime.validators.glyphs import GlyphsValidator


def test_valid_level_passes(config, catalog, blank_level) -> None:
    """Verify valid level passes."""
    result = GlyphsValidator(catalog).validate(blank_level)
    assert result.is_valid
    assert result.penalty == 0.0


def test_unknown_tile_id_fails(config, catalog) -> None:
    """Verify unknown tile id fails."""
    h, w = (config.game.height_tiles, config.game.width_tiles)
    tiles = np.zeros((h, w), dtype=np.int16)
    tiles[0, 0] = 9999
    level = LevelGrid(tiles=tiles, width_tiles=w, height_tiles=h, tile_size_px=16)
    result = GlyphsValidator(catalog).validate(level)
    assert not result.is_valid
    assert result.penalty == 1.0
    assert any(v.rule == "glyphs.unknown_id" for v in result.violations)


def test_negative_tile_id_fails(config, catalog) -> None:
    """Verify negative tile id fails."""
    h, w = (config.game.height_tiles, config.game.width_tiles)
    tiles = np.zeros((h, w), dtype=np.int16)
    tiles[0, 0] = -1
    level = LevelGrid(tiles=tiles, width_tiles=w, height_tiles=h, tile_size_px=16)
    result = GlyphsValidator(catalog).validate(level)
    assert not result.is_valid


def test_violation_severity_hard(config, catalog) -> None:
    """Verify violation severity hard."""
    h, w = (config.game.height_tiles, config.game.width_tiles)
    tiles = np.zeros((h, w), dtype=np.int16)
    tiles[0, 0] = 9999
    level = LevelGrid(tiles=tiles, width_tiles=w, height_tiles=h, tile_size_px=16)
    result = GlyphsValidator(catalog).validate(level)
    assert all(v.severity == "hard" for v in result.violations)
