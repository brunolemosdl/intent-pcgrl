from __future__ import annotations

import numpy as np
from pcgrl.core.domain.level import AreaType, LevelGrid
from pcgrl.rendering.ascii import level_to_ascii


def test_level_to_ascii_returns_string(config, catalog, blank_level) -> None:
    """Verify level to ascii returns string."""
    result = level_to_ascii(blank_level, catalog)
    assert isinstance(result, str)


def test_level_to_ascii_line_count(config, catalog, blank_level) -> None:
    """Verify level to ascii line count."""
    result = level_to_ascii(blank_level, catalog)
    lines = result.splitlines()
    assert len(lines) == config.game.height_tiles


def test_level_to_ascii_line_width(config, catalog, blank_level) -> None:
    """Verify level to ascii line width."""
    result = level_to_ascii(blank_level, catalog)
    for line in result.splitlines():
        assert len(line) == config.game.width_tiles


def test_level_to_ascii_all_empty(config, catalog, blank_level) -> None:
    """Verify level to ascii all empty."""
    result = level_to_ascii(blank_level, catalog)
    assert all(c == "-" for c in result.replace("\n", ""))


def test_level_to_ascii_solid_row(config, catalog) -> None:
    """Verify level to ascii solid row."""
    h, w = (config.game.height_tiles, config.game.width_tiles)
    solid_id = catalog.id_for_name("solid")
    empty_id = catalog.id_for_name("empty")
    tiles = np.full((h, w), empty_id, dtype=np.int16)
    tiles[-1, :] = solid_id
    level = LevelGrid(
        tiles=tiles, width_tiles=w, height_tiles=h, tile_size_px=16, area_type=AreaType.OVERWORLD
    )
    result = level_to_ascii(level, catalog)
    last_line = result.splitlines()[-1]
    assert all(c == "X" for c in last_line)


def test_level_to_ascii_matches_to_ascii(config, catalog, blank_level) -> None:
    """Verify level to ascii matches to ascii."""
    assert level_to_ascii(blank_level, catalog) == blank_level.to_ascii(catalog)
