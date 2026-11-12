from __future__ import annotations

import numpy as np
import pytest
from pcgrl.core.domain.level import AreaType, LevelGrid
from pcgrl.core.tiles.catalog import TileCatalog


def test_area_type_values() -> None:
    """Verify area type values."""
    assert AreaType.OVERWORLD.value == "overworld"
    assert list(AreaType) == [AreaType.OVERWORLD]


def test_level_grid_dimensions(catalog: TileCatalog) -> None:
    """Verify level grid dimensions."""
    tiles = np.zeros((4, 8), dtype=np.int16)
    level = LevelGrid(
        tiles=tiles, width_tiles=8, height_tiles=4, tile_size_px=16, area_type=AreaType.OVERWORLD
    )
    assert level.width_tiles == 8
    assert level.height_tiles == 4


def test_to_ascii_renders_glyphs(catalog: TileCatalog) -> None:
    """Verify to ascii renders glyphs."""
    empty_id = catalog.id_for_name("empty")
    solid_id = catalog.id_for_name("solid")
    tiles = np.full((2, 4), empty_id, dtype=np.int16)
    tiles[1, :] = solid_id
    level = LevelGrid(
        tiles=tiles, width_tiles=4, height_tiles=2, tile_size_px=16, area_type=AreaType.OVERWORLD
    )
    text = level.to_ascii(catalog)
    lines = text.splitlines()
    assert lines[0] == "----"
    assert lines[1] == "XXXX"


def test_from_ascii_roundtrip(catalog: TileCatalog) -> None:
    """Verify from ascii roundtrip."""
    ascii_text = "----\nXXXX"
    level = LevelGrid.from_ascii(ascii_text, catalog, tile_size_px=16, area_type=AreaType.OVERWORLD)
    assert level.width_tiles == 4
    assert level.height_tiles == 2
    result = level.to_ascii(catalog)
    assert result == ascii_text


def test_from_ascii_roundtrip_flagpole(config) -> None:
    """Verify from ascii roundtrip flagpole."""
    from pcgrl.core.settings.loader import load_config
    from pcgrl.core.tiles.catalog import load_catalog

    catalog = load_catalog(load_config(training="initial"))
    flag_id = catalog.id_for_name("flagpole")
    empty_id = catalog.id_for_name("empty")
    solid_id = catalog.id_for_name("solid")
    tiles = np.full((3, 4), empty_id, dtype=np.int16)
    tiles[2, :] = solid_id
    tiles[0, 2] = flag_id
    level = LevelGrid(
        tiles=tiles,
        width_tiles=4,
        height_tiles=3,
        tile_size_px=16,
        area_type=AreaType.OVERWORLD,
    )
    ascii_text = level.to_ascii(catalog)
    restored = LevelGrid.from_ascii(
        ascii_text,
        catalog,
        tile_size_px=16,
        area_type=AreaType.OVERWORLD,
    )
    assert np.array_equal(restored.tiles, level.tiles)
    assert "F" in ascii_text


def test_from_ascii_empty_raises() -> None:
    """Verify from ascii empty raises."""

    with pytest.raises(ValueError, match="empty"):
        from pcgrl.core.settings.loader import load_config
        from pcgrl.core.tiles.catalog import load_catalog

        cfg = load_config(training="initial")
        cat = load_catalog(cfg)
        LevelGrid.from_ascii("", cat)


def test_copy_with_tiles(catalog: TileCatalog) -> None:
    """Verify copy with tiles."""
    tiles = np.zeros((4, 8), dtype=np.int16)
    level = LevelGrid(
        tiles=tiles, width_tiles=8, height_tiles=4, tile_size_px=16, area_type=AreaType.OVERWORLD
    )
    new_tiles = np.ones((4, 8), dtype=np.int16)
    copy = level.copy_with_tiles(new_tiles)
    assert copy.width_tiles == 8
    assert copy.height_tiles == 4
    assert np.all(copy.tiles == 1)
    assert np.all(level.tiles == 0)


def test_copy_with_tiles_is_independent(catalog: TileCatalog) -> None:
    """Verify copy with tiles is independent."""
    tiles = np.zeros((4, 8), dtype=np.int16)
    level = LevelGrid(
        tiles=tiles, width_tiles=8, height_tiles=4, tile_size_px=16, area_type=AreaType.OVERWORLD
    )
    new_tiles = np.ones((4, 8), dtype=np.int16)
    copy = level.copy_with_tiles(new_tiles)
    new_tiles[0, 0] = 99
    assert copy.tiles[0, 0] == 1
