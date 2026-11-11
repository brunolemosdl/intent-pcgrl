from __future__ import annotations

from pcgrl.rendering.png import PngRenderer, level_to_engine_string
from PIL import Image


def test_level_to_engine_string_width(config, catalog, blank_level) -> None:
    """Verify level to engine string width."""
    text = level_to_engine_string(blank_level, catalog)
    lines = text.splitlines()
    assert len(lines) == config.game.height_tiles


def test_level_to_engine_string_all_dash(config, catalog, blank_level) -> None:
    """Verify level to engine string all dash."""
    text = level_to_engine_string(blank_level, catalog)
    for char in text.replace("\n", ""):
        assert char == "-"


def test_level_to_engine_string_reserved_corner_is_solid(config, catalog) -> None:
    """Verify level to engine string reserved corner is solid."""
    import numpy as np
    from pcgrl.core.domain.level import AreaType, LevelGrid

    empty_id = catalog.id_for_name("empty")
    solid_id = catalog.id_for_name("solid")
    tiles = np.full((config.game.height_tiles, 8), empty_id, dtype=np.int16)
    tiles[1, 7] = solid_id
    level = LevelGrid(
        tiles=tiles,
        width_tiles=8,
        height_tiles=config.game.height_tiles,
        tile_size_px=16,
        area_type=AreaType.OVERWORLD,
    )
    text = level_to_engine_string(level, catalog)
    assert text.splitlines()[1][7] == "X"


def test_png_renderer_render_returns_image(config, catalog, blank_level) -> None:
    """Verify png renderer render returns image."""
    renderer = PngRenderer(catalog, tile_size_px=config.game.tile_size_px)
    img = renderer.render(blank_level)
    assert isinstance(img, Image.Image)


def test_png_renderer_image_dimensions(config, catalog, blank_level) -> None:
    """Verify png renderer image dimensions."""
    tile_size = 8
    renderer = PngRenderer(catalog, tile_size_px=tile_size)
    img = renderer.render(blank_level)
    expected_w = blank_level.width_tiles * tile_size
    expected_h = blank_level.height_tiles * tile_size
    assert img.size == (expected_w, expected_h)


def test_png_renderer_fallback_render_mode(config, catalog, blank_level) -> None:
    """Verify png renderer fallback render mode."""
    renderer = PngRenderer(catalog, tile_size_px=16)
    img = renderer._render_fallback(blank_level)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"


def test_png_renderer_fallback_solid_color(config, catalog) -> None:
    """Verify png renderer fallback solid color."""
    import numpy as np
    from pcgrl.core.domain.level import AreaType, LevelGrid

    h, w = (config.game.height_tiles, config.game.width_tiles)
    solid_id = catalog.id_for_name("solid")
    tiles = np.full((h, w), solid_id, dtype=np.int16)
    level = LevelGrid(
        tiles=tiles, width_tiles=w, height_tiles=h, tile_size_px=16, area_type=AreaType.OVERWORLD
    )
    renderer = PngRenderer(catalog, tile_size_px=16)
    img = renderer._render_fallback(level)
    assert isinstance(img, Image.Image)


def test_png_renderer_no_atlas_uses_fallback(config, catalog, blank_level) -> None:
    """Verify png renderer no atlas uses fallback."""
    from unittest.mock import patch

    with patch("pcgrl.rendering.png.SpriteAtlas.available", return_value=False):
        renderer = PngRenderer(catalog, tile_size_px=16)
    assert renderer._atlas is None
    img = renderer.render(blank_level)
    assert isinstance(img, Image.Image)
