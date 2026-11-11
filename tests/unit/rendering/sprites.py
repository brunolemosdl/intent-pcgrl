from __future__ import annotations

import pytest
from pcgrl.rendering.layout import SpriteType, parse_layout
from pcgrl.rendering.sprites import SpriteAtlas, composite_layout


def test_sprite_atlas_available() -> None:
    """Verify sprite atlas available."""
    assert SpriteAtlas.available()


@pytest.fixture(scope="module")
def atlas() -> SpriteAtlas:
    """Atlas."""
    return SpriteAtlas()


def test_level_tile_returns_image(atlas) -> None:
    """Verify level tile returns image."""
    from PIL import Image

    tile = atlas.level_tile(1)
    assert isinstance(tile, Image.Image)


def test_enemy_tile_returns_image(atlas) -> None:
    """Verify enemy tile returns image."""
    from PIL import Image

    tile = atlas.enemy_tile(SpriteType.GOOMBA)
    assert isinstance(tile, Image.Image)


def test_mario_tile_returns_image(atlas) -> None:
    """Verify mario tile returns image."""
    from PIL import Image

    tile = atlas.mario_tile(0)
    assert isinstance(tile, Image.Image)


def test_composite_layout_size(atlas) -> None:
    """Verify composite layout size."""
    level = "X" * 16 + "\n" + "X" * 16
    parsed = parse_layout(level)
    image = composite_layout(parsed, atlas=atlas, tile_size_px=16)
    assert image.size == (16 * 16, 2 * 16)


def test_composite_layout_with_enemy(atlas) -> None:
    """Verify composite layout with enemy."""
    level = "g" + "-" * 15 + "\n" + "X" * 16
    parsed = parse_layout(level)
    from PIL import Image

    image = composite_layout(parsed, atlas=atlas, tile_size_px=16)
    assert isinstance(image, Image.Image)


def test_composite_layout_with_mario(atlas) -> None:
    """Verify composite layout with mario."""
    level = "M" + "-" * 15 + "\n" + "X" * 16
    parsed = parse_layout(level)
    from PIL import Image

    assert parsed.mario_present
    image = composite_layout(parsed, atlas=atlas, tile_size_px=16)
    assert isinstance(image, Image.Image)


def test_composite_empty_layout(atlas) -> None:
    """Verify composite empty layout."""
    from pcgrl.rendering.layout import ParsedLayout
    from PIL import Image

    empty = ParsedLayout(
        level_tiles=[],
        sprite_types=[],
        mario_tile_x=0,
        mario_tile_y=0,
        exit_tile_x=0,
        exit_tile_y=0,
        mario_present=False,
    )
    image = composite_layout(empty, atlas=atlas, tile_size_px=16)
    assert isinstance(image, Image.Image)
    assert image.size == (0, 0)
