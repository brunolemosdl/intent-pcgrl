from __future__ import annotations

import pytest
from pcgrl.core.domain.tile import Tile, TileCategory
from pydantic import ValidationError


def test_tile_category_values() -> None:
    """Verify tile category values."""
    assert TileCategory.EMPTY.value == "empty"
    assert TileCategory.SOLID.value == "solid"
    assert TileCategory.ENEMY.value == "enemy"
    assert TileCategory.PIPE.value == "pipe"
    assert TileCategory.ENDPOINT.value == "endpoint"


def test_tile_frozen() -> None:
    """Verify tile frozen."""
    tile = Tile(
        tile_id=0,
        name="empty",
        glyph="-",
        category=TileCategory.EMPTY,
        passable=True,
        is_solid_top=False,
    )
    with pytest.raises(ValidationError):
        tile.name = "solid"


def test_tile_default_aliases_empty() -> None:
    """Verify tile default aliases empty."""
    tile = Tile(
        tile_id=0,
        name="empty",
        glyph="-",
        category=TileCategory.EMPTY,
        passable=True,
        is_solid_top=False,
    )
    assert tile.aliases == ()


def test_tile_with_aliases() -> None:
    """Verify tile with aliases."""
    tile = Tile(
        tile_id=1,
        name="solid",
        glyph="X",
        category=TileCategory.SOLID,
        passable=False,
        is_solid_top=True,
        aliases=("ground", "block"),
    )
    assert "ground" in tile.aliases
    assert "block" in tile.aliases


def test_tile_passable_for_empty() -> None:
    """Verify tile passable for empty."""
    tile = Tile(
        tile_id=0,
        name="empty",
        glyph="-",
        category=TileCategory.EMPTY,
        passable=True,
        is_solid_top=False,
    )
    assert tile.passable
    assert not tile.is_solid_top


def test_tile_solid_top_for_solid() -> None:
    """Verify tile solid top for solid."""
    tile = Tile(
        tile_id=1,
        name="solid",
        glyph="X",
        category=TileCategory.SOLID,
        passable=False,
        is_solid_top=True,
    )
    assert tile.is_solid_top
