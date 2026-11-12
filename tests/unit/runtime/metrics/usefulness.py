from __future__ import annotations

import numpy as np
from pcgrl.core.domain.level import AreaType, LevelGrid
from pcgrl.runtime.metrics.usefulness import (
    block_locally_hittable,
    coin_locally_collectible,
    enemy_locally_grounded,
    placement_is_locally_useless,
    solid_is_contiguous_platform,
)


def _ground_level(config, catalog) -> LevelGrid:
    """Build a level with a full solid ground row."""
    h, w = config.game.height_tiles, config.game.width_tiles
    tiles = np.full((h, w), catalog.id_for_name("empty"), dtype=np.int16)
    tiles[h - 1, :] = catalog.id_for_name("solid")
    return LevelGrid(
        tiles=tiles, width_tiles=w, height_tiles=h, tile_size_px=16, area_type=AreaType.OVERWORLD
    )


def test_coin_near_surface_is_collectible(config, catalog) -> None:
    """Verify coin near surface is collectible."""
    level = _ground_level(config, catalog)
    h = config.game.height_tiles
    tiles = level.tiles.copy()
    tiles[h - 4, 5] = catalog.id_for_name("coin")  # 3 above ground
    level = level.copy_with_tiles(tiles)
    assert coin_locally_collectible(level, h - 4, 5, catalog, config)
    assert not placement_is_locally_useless(level, h - 4, 5, catalog, config)


def test_stranded_sky_coin_is_useless(config, catalog) -> None:
    """Verify a sky coin far from any standable surface is locally useless."""
    # A coin stranded deep in the sky, with no standable cell within jump reach,
    # is locally useless (this was the flood-the-sky exploit). Floating coins
    # near a surface are still allowed (see test_coin_near_surface_is_collectible).
    level = _ground_level(config, catalog)
    tiles = level.tiles.copy()
    tiles[0, 5] = catalog.id_for_name("coin")  # row 0, ground far below
    level = level.copy_with_tiles(tiles)
    assert not coin_locally_collectible(level, 0, 5, catalog, config)
    assert placement_is_locally_useless(level, 0, 5, catalog, config)


def test_buried_coin_is_useless(config, catalog) -> None:
    """Verify a coin sealed on all sides is locally useless."""
    # A coin sealed inside solids on every side is locally useless.
    level = _ground_level(config, catalog)
    solid = catalog.id_for_name("solid")
    tiles = level.tiles.copy()
    y, x = 6, 5
    tiles[y, x] = catalog.id_for_name("coin")
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        tiles[y + dy, x + dx] = solid
    level = level.copy_with_tiles(tiles)
    assert not coin_locally_collectible(level, y, x, catalog, config)
    assert placement_is_locally_useless(level, y, x, catalog, config)


def test_block_hittable_above_ground(config, catalog) -> None:
    """Verify block hittable above ground."""
    level = _ground_level(config, catalog)
    h = config.game.height_tiles
    tiles = level.tiles.copy()
    tiles[h - 4, 5] = catalog.id_for_name("brick")  # empty below, ground within reach
    level = level.copy_with_tiles(tiles)
    assert block_locally_hittable(level, h - 4, 5, catalog, config)


def test_stranded_sky_block_is_useless(config, catalog) -> None:
    """Verify a sky block far from any standable surface is locally useless."""
    # A block stranded deep in the sky (no standable cell within jump reach) is
    # locally useless. Blocks bumpable from a nearby surface are still allowed
    # (see test_block_hittable_above_ground).
    level = _ground_level(config, catalog)
    tiles = level.tiles.copy()
    tiles[0, 5] = catalog.id_for_name("brick")
    level = level.copy_with_tiles(tiles)
    assert not block_locally_hittable(level, 0, 5, catalog, config)
    assert placement_is_locally_useless(level, 0, 5, catalog, config)


def test_block_without_clearance_below_is_useless(config, catalog) -> None:
    """Verify block without clearance below is useless."""
    # No passable cell directly below -> cannot be bumped -> locally useless.
    level = _ground_level(config, catalog)
    tiles = level.tiles.copy()
    h = config.game.height_tiles
    tiles[h - 2, 5] = catalog.id_for_name("brick")  # sitting directly on ground
    level = level.copy_with_tiles(tiles)
    assert not block_locally_hittable(level, h - 2, 5, catalog, config)
    assert placement_is_locally_useless(level, h - 2, 5, catalog, config)


def test_enemy_grounded_vs_floating(config, catalog) -> None:
    """Verify enemy grounded vs floating."""
    level = _ground_level(config, catalog)
    h = config.game.height_tiles
    tiles = level.tiles.copy()
    tiles[h - 2, 5] = catalog.id_for_name("goomba")  # on the ground
    tiles[3, 9] = catalog.id_for_name("goomba")  # floating
    level = level.copy_with_tiles(tiles)
    assert enemy_locally_grounded(level, h - 2, 5, catalog, config)
    assert not enemy_locally_grounded(level, 3, 9, catalog, config)
    assert placement_is_locally_useless(level, 3, 9, catalog, config)


def test_contiguous_platform_detection(config, catalog) -> None:
    """Verify contiguous platform detection."""
    width = config.metrics.semantic.min_platform_width
    level = _ground_level(config, catalog)
    tiles = level.tiles.copy()
    # A run at least `min_platform_width` wide qualifies as a platform.
    tiles[5, 10 : 10 + width] = catalog.id_for_name("solid")
    level = level.copy_with_tiles(tiles)
    assert solid_is_contiguous_platform(level, 5, 10, catalog, config)
    # A run narrower than `min_platform_width` does not qualify.
    tiles2 = level.tiles.copy()
    tiles2[5, 10 + width - 1] = catalog.id_for_name("empty")
    isolated = level.copy_with_tiles(tiles2)
    assert not solid_is_contiguous_platform(isolated, 5, 10, catalog, config)
