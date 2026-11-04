from __future__ import annotations

from pcgrl.core.domain.level import LevelGrid
from pcgrl.core.settings.schema import RuntimeConfig
from pcgrl.core.tiles.catalog import TileCatalog


def _category(config: RuntimeConfig, key: str) -> set[str]:
    """Return the named category tile-name set from config."""
    return set(config.categories.get(key, []))


def block_names(config: RuntimeConfig) -> set[str]:
    """Return interactive block and brick tile names from ``solid_like``.

    Args:
        config: Runtime configuration.

    Returns:
        Set of block-like tile names.
    """
    return {
        name
        for name in config.categories.get("solid_like", [])
        if "block" in name or name.startswith("brick") or name.startswith("question")
    }


def solid_like_ids(catalog: TileCatalog, config: RuntimeConfig) -> set[int]:
    """Return catalog ids for all ``solid_like`` category names."""
    return {
        catalog.id_for_name(name)
        for name in config.categories.get("solid_like", [])
        if name in catalog.name_to_id
    }


def near_standable(standable: set[tuple[int, int]], y: int, x: int, threshold: int) -> bool:
    """True if any standable cell is within Manhattan `threshold` of (y, x)."""
    for dy in range(-threshold, threshold + 1):
        budget = threshold - abs(dy)
        for dx in range(-budget, budget + 1):
            if (y + dy, x + dx) in standable:
                return True
    return False


def support_distance_below(level: LevelGrid, y: int, x: int, solid_ids: set[int]) -> int | None:
    """Rows from (y, x) down to the nearest solid-like tile below it, or None."""
    height = level.height_tiles
    for d in range(1, height - y):
        if int(level.tiles[y + d, x]) in solid_ids:
            return d
    return None


def _standable_local(level: LevelGrid, y: int, x: int, catalog: TileCatalog) -> bool:
    """A cell a player can stand in: passable, with a solid-top tile directly
    below it. Purely local (no BFS)."""
    if y + 1 >= level.height_tiles:
        return False
    if not catalog.tile_for_id(int(level.tiles[y, x])).passable:
        return False
    return catalog.tile_for_id(int(level.tiles[y + 1, x])).is_solid_top


def _standable_within(level: LevelGrid, y: int, x: int, catalog: TileCatalog, reach: int) -> bool:
    """True if any standable cell lies within Manhattan `reach` of (y, x).

    Cheap local proxy for "a player could plausibly get under/onto this cell";
    used to reject deep-sky floating objects without a full reachability BFS.
    Real route reachability is still confirmed downstream at episode end.
    """
    height, width = level.height_tiles, level.width_tiles
    for dy in range(-reach, reach + 1):
        ny = y + dy
        if not 0 <= ny < height:
            continue
        budget = reach - abs(dy)
        for dx in range(-budget, budget + 1):
            nx = x + dx
            if 0 <= nx < width and _standable_local(level, ny, nx, catalog):
                return True
    return False


def coin_locally_collectible(
    level: LevelGrid, y: int, x: int, catalog: TileCatalog, config: RuntimeConfig
) -> bool:
    """Cheap local plausibility for a coin (floating coins allowed, but only

    Description:
        near a standable cell).

        A coin must sit in open space (a passable orthogonal neighbour) AND have a
        standable cell within jump reach, so an arc stranded deep in the sky is
        treated as noise instead of free reward. True reachability of floating coins
        is still confirmed downstream by the reachability flood.

    Args:
        level: See signature.
        y: See signature.
        x: See signature.
        catalog: See signature.
        config: See signature.

    Returns:
        bool: Result of the operation.
    """
    height, width = level.height_tiles, level.width_tiles
    open_neighbour = False
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ny, nx = y + dy, x + dx
        if (
            0 <= ny < height
            and 0 <= nx < width
            and catalog.tile_for_id(int(level.tiles[ny, nx])).passable
        ):
            open_neighbour = True
            break
    if not open_neighbour:
        return False
    reach = int(config.metrics.semantic.coin_max_height_above_surface)
    return _standable_within(level, y, x, catalog, reach)


def block_locally_hittable(
    level: LevelGrid, y: int, x: int, catalog: TileCatalog, config: RuntimeConfig
) -> bool:
    """Cheap local plausibility for a block (floating blocks allowed, but only

    Description:
        near a standable cell).

        Requires clearance directly below (Mario can pass under and bump it) AND a
        standable cell within jump reach, so a block stranded deep in the sky is
        treated as noise. Whether the cell below is actually on a route is confirmed
        downstream.

    Args:
        level: See signature.
        y: See signature.
        x: See signature.
        catalog: See signature.
        config: See signature.

    Returns:
        bool: Result of the operation.
    """
    height = level.height_tiles
    if y + 1 >= height:
        return False
    if not catalog.tile_for_id(int(level.tiles[y + 1, x])).passable:
        return False
    reach = int(config.metrics.semantic.block_max_height_above_surface)
    return _standable_within(level, y, x, catalog, reach)


def enemy_locally_grounded(
    level: LevelGrid, y: int, x: int, catalog: TileCatalog, config: RuntimeConfig
) -> bool:
    """Return True when a grounded enemy has solid support (flying exempt)."""
    name = catalog.name_for_id(int(level.tiles[y, x]))
    if catalog.is_flying_enemy(name):
        return True
    if y + 1 >= level.height_tiles:
        return False
    return catalog.tile_for_id(int(level.tiles[y + 1, x])).is_solid_top


def solid_is_contiguous_platform(
    level: LevelGrid, y: int, x: int, catalog: TileCatalog, config: RuntimeConfig
) -> bool:
    """An elevated solid counts as a platform tile only if it has a horizontal

    Description:
        solid-like neighbour (no isolated floating tiles). Ground row is exempt.

    Args:
        level: See signature.
        y: See signature.
        x: See signature.
        catalog: See signature.
        config: See signature.

    Returns:
        bool: Result of the operation.
    """
    height, width = level.height_tiles, level.width_tiles
    if y >= height - 1:
        return True
    ids = solid_like_ids(catalog, config)
    left = x
    right = x
    while left - 1 >= 0 and int(level.tiles[y, left - 1]) in ids:
        left -= 1
    while right + 1 < width and int(level.tiles[y, right + 1]) in ids:
        right += 1
    run_width = right - left + 1
    if run_width < config.metrics.semantic.min_platform_width:
        return False
    if y <= 0:
        return False
    for c in range(left, right + 1):
        if not catalog.tile_for_id(int(level.tiles[y - 1, c])).passable:
            return False
    return True


def placement_is_locally_useless(
    level: LevelGrid, y: int, x: int, catalog: TileCatalog, config: RuntimeConfig
) -> bool:
    """True iff the object tile at (y, x) fails its cheap local usefulness test.

    Description:
        Non-object tiles (empty, ground-row solids, pipes) are not judged.

    Args:
        level: See signature.
        y: See signature.
        x: See signature.
        catalog: See signature.
        config: See signature.

    Returns:
        bool: Result of the operation.
    """
    name = catalog.name_for_id(int(level.tiles[y, x]))
    if name in _category(config, "enemies"):
        return not enemy_locally_grounded(level, y, x, catalog, config)
    # Question blocks are both "rewards" and "blocks"; treat them as blocks
    # (hit from below), so check blocks before rewards.
    if name in block_names(config):
        return not block_locally_hittable(level, y, x, catalog, config)
    if name in _category(config, "rewards"):
        return not coin_locally_collectible(level, y, x, catalog, config)
    if name == "solid" and y < level.height_tiles - 1:
        return not solid_is_contiguous_platform(level, y, x, catalog, config)
    return False
