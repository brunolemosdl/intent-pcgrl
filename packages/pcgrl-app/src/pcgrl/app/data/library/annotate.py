from __future__ import annotations

import numpy as np

from pcgrl.app.data.library.extract import ChunkWindow
from pcgrl.core.domain.tile import TileCategory
from pcgrl.core.library.types import Chunk
from pcgrl.core.tiles.catalog import TileCatalog

_GROUND_CATEGORIES = frozenset(
    {TileCategory.SOLID, TileCategory.PIPE, TileCategory.BLOCK, TileCategory.PLATFORM}
)


def surface_height(tiles: np.ndarray, column: int, catalog: TileCatalog) -> int:
    """Count contiguous ground rows from the bottom at one column.

    Args:
        tiles: Chunk tile matrix.
        column: Column index to measure.
        catalog: Tile catalog for ground-category classification.

    Returns:
        Standable surface height, or ``0`` for a bottomless gap.
    """
    height = 0
    for row in range(tiles.shape[0] - 1, -1, -1):
        tile = catalog.tile_for_id(int(tiles[row, column]))
        if tile.category in _GROUND_CATEGORIES:
            height += 1
        else:
            break
    return height


def count_categories(tiles: np.ndarray, catalog: TileCatalog) -> dict[str, int]:
    """Count non-empty tiles grouped by tile category name.

    Args:
        tiles: Tile matrix to summarize.
        catalog: Tile catalog for category lookup.

    Returns:
        Mapping from category name to tile count.
    """
    counts: dict[str, int] = {}
    ids, frequencies = np.unique(tiles, return_counts=True)
    for tile_id, frequency in zip(ids.tolist(), frequencies.tolist(), strict=True):
        category = catalog.tile_for_id(int(tile_id)).category
        if category == TileCategory.EMPTY:
            continue
        counts[category.value] = counts.get(category.value, 0) + int(frequency)
    return counts


def annotate_window(window: ChunkWindow, chunk_id: int, catalog: TileCatalog) -> Chunk:
    """Turn an extracted window into an annotated chunk record.

    Args:
        window: Raw chunk window with family and provenance metadata.
        chunk_id: Sequential id assigned during library construction.
        catalog: Tile catalog for category and surface-height analysis.

    Returns:
        ``Chunk`` with seam heights, gap flags, and category counts.
    """
    entry_height = surface_height(window.tiles, 0, catalog)
    exit_height = surface_height(window.tiles, window.tiles.shape[1] - 1, catalog)
    return Chunk(
        chunk_id=chunk_id,
        tiles=window.tiles,
        family=window.family,
        level_id=window.level_id,
        column_offset=window.column_offset,
        entry_height=entry_height,
        exit_height=exit_height,
        gap_at_left=entry_height == 0,
        gap_at_right=exit_height == 0,
        category_counts=count_categories(window.tiles, catalog),
    )
