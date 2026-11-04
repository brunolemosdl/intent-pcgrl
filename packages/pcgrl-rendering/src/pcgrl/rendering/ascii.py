from __future__ import annotations

from pcgrl.core.domain.level import LevelGrid
from pcgrl.core.tiles.catalog import TileCatalog


def level_to_ascii(level: LevelGrid, catalog: TileCatalog) -> str:
    """Render a level grid as newline-separated ASCII using catalog glyphs.

    Args:
        level: Level grid to serialize.
        catalog: Tile catalog mapping ids to display glyphs.

    Returns:
        Multi-line ASCII representation of ``level``.
    """
    return level.to_ascii(catalog)
