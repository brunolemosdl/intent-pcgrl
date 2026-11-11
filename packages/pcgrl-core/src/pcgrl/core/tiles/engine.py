from __future__ import annotations

from pcgrl.core.domain.level import LevelGrid
from pcgrl.core.tiles.catalog import TileCatalog
from pcgrl.core.tiles.normalize import canonical_to_engine


def level_to_engine_string(level: LevelGrid, catalog: TileCatalog) -> str:
    """Convert a level grid to Mario-bridge engine glyph text.

    Args:
        level: Level grid with tile ids.
        catalog: Tile catalog used to resolve canonical names.

    Returns:
        Newline-separated engine glyph rows.
    """
    rows: list[str] = []
    for row in level.tiles:
        glyphs = [canonical_to_engine(catalog.name_for_id(int(tile_id))) for tile_id in row]
        rows.append("".join(glyphs))
    return "\n".join(rows)
