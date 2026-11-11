from __future__ import annotations

from enum import StrEnum
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from pcgrl.core.tiles.catalog import TileCatalog


class AreaType(StrEnum):
    """Playfield theme for a level grid."""

    OVERWORLD = "overworld"


class LevelGrid(BaseModel):
    """Immutable tile-id grid with layout metadata.

    Description:
        Stores the numeric tile matrix plus dimensions used by validators,
        renderers, and the gameplay bridge.

    Attributes:
        tiles: Tile-id matrix with shape ``(height_tiles, width_tiles)``.
        width_tiles: Level width in tiles.
        height_tiles: Level height in tiles.
        tile_size_px: Pixel size of each tile for rendering.
        area_type: Playfield theme (currently overworld only).
        metadata: Optional provenance or generation tags.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    tiles: np.ndarray
    width_tiles: int
    height_tiles: int
    tile_size_px: int
    area_type: AreaType = AreaType.OVERWORLD
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_ascii(self, catalog: TileCatalog) -> str:
        """Render the grid as newline-separated ASCII glyph rows.

        Args:
            catalog: Tile catalog used to map ids to glyphs.

        Returns:
            Multi-line ASCII representation of the level.
        """
        rows: list[str] = []
        for row in self.tiles:
            rows.append("".join(catalog.glyph_for_id(int(tile_id)) for tile_id in row))
        return "\n".join(rows)

    @classmethod
    def from_ascii(cls, text: str, catalog: TileCatalog, **kwargs: Any) -> LevelGrid:
        """Parse ASCII glyph text into a level grid.

        Args:
            text: Multi-line ASCII level text.
            catalog: Tile catalog used to map glyphs to ids.
            **kwargs: Optional overrides for ``tile_size_px``, ``area_type``,
                and ``metadata``.

        Returns:
            Parsed level grid padded to a rectangular shape.

        Raises:
            ValueError: When ``text`` is empty.
            KeyError: When an unknown glyph appears in ``text``.
        """
        lines = [line.rstrip("\n") for line in text.strip().splitlines()]
        if not lines:
            raise ValueError("ASCII level text is empty")
        height = len(lines)
        width = max(len(line) for line in lines)
        matrix = np.full((height, width), catalog.id_for_name("empty"), dtype=np.int16)
        for y, line in enumerate(lines):
            padded = line.ljust(width, catalog.glyph_for_id(catalog.id_for_name("empty")))
            for x, glyph in enumerate(padded):
                matrix[y, x] = catalog.id_for_glyph(glyph)
        return cls(
            tiles=matrix,
            width_tiles=width,
            height_tiles=height,
            tile_size_px=kwargs.get("tile_size_px", 16),
            area_type=kwargs.get("area_type", AreaType.OVERWORLD),
            metadata=kwargs.get("metadata", {}),
        )

    def copy_with_tiles(self, tiles: np.ndarray) -> LevelGrid:
        """Return a copy of the level with a new tile matrix.

        Args:
            tiles: Replacement tile-id matrix.

        Returns:
            New ``LevelGrid`` sharing all other fields.
        """
        return self.model_copy(update={"tiles": tiles.copy()})
