from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class TileCategory(StrEnum):
    """Semantic category assigned to each tile in the catalog.

    Description:
        Groups tiles for intent metrics, masking rules, and corpus profiling.
        Values mirror the Mario tile families used across the pipeline.

    Examples:
        >>> TileCategory.SOLID
        <TileCategory.SOLID: 'solid'>
    """

    EMPTY = "empty"
    SOLID = "solid"
    BLOCK = "block"
    COIN = "coin"
    ENEMY = "enemy"
    PIPE = "pipe"
    ENDPOINT = "endpoint"
    PLATFORM = "platform"
    TERRAIN = "terrain"


class Tile(BaseModel):
    """Immutable descriptor for a single tile type in the catalog.

    Description:
        Carries the numeric id, display glyph, gameplay flags, and optional
        name aliases used when parsing ASCII levels or engine strings.

    Attributes:
        tile_id: Numeric identifier referenced in level grids.
        name: Canonical tile name (e.g. ``ground``, ``goomba``).
        glyph: Single-character ASCII representation.
        category: Semantic ``TileCategory`` for metrics and rules.
        passable: Whether Mario can occupy this cell.
        is_solid_top: Whether the top surface blocks upward movement.
        aliases: Alternate names accepted by the catalog resolver.
    """

    model_config = ConfigDict(frozen=True)
    tile_id: int
    name: str
    glyph: str
    category: TileCategory
    passable: bool
    is_solid_top: bool
    aliases: tuple[str, ...] = Field(default_factory=tuple)
