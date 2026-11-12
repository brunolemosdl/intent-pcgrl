from __future__ import annotations

from dataclasses import dataclass, field

from pcgrl.core.domain.tile import Tile, TileCategory
from pcgrl.core.settings.schema import RuntimeConfig


def _infer_category(name: str, categories: dict[str, list[str]]) -> TileCategory:
    """Map a tile name to a semantic category using config groupings."""
    for category_name, members in categories.items():
        if name in members:
            mapping = {
                "solid_like": TileCategory.SOLID,
                "enemies": TileCategory.ENEMY,
                "pipes": TileCategory.PIPE,
                "rewards": TileCategory.COIN,
                "endpoint_types": TileCategory.ENDPOINT,
                "platform_types": TileCategory.PLATFORM,
            }
            return mapping.get(category_name, TileCategory.TERRAIN)
    if name == "empty":
        return TileCategory.EMPTY
    if name in {"player", "flagpole"}:
        return TileCategory.ENDPOINT
    if name.startswith("question") or name.startswith("brick") or name.startswith("hidden"):
        return TileCategory.BLOCK
    return TileCategory.TERRAIN


def _is_passable(name: str, category: TileCategory) -> bool:
    """Return whether Mario can occupy the cell."""
    if category in {TileCategory.EMPTY, TileCategory.COIN, TileCategory.ENDPOINT}:
        return True
    return name in {"coin", "player", "flagpole"}


def _is_solid_top(name: str, category: TileCategory) -> bool:
    """Return whether the top surface blocks upward movement."""
    if category in {
        TileCategory.SOLID,
        TileCategory.BLOCK,
        TileCategory.PIPE,
        TileCategory.PLATFORM,
    }:
        return True
    return name in {"solid", "brick", "used_block"}


@dataclass(frozen=True)
class TileCatalog:
    """Immutable lookup tables for tile ids, names, glyphs, and categories.

    Attributes:
        tiles: Ordered tile descriptors indexed by id.
        name_to_id: Canonical name to numeric id mapping.
        glyph_to_id: Single-character glyph to id mapping.
        alias_to_name: Alias name to canonical name mapping.
        area_tile_names: Allowed tile names per area type.
        flying_enemy_names: Enemy names treated as aerial threats.
    """

    tiles: tuple[Tile, ...]
    name_to_id: dict[str, int]
    glyph_to_id: dict[str, int]
    alias_to_name: dict[str, str]
    area_tile_names: dict[str, frozenset[str]] = field(default_factory=dict)
    flying_enemy_names: frozenset[str] = field(default_factory=frozenset)

    @property
    def size(self) -> int:
        """Return the number of tiles in the catalog."""
        return len(self.tiles)

    def tile_for_id(self, tile_id: int) -> Tile:
        """Return the tile descriptor for a numeric id.

        Args:
            tile_id: Catalog index.

        Returns:
            Matching ``Tile`` descriptor.
        """
        return self.tiles[tile_id]

    def id_for_name(self, name: str) -> int:
        """Resolve a canonical or alias name to a tile id.

        Args:
            name: Tile name or alias.

        Returns:
            Numeric tile id.

        Raises:
            KeyError: When the name is unknown.
        """
        canonical = self.alias_to_name.get(name, name)
        if canonical not in self.name_to_id:
            raise KeyError(f"Unknown tile name: {name}")
        return self.name_to_id[canonical]

    def id_for_glyph(self, glyph: str) -> int:
        """Resolve a single-character glyph to a tile id.

        Args:
            glyph: One-character ASCII glyph.

        Returns:
            Numeric tile id.

        Raises:
            KeyError: When the glyph is unknown.
        """
        if glyph not in self.glyph_to_id:
            raise KeyError(f"Unknown tile glyph: {glyph!r}")
        return self.glyph_to_id[glyph]

    def glyph_for_id(self, tile_id: int) -> str:
        """Return the display glyph for a tile id."""
        return self.tiles[tile_id].glyph

    def name_for_id(self, tile_id: int) -> str:
        """Return the canonical name for a tile id."""
        return self.tiles[tile_id].name

    def names_in_category(self, category: TileCategory) -> tuple[str, ...]:
        """Return canonical names for all tiles in a category.

        Args:
            category: Semantic tile category filter.

        Returns:
            Tuple of matching tile names in catalog order.
        """
        return tuple(tile.name for tile in self.tiles if tile.category == category)

    def allowed_names(self, area_type: str) -> frozenset[str]:
        """Return tile names permitted in an area type.

        Args:
            area_type: Area key from game configuration.

        Returns:
            Allowed canonical names, or an empty set when undefined.
        """
        return self.area_tile_names.get(area_type, frozenset())

    def is_flying_enemy(self, name: str) -> bool:
        """Return whether a tile name denotes a flying enemy."""
        return name in self.flying_enemy_names


def load_catalog(config: RuntimeConfig) -> TileCatalog:
    """Build a tile catalog from validated runtime settings.

    Args:
        config: Runtime configuration containing tile definitions.

    Returns:
        Fully indexed ``TileCatalog`` instance.

    Raises:
        ValueError: When tile names, glyphs, or aliases are inconsistent.
    """
    categories = config.categories
    aliases = config.aliases
    seen_glyphs: dict[str, str] = {}
    tiles: list[Tile] = []
    name_to_id: dict[str, int] = {}
    glyph_to_id: dict[str, int] = {}
    alias_to_name = dict(aliases)
    for tile_id, (name, glyph) in enumerate(config.tiles.items()):
        if name in name_to_id:
            raise ValueError(f"Duplicate tile name: {name}")
        if len(glyph) != 1:
            raise ValueError(f"Tile {name} glyph must be a single character")
        if glyph in seen_glyphs and seen_glyphs[glyph] != name:
            raise ValueError(f"Duplicate tile glyph {glyph!r}: {seen_glyphs[glyph]!r} and {name!r}")
        seen_glyphs[glyph] = name
        category = _infer_category(name, categories)
        tile = Tile(
            tile_id=tile_id,
            name=name,
            glyph=glyph,
            category=category,
            passable=_is_passable(name, category),
            is_solid_top=_is_solid_top(name, category),
            aliases=tuple((key for key, value in aliases.items() if value == name)),
        )
        tiles.append(tile)
        name_to_id[name] = tile_id
        glyph_to_id[glyph] = tile_id
    for alias, canonical in aliases.items():
        if canonical not in name_to_id:
            raise ValueError(f"Alias {alias} points to unknown tile {canonical}")
    area_tile_names = {area: frozenset(names) for area, names in config.area_tiles.items()}
    flying_enemy_names = frozenset(categories.get("flying_enemies", []))
    return TileCatalog(
        tiles=tuple(tiles),
        name_to_id=name_to_id,
        glyph_to_id=glyph_to_id,
        alias_to_name=alias_to_name,
        area_tile_names=area_tile_names,
        flying_enemy_names=flying_enemy_names,
    )
