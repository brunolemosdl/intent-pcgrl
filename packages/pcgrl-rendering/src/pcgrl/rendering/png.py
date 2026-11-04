from __future__ import annotations

from PIL import Image, ImageDraw

from pcgrl.core.domain.level import LevelGrid
from pcgrl.core.tiles.catalog import TileCatalog
from pcgrl.core.tiles.engine import level_to_engine_string
from pcgrl.rendering.layout import parse_layout
from pcgrl.rendering.sprites import SpriteAtlas, composite_layout

TILE_COLORS = {
    "empty": (135, 206, 235),
    "solid": (136, 68, 0),
    "brick": (180, 90, 30),
    "coin": (255, 215, 0),
    "goomba": (120, 40, 20),
    "player": (255, 0, 0),
    "flagpole": (0, 180, 0),
}


class PngRenderer:
    """Rasterize ``LevelGrid`` instances to RGB PIL images.

    Description:
        Uses Mario sprite sheets when assets are available; otherwise falls
        back to flat colored rectangles per tile name.

    Attributes:
        tile_size_px: Edge length of one rendered tile in pixels.
    """

    def __init__(self, catalog: TileCatalog, tile_size_px: int = 16) -> None:
        """Load catalog metadata and optionally initialize a sprite atlas.

        Args:
            catalog: Tile catalog for engine serialization and fallback colors.
            tile_size_px: Pixel size of one tile in the output image.
        """
        self._catalog = catalog
        self._tile_size_px = tile_size_px
        self._atlas = SpriteAtlas() if SpriteAtlas.available() else None

    def render(self, level: LevelGrid) -> Image.Image:
        """Render a level grid to a PIL RGB image.

        Args:
            level: Level grid to draw.

        Returns:
            Composited sprite image when assets exist, otherwise a flat fallback.
        """
        if self._atlas is not None:
            engine_text = level_to_engine_string(level, self._catalog)
            parsed = parse_layout(engine_text)
            return composite_layout(parsed, atlas=self._atlas, tile_size_px=self._tile_size_px)
        return self._render_fallback(level)

    def _render_fallback(self, level: LevelGrid) -> Image.Image:
        """Draw solid-color rectangles for each tile when sprites are unavailable."""
        width = level.width_tiles * self._tile_size_px
        height = level.height_tiles * self._tile_size_px
        image = Image.new("RGB", (width, height), (135, 206, 235))
        draw = ImageDraw.Draw(image)
        for y in range(level.height_tiles):
            for x in range(level.width_tiles):
                name = self._catalog.name_for_id(int(level.tiles[y, x]))
                color = TILE_COLORS.get(name, (100, 100, 100))
                x0 = x * self._tile_size_px
                y0 = y * self._tile_size_px
                draw.rectangle(
                    [x0, y0, x0 + self._tile_size_px - 1, y0 + self._tile_size_px - 1], fill=color
                )
        return image
