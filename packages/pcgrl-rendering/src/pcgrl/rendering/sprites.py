from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PIL import Image

from pcgrl.core.helpers.path import asset_path
from pcgrl.rendering.layout import ENEMY_SHEET_INDEX, ParsedLayout, SpriteType

SKY_COLOR = (92, 148, 252)
DEFAULT_ASSETS_DIR = asset_path("mario")


@lru_cache(maxsize=8)
def _load_sheet(path: str, tile_width: int, tile_height: int) -> tuple[Image.Image, ...]:
    """Slice a PNG sprite sheet into fixed-size RGBA tiles."""
    source = Image.open(path).convert("RGBA")
    columns = source.width // tile_width
    rows = source.height // tile_height
    tiles: list[Image.Image] = []
    for row in range(rows):
        for column in range(columns):
            box = (
                column * tile_width,
                row * tile_height,
                (column + 1) * tile_width,
                (row + 1) * tile_height,
            )
            tiles.append(source.crop(box))
    return tuple(tiles)


def _sheet_tile(sheet: tuple[Image.Image, ...], index: int, columns: int) -> Image.Image:
    """Return the tile at ``index`` assuming row-major sheet layout."""
    return sheet[index // columns * columns + index % columns]


class SpriteAtlas:
    """Lazy loader for Mario level, enemy, and player sprite sheets.

    Description:
        Crops vendor PNG sheets into addressable tiles used by
        ``composite_layout`` when rendering engine-format levels.

    Attributes:
        assets_dir: Root directory containing ``mapsheet.png`` and related files.
    """

    def __init__(self, assets_dir: Path | None = None) -> None:
        """Load level, enemy, and small-Mario sheets from ``assets_dir``.

        Args:
            assets_dir: Directory with Mario PNG assets; defaults to vendor path.
        """
        root = assets_dir or DEFAULT_ASSETS_DIR
        self._level = _load_sheet(str(root / "mapsheet.png"), 16, 16)
        self._enemies = _load_sheet(str(root / "enemysheet.png"), 16, 32)
        self._small_mario = _load_sheet(str(root / "smallmariosheet.png"), 16, 16)
        self._level_columns = 8

    @staticmethod
    def available(assets_dir: Path | None = None) -> bool:
        """Return whether required sprite assets exist on disk.

        Args:
            assets_dir: Directory to inspect; defaults to vendor Mario assets.

        Returns:
            ``True`` when ``mapsheet.png`` is present under ``assets_dir``.
        """
        root = assets_dir or DEFAULT_ASSETS_DIR
        return (root / "mapsheet.png").is_file()

    def level_tile(self, index: int) -> Image.Image:
        """Return a terrain tile cropped from the level sprite sheet.

        Args:
            index: Tile index in the engine layout encoding.

        Returns:
            RGBA tile image for the given index.
        """
        return _sheet_tile(self._level, index, self._level_columns)

    def enemy_tile(self, sprite: SpriteType) -> Image.Image:
        """Return the enemy sprite image for a parsed layout sprite type.

        Args:
            sprite: Enemy enum value from ``parse_layout``.

        Returns:
            RGBA enemy tile aligned to the bottom of a 16px cell.
        """
        index = ENEMY_SHEET_INDEX[sprite]
        return _sheet_tile(self._enemies, index, self._level_columns)

    def mario_tile(self, index: int = 0) -> Image.Image:
        """Return a small-Mario pose tile from the player sheet.

        Args:
            index: Frame index within the small-Mario sheet.

        Returns:
            RGBA Mario tile for compositing at the spawn position.
        """
        columns = max(1, len(self._small_mario))
        return _sheet_tile(self._small_mario, index, columns)


def composite_layout(
    parsed: ParsedLayout, *, atlas: SpriteAtlas, tile_size_px: int = 16
) -> Image.Image:
    """Composite parsed engine layout into a single RGB level image.

    Args:
        parsed: Output of ``parse_layout`` with tiles, sprites, and spawn coords.
        atlas: Loaded Mario sprite sheets.
        tile_size_px: Pixel edge length for each layout cell.

    Returns:
        RGB image with sky background, terrain, enemies, and optional Mario.
    """
    width_tiles = len(parsed.level_tiles)
    height_tiles = len(parsed.level_tiles[0]) if width_tiles else 0
    image = Image.new("RGB", (width_tiles * tile_size_px, height_tiles * tile_size_px), SKY_COLOR)
    for y in range(height_tiles):
        for x in range(width_tiles):
            tile_index = parsed.level_tiles[x][y]
            if tile_index == 0:
                continue
            sprite = atlas.level_tile(tile_index)
            image.paste(sprite, (x * tile_size_px, y * tile_size_px), sprite)
    for y in range(height_tiles):
        for x in range(width_tiles):
            sprite_type = parsed.sprite_types[x][y]
            if sprite_type is SpriteType.NONE:
                continue
            enemy = atlas.enemy_tile(sprite_type)
            px = x * tile_size_px
            py = y * tile_size_px + tile_size_px - enemy.height
            image.paste(enemy, (px, py), enemy)
    if parsed.mario_present:
        mario = atlas.mario_tile(0)
        px = parsed.mario_tile_x * tile_size_px
        py = parsed.mario_tile_y * tile_size_px + tile_size_px - mario.height
        image.paste(mario, (px, py), mario)
    return image
