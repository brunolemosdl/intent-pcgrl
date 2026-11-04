from __future__ import annotations

from pathlib import Path

import numpy as np

from pcgrl.core.domain.level import AreaType, LevelGrid
from pcgrl.core.tiles.catalog import TileCatalog
from pcgrl.core.tiles.normalize import corpus_glyph_to_name


def parse_level_file(path: Path, catalog: TileCatalog, tile_size_px: int = 16) -> LevelGrid:
    """Parse a VGLC level file into a normalized 16-row ``LevelGrid``.

    Args:
        path: Path to a ``.txt`` VGLC level file.
        catalog: Tile catalog for glyph-to-id resolution.
        tile_size_px: Pixel size recorded in level metadata.

    Returns:
        Parsed overworld level grid padded or trimmed to 16 rows.

    Raises:
        ValueError: When normalization to 16 rows fails.
        OSError: When the level file cannot be read.
    """
    text = path.read_text(encoding="utf-8").strip("\n")
    lines = [line.rstrip("\n") for line in text.splitlines() if line.strip() != ""]
    if len(lines) < 16:
        lines.extend(["-"] * (16 - len(lines)))
    elif len(lines) > 16:
        lines = lines[:16]
    if len(lines) != 16:
        raise ValueError(f"{path.name} must normalize to 16 rows, got {len(lines)}")
    width = max(len(line) for line in lines)
    matrix = np.full((16, width), catalog.id_for_name("empty"), dtype=np.int16)
    for y, line in enumerate(lines):
        padded = line.ljust(width, "-")
        for x, glyph in enumerate(padded):
            name = corpus_glyph_to_name(glyph)
            matrix[y, x] = catalog.id_for_name(name)
    level_id = f"{path.parent.name}/{path.stem}" if path.parent.name != "mario" else path.stem
    return LevelGrid(
        tiles=matrix,
        width_tiles=width,
        height_tiles=16,
        tile_size_px=tile_size_px,
        area_type=AreaType.OVERWORLD,
        metadata={"level_id": level_id, "source_path": str(path)},
    )
