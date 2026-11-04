from __future__ import annotations

import numpy as np
import pytest
from pcgrl.app.data.library.annotate import (
    annotate_window,
    count_categories,
    surface_height,
)
from pcgrl.app.data.library.extract import ChunkWindow
from pcgrl.core.library.types import ChunkFamily


def _tiles_from_rows(rows: list[str], catalog) -> np.ndarray:
    """ tiles from rows."""
    from pcgrl.core.tiles.normalize import corpus_glyph_to_name

    width = max(len(row) for row in rows)
    tiles = np.full((len(rows), width), catalog.id_for_name("empty"), dtype=np.int16)
    for y, row in enumerate(rows):
        for x, glyph in enumerate(row.ljust(width, "-")):
            tiles[y, x] = catalog.id_for_name(corpus_glyph_to_name(glyph))
    return tiles


@pytest.fixture
def stair_tiles(catalog):
    """Stair tiles."""
    # Column 0: floor 2 high; column 3: stair block adds a third row;
    # column 7: bottomless gap.
    rows = [
        "--------",
        "---X----",
        "XXXXXXX-",
        "XXXXXXX-",
    ]
    return _tiles_from_rows(rows, catalog)


def test_surface_height_flat_stair_and_gap(stair_tiles, catalog) -> None:
    """Verify surface height flat stair and gap."""
    assert surface_height(stair_tiles, 0, catalog) == 2
    assert surface_height(stair_tiles, 3, catalog) == 3
    assert surface_height(stair_tiles, 7, catalog) == 0


def test_surface_height_ignores_floating_structures(catalog) -> None:
    """Verify surface height ignores floating structures."""
    rows = [
        "SS--",
        "----",
        "XXXX",
    ]
    tiles = _tiles_from_rows(rows, catalog)
    # The floating bricks at the top do not extend the standable ground.
    assert surface_height(tiles, 0, catalog) == 1


def test_count_categories_excludes_empty(catalog) -> None:
    """Verify count categories excludes empty."""
    rows = [
        "o-E-",
        "S?--",
        "XXXX",
    ]
    counts = count_categories(_tiles_from_rows(rows, catalog), catalog)
    assert counts["coin"] >= 1
    assert counts["enemy"] == 1
    # 'S' (brick) and '?' (question block) are solid_like in the catalog, so
    # they fold into the solid count alongside the 4 floor tiles.
    assert counts["solid"] == 6
    assert "empty" not in counts


def test_annotate_window_builds_chunk(stair_tiles, catalog) -> None:
    """Verify annotate window builds chunk."""
    window = ChunkWindow(
        tiles=stair_tiles,
        family=ChunkFamily.MIDDLE,
        level_id="mario-1/1-1",
        column_offset=24,
    )
    chunk = annotate_window(window, 5, catalog)
    assert chunk.chunk_id == 5
    assert chunk.level_id == "mario-1/1-1"
    assert chunk.column_offset == 24
    assert chunk.entry_height == 2
    assert chunk.exit_height == 0
    assert not chunk.gap_at_left
    assert chunk.gap_at_right
    assert chunk.category_counts["solid"] > 0
