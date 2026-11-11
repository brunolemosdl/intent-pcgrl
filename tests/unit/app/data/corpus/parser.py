from __future__ import annotations

from pathlib import Path

import pytest
from pcgrl.app.data.corpus.parser import parse_level_file
from pcgrl.core.domain.level import AreaType


def test_parse_level_file_overworld_area_type(config, catalog) -> None:
    """Verify parse level file overworld area type."""
    corpus_root = Path("data/raw/corpus/mario")
    if not corpus_root.is_dir():
        pytest.skip("Corpus not available")
    level_file = next(corpus_root.rglob("*.txt"))
    level = parse_level_file(level_file, catalog, tile_size_px=16)
    assert level.area_type == AreaType.OVERWORLD


def test_parse_level_file_dimensions(config, catalog) -> None:
    """Verify parse level file dimensions."""
    corpus_root = Path("data/raw/corpus/mario")
    if not corpus_root.is_dir():
        pytest.skip("Corpus not available")
    level_file = next(corpus_root.rglob("*.txt"))
    level = parse_level_file(level_file, catalog, tile_size_px=16)
    assert level.height_tiles == 16
    assert level.width_tiles > 0


def test_parse_level_file_metadata_level_id(config, catalog) -> None:
    """Verify parse level file metadata level id."""
    corpus_root = Path("data/raw/corpus/mario")
    if not corpus_root.is_dir():
        pytest.skip("Corpus not available")
    level_file = next(corpus_root.rglob("*.txt"))
    level = parse_level_file(level_file, catalog, tile_size_px=16)
    assert "level_id" in level.metadata


def test_parse_level_file_tiles_are_int16(config, catalog) -> None:
    """Verify parse level file tiles are int16."""
    corpus_root = Path("data/raw/corpus/mario")
    if not corpus_root.is_dir():
        pytest.skip("Corpus not available")
    level_file = next(corpus_root.rglob("*.txt"))
    level = parse_level_file(level_file, catalog, tile_size_px=16)
    assert level.tiles.dtype.name == "int16"
