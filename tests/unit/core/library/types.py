from __future__ import annotations

import numpy as np
import pytest
from pcgrl.core.library.types import Chunk, ChunkFamily, ChunkLibrary


def _chunk(
    chunk_id: int,
    *,
    family: ChunkFamily = ChunkFamily.MIDDLE,
    level_id: str = "1-1",
    width: int = 16,
) -> Chunk:
    """Chunk."""
    rng = np.random.default_rng(chunk_id)
    return Chunk(
        chunk_id=chunk_id,
        tiles=rng.integers(0, 5, size=(14, width), dtype=np.int16),
        family=family,
        level_id=level_id,
        column_offset=chunk_id * 8,
        entry_height=2,
        exit_height=3,
        gap_at_left=False,
        gap_at_right=bool(chunk_id % 2),
        category_counts={"enemy": 1, "coin": 4},
    )


def _library(chunks: list[Chunk]) -> ChunkLibrary:
    """library."""
    return ChunkLibrary(
        chunks,
        train_level_ids=["1-1", "2-1"],
        heldout_level_ids=["8-1"],
        tile_names=["empty", "solid", "brick", "question", "coin"],
        metadata={"stride_tiles": 8},
    )


def test_chunk_dimensions() -> None:
    """Verify chunk dimensions."""
    chunk = _chunk(0, width=8)
    assert chunk.width == 8
    assert chunk.height == 14


def test_library_roundtrip(tmp_path) -> None:
    """Verify library roundtrip."""
    original = _library(
        [
            _chunk(0, family=ChunkFamily.START),
            _chunk(1),
            _chunk(2, family=ChunkFamily.ENDING, level_id="2-1", width=8),
        ]
    )
    original.save(tmp_path)
    loaded = ChunkLibrary.load(tmp_path)
    assert len(loaded) == 3
    assert loaded.train_level_ids == ["1-1", "2-1"]
    assert loaded.heldout_level_ids == ["8-1"]
    assert loaded.tile_names[1] == "solid"
    assert loaded.metadata["stride_tiles"] == 8
    for before, after in zip(original.chunks, loaded.chunks, strict=True):
        assert np.array_equal(before.tiles, after.tiles)
        assert after.tiles.dtype == np.int16
        assert before.annotations_dict() == after.annotations_dict()


def test_library_by_family_and_widths() -> None:
    """Verify library by family and widths."""
    library = _library(
        [
            _chunk(0, family=ChunkFamily.START),
            _chunk(1, width=8),
            _chunk(2, family=ChunkFamily.ENDING, level_id="2-1"),
        ]
    )
    assert [c.chunk_id for c in library.by_family(ChunkFamily.START)] == [0]
    assert [c.chunk_id for c in library.by_family(ChunkFamily.MIDDLE)] == [1]
    assert library.widths() == (8, 16)


def test_library_rejects_non_contiguous_ids() -> None:
    """Verify library rejects heldout leakage."""
    with pytest.raises(ValueError, match="contiguous"):
        _library([_chunk(0), _chunk(2)])


def test_library_rejects_heldout_leakage() -> None:
    """Verify library rejects heldout leakage."""
    with pytest.raises(ValueError, match="heldout"):
        _library([_chunk(0, level_id="8-1")])


def test_library_rejects_overlapping_splits() -> None:
    """Verify library rejects overlapping splits."""
    with pytest.raises(ValueError, match="both splits"):
        ChunkLibrary(
            [_chunk(0)],
            train_level_ids=["1-1"],
            heldout_level_ids=["1-1"],
            tile_names=["empty"],
        )


def test_load_missing_directory_raises(tmp_path) -> None:
    """Verify load missing directory raises."""
    with pytest.raises(FileNotFoundError, match="library"):
        ChunkLibrary.load(tmp_path / "nope")


def test_load_rejects_unknown_format_version(tmp_path) -> None:
    """Verify load rejects unknown format version."""
    library = _library([_chunk(0)])
    _, json_path = library.save(tmp_path)
    payload = json_path.read_text(encoding="utf-8").replace(
        '"format_version": 1', '"format_version": 99'
    )
    json_path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="format_version"):
        ChunkLibrary.load(tmp_path)
