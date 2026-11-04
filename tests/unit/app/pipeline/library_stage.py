from __future__ import annotations

import pytest
from pcgrl.core.helpers.path import resolve_path
from pcgrl.core.library.types import ChunkFamily, ChunkLibrary
from pcgrl.core.settings.loader import load_config


@pytest.fixture
def library_config(tmp_path):
    """Library config."""
    override = tmp_path / "override.yaml"
    override.write_text(
        f"library:\n  output_root: {tmp_path}/library\n  heldout_level_fraction: 0.2\n"
    )
    return load_config(training="initial", override_path=override, apply_profile=False)


def _require_corpus(config) -> None:
    """ require corpus."""
    corpus_root = resolve_path(config.dataset.corpus_root)
    if not corpus_root.is_dir() or not any(corpus_root.rglob("*.txt")):
        pytest.skip("Corpus files not present")


def test_run_library_end_to_end(library_config) -> None:
    """Verify run library end to end."""
    _require_corpus(library_config)
    from pcgrl.app.pipeline.library import run_library

    output_root = run_library(library_config)
    library = ChunkLibrary.load(output_root)
    assert 0 < len(library) <= library_config.library.max_chunks
    assert library.by_family(ChunkFamily.START)
    assert library.by_family(ChunkFamily.MIDDLE)
    assert library.by_family(ChunkFamily.ENDING)
    # Split is by LEVEL: nothing in the library may come from heldout levels.
    heldout = set(library.heldout_level_ids)
    assert heldout
    assert all(chunk.level_id not in heldout for chunk in library.chunks)
    # SMB1-only by default: chunks must come from the mario-1 set.
    assert all(chunk.level_id.startswith("mario-1/") for chunk in library.chunks)
    # Raw alignment fix: chunks keep the native 14-row height, and the old
    # processed-npz corruption signature (full floor at row 11 with two empty
    # rows BELOW it) never appears. All-empty bottoms are still legal — the
    # athletic levels (1-3 etc.) run over bottomless pits.
    empty_id = library.tile_names.index("empty")
    solid_id = library.tile_names.index("solid")
    for chunk in library.chunks:
        assert chunk.height == 14
        misaligned_floor = (chunk.tiles[11] == solid_id).all() and (
            chunk.tiles[12:] == empty_id
        ).all()
        assert not misaligned_floor
    # And ground-floor levels do put solid on the LAST row.
    assert any((chunk.tiles[-1] == solid_id).any() for chunk in library.chunks)


def test_run_library_is_deterministic(library_config) -> None:
    """Verify run library is deterministic."""
    _require_corpus(library_config)
    from pcgrl.app.pipeline.library import build_chunk_library

    first = build_chunk_library(library_config)
    second = build_chunk_library(library_config)
    assert len(first) == len(second)
    assert first.heldout_level_ids == second.heldout_level_ids
    for a, b in zip(first.chunks, second.chunks, strict=True):
        assert a.annotations_dict() == b.annotations_dict()
