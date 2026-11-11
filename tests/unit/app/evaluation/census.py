from __future__ import annotations

import numpy as np
import pytest
from pcgrl.app.evaluation.census import (
    census_distance,
    diversity_metrics,
    merge_censuses,
    structural_census,
)
from pcgrl.core.domain.level import AreaType, LevelGrid
from pcgrl.core.tiles.normalize import corpus_glyph_to_name


def _level(rows: list[str], catalog, metadata=None) -> LevelGrid:
    """level."""
    width = max(len(row) for row in rows)
    tiles = np.full((len(rows), width), catalog.id_for_name("empty"), dtype=np.int16)
    for y, row in enumerate(rows):
        for x, glyph in enumerate(row.ljust(width, "-")):
            tiles[y, x] = catalog.id_for_name(corpus_glyph_to_name(glyph))
    return LevelGrid(
        tiles=tiles,
        width_tiles=width,
        height_tiles=len(rows),
        tile_size_px=16,
        area_type=AreaType.OVERWORLD,
        metadata=metadata or {},
    )


@pytest.fixture
def structured_level(catalog):
    """Structured level."""
    rows = [
        "------------------------",
        "----SSS-----------------",
        "------------<>----------",
        "------------[]----------",
        "XXXX--XXXXXXXXXXXX--XXXX",
        "XXXX--XXXXXXXXXXXX--XXXX",
    ]
    return _level(rows, catalog)


def test_structural_census_histograms(structured_level, catalog, config) -> None:
    """Verify structural census histograms."""
    census = structural_census(structured_level, catalog, config)
    assert census["gap_widths"] == {2: 2}
    assert 3 in census["platform_lengths"]
    assert census["pipe_heights"] == {2: 1}
    assert census["isolated_block_rate"] == 0.0
    assert len(census["content_thirds"]) == 3


def test_isolated_blocks_detected(catalog, config) -> None:
    """Verify isolated blocks detected."""
    rows = [
        "--S---S---S-",
        "------------",
        "XXXXXXXXXXXX",
    ]
    census = structural_census(_level(rows, catalog), catalog, config)
    assert census["isolated_block_rate"] == 1.0


def test_census_distance_zero_for_identical(structured_level, catalog, config) -> None:
    """Verify census distance zero for identical."""
    census = structural_census(structured_level, catalog, config)
    distances = census_distance(census, census)
    assert all(value == 0.0 for value in distances.values())


def test_census_distance_positive_for_different(structured_level, catalog, config) -> None:
    """Verify census distance positive for different."""
    flat = _level(["-" * 24, "X" * 24, "X" * 24], catalog)
    distance = census_distance(
        structural_census(structured_level, catalog, config),
        structural_census(flat, catalog, config),
    )
    # The flat level has an EMPTY gap histogram; total-variation against an
    # empty distribution carries half mass by convention.
    assert distance["gap_widths"] == 0.5


def test_merge_censuses_accumulates(structured_level, catalog, config) -> None:
    """Verify diversity metrics duplicates."""
    census = structural_census(structured_level, catalog, config)
    merged = merge_censuses([census, census])
    assert merged["gap_widths"][2] == 4
    assert merged["isolated_block_rate"] == census["isolated_block_rate"]


def test_diversity_metrics_duplicates(catalog) -> None:
    """Verify diversity metrics duplicates."""
    base = ["----", "XXXX"]
    a = _level(base, catalog, metadata={"chunk_ids": [0, 1, 2]})
    b = _level(base, catalog, metadata={"chunk_ids": [0, 1, 2]})
    c = _level(["S---", "XXXX"], catalog, metadata={"chunk_ids": [0, 2, 1]})
    metrics = diversity_metrics([a, b, c])
    assert metrics["duplicate_sequence_rate"] == pytest.approx(1 / 3, abs=1e-3)
    assert metrics["pairwise_tile_distance"] > 0.0
    assert diversity_metrics([a])["pairwise_tile_distance"] == 0.0
