from __future__ import annotations

import numpy as np
import pytest
from pcgrl.app.data.library.extract import (
    ChunkWindow,
    cap_windows,
    deduplicate_windows,
    extract_windows,
    parse_raw_level,
    split_levels_for_heldout,
)
from pcgrl.core.library.types import ChunkFamily


@pytest.fixture
def raw_level(tmp_path, catalog):
    """Raw level."""
    # 14-row VGLC-style level, 40 columns: flat floor with one structure.
    rows = ["-" * 40 for _ in range(11)]
    rows.append("----SS?S" + "-" * 32)
    rows.append("X" * 40)
    rows.append("X" * 40)
    path = tmp_path / "1-1.txt"
    path.write_text("\n".join(rows) + "\n")
    return parse_raw_level(path, catalog)


def test_parse_raw_level_keeps_floor_on_last_row(raw_level, catalog) -> None:
    """Verify parse raw level keeps floor on last row."""
    # Regression for the alignment bug: the old parser padded EMPTY rows below
    # the floor; raw parsing must keep the native height with ground at bottom.
    assert raw_level.tiles.shape == (14, 40)
    solid = catalog.id_for_name("solid")
    assert (raw_level.tiles[-1] == solid).all()
    assert (raw_level.tiles[-2] == solid).all()
    assert raw_level.level_id.endswith("/1-1")


def test_extract_windows_families_and_provenance(raw_level) -> None:
    """Verify extract windows families and provenance."""
    windows = extract_windows(
        raw_level, chunk_widths=[16], stride_tiles=8, ending_columns=16
    )
    offsets = [w.column_offset for w in windows]
    assert offsets == [0, 8, 24]
    assert windows[0].family == ChunkFamily.START
    assert windows[1].family == ChunkFamily.MIDDLE
    assert windows[-1].family == ChunkFamily.ENDING
    assert windows[-1].column_offset == 40 - 16
    for window in windows:
        assert window.tiles.shape == (14, 16)
        assert np.array_equal(
            window.tiles,
            raw_level.tiles[:, window.column_offset : window.column_offset + 16],
        )


def test_extract_windows_skips_partial_endings(raw_level) -> None:
    """Verify extract windows skips partial endings."""
    # A middle window ending past width-ending_columns would contain a partial
    # staircase; offset 16 spans [16, 32) > 24 with ending_columns=16, so only
    # offsets 0, 8 and the final window survive.
    windows = extract_windows(
        raw_level, chunk_widths=[16], stride_tiles=8, ending_columns=16
    )
    assert all(
        w.column_offset + 16 <= 24 or w.family != ChunkFamily.MIDDLE for w in windows
    )


def test_deduplicate_windows_keeps_first_provenance(raw_level) -> None:
    """Verify deduplicate windows keeps first provenance."""
    windows = extract_windows(
        raw_level, chunk_widths=[16], stride_tiles=8, ending_columns=16
    )
    flat_a = next(w for w in windows if w.column_offset == 8)
    duplicate = ChunkWindow(
        tiles=flat_a.tiles.copy(),
        family=ChunkFamily.MIDDLE,
        level_id="other/9-9",
        column_offset=99,
    )
    unique = deduplicate_windows([*windows, duplicate])
    assert len(unique) == len(windows)
    assert all(w.level_id != "other/9-9" for w in unique)


def test_split_levels_for_heldout_is_disjoint_and_deterministic() -> None:
    """Verify split levels for heldout is disjoint and deterministic."""
    ids = [f"mario-1/{i}-1" for i in range(1, 11)]
    train_a, heldout_a = split_levels_for_heldout(ids, heldout_fraction=0.2, seed=42)
    train_b, heldout_b = split_levels_for_heldout(ids, heldout_fraction=0.2, seed=42)
    assert train_a == train_b
    assert heldout_a == heldout_b
    assert len(heldout_a) == 2
    assert not set(train_a) & set(heldout_a)
    assert sorted(train_a + heldout_a) == sorted(ids)


def test_split_levels_keeps_at_least_one_heldout() -> None:
    """Verify split levels keeps at least one heldout."""
    train, heldout = split_levels_for_heldout(
        ["a", "b", "c"], heldout_fraction=0.1, seed=0
    )
    assert len(heldout) == 1
    assert len(train) == 2


def test_cap_windows_protects_start_and_ending(raw_level) -> None:
    """Verify cap windows protects start and ending."""
    windows = extract_windows(
        raw_level, chunk_widths=[16], stride_tiles=2, ending_columns=4
    )
    capped = cap_windows(windows, max_chunks=4, seed=7)
    assert len(capped) == 4
    families = [w.family for w in capped]
    assert ChunkFamily.START in families
    assert ChunkFamily.ENDING in families
