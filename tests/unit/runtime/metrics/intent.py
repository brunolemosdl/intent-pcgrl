from __future__ import annotations

import numpy as np
import pytest
from pcgrl.core.domain.level import AreaType, LevelGrid
from pcgrl.core.tiles.normalize import corpus_glyph_to_name
from pcgrl.runtime.metrics.intent import (
    INTENT_KEYS,
    compute_gap_difficulty,
    compute_intent_vector,
    compute_path_smoothness,
    compute_platform_usage,
    compute_verticality,
    gap_widths,
    ground_height_profile,
    traversable_path_envelope,
)


def _level(rows: list[str], catalog) -> LevelGrid:
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
        metadata={},
    )


def _flat_level(catalog, width: int = 32, sky_rows: int = 12) -> LevelGrid:
    """flat level."""
    rows = ["-" * width for _ in range(sky_rows)]
    rows += ["X" * width, "X" * width]
    return _level(rows, catalog)


def test_ground_height_profile_flat_and_gap(catalog, config) -> None:
    """Verify ground height profile flat and gap."""
    rows = [
        "--------",
        "XXX--XXX",
        "XXX--XXX",
    ]
    level = _level(rows, catalog)
    assert ground_height_profile(level, catalog, config) == [2, 2, 2, 0, 0, 2, 2, 2]


def test_gap_widths_groups_runs() -> None:
    """Verify gap widths groups runs."""
    assert gap_widths([2, 0, 0, 2, 0, 2, 2]) == [2, 1]
    assert gap_widths([2, 2]) == []
    assert gap_widths([0, 0, 0]) == [3]


def test_gap_difficulty_zero_without_gaps() -> None:
    """Verify gap difficulty zero without gaps."""
    assert compute_gap_difficulty([2] * 32, jump_reach=5, width_tiles=32) == 0.0


def test_gap_difficulty_monotonic_in_width_and_count() -> None:
    """Verify gap difficulty monotonic in width and count."""
    base = [2] * 48
    one_small = base.copy()
    one_small[10:12] = [0, 0]
    one_wide = base.copy()
    one_wide[10:16] = [0] * 6
    two_wide = one_wide.copy()
    two_wide[30:36] = [0] * 6
    small = compute_gap_difficulty(one_small, jump_reach=5, width_tiles=48)
    wide = compute_gap_difficulty(one_wide, jump_reach=5, width_tiles=48)
    double = compute_gap_difficulty(two_wide, jump_reach=5, width_tiles=48)
    assert 0.0 < small < wide < double <= 1.0


def test_envelope_includes_floating_platforms(catalog, config) -> None:
    """Verify envelope includes floating platforms."""
    # Floor 2-high; a brick platform 3 rows above the floor surface, reachable
    # with jump_height=4. Envelope must register the platform height (5); a
    # ground-stack-only surface would not.
    rows = [
        "----------------",
        "----------------",
        "----------------",
        "-----SSS--------",
        "----------------",
        "----------------",
        "XXXXXXXXXXXXXXXX",
        "XXXXXXXXXXXXXXXX",
    ]
    level = _level(rows, catalog)
    envelope = traversable_path_envelope(level, catalog, config)
    assert envelope[0] == 2
    assert max(h for h in envelope if h is not None) == 5
    assert envelope[6] == 5


def test_verticality_flat_vs_platforms(catalog, config) -> None:
    """Verify verticality flat vs platforms."""
    flat = traversable_path_envelope(_flat_level(catalog), catalog, config)
    assert compute_verticality(flat, 14) == 0.0
    assert compute_verticality([2, 2, 8, 8], 14) == pytest.approx(6 / 12)
    assert compute_verticality([None, None], 14) == 0.0


def test_path_smoothness_flat_is_one() -> None:
    """Verify path smoothness flat is one."""
    assert compute_path_smoothness([2] * 10, jump_height=4) == 1.0
    rough = [2, 6, 2, 6, 2, 6]
    assert compute_path_smoothness(rough, jump_height=4) == 0.0
    mild = [2, 3, 2, 3, 2]
    assert 0.0 < compute_path_smoothness(mild, jump_height=4) < 1.0


def test_platform_usage_counts_elevated_columns() -> None:
    """Verify platform usage counts elevated columns."""
    envelope = [2, 2, 5, 5, None, 2]
    ground = [2, 2, 2, 2, 0, 2]
    assert compute_platform_usage(envelope, ground) == pytest.approx(2 / 5)
    assert compute_platform_usage([None, None], [0, 0]) == 0.0


def test_intent_vector_shape_and_range(catalog, config) -> None:
    """Verify intent vector shape and range."""
    rows = [
        "----------------",
        "--o-------------",
        "--?S------------",
        "----------------",
        "----E-----------",
        "XXXXXX--XXXXXXXX",
        "XXXXXX--XXXXXXXX",
    ]
    level = _level(rows, catalog)
    vector = compute_intent_vector(level, catalog, config)
    assert vector.shape == (len(INTENT_KEYS),)
    assert ((vector >= 0.0) & (vector <= 1.0)).all()
    names = dict(zip(INTENT_KEYS, vector.tolist(), strict=True))
    assert names["enemy_density"] > 0.0
    assert names["coin_density"] > 0.0
    assert names["interactive_block_density"] > 0.0
    assert names["pipe_density"] == 0.0
    assert names["gap_difficulty"] > 0.0


def test_intent_vector_empty_level_is_mostly_zero(catalog, config) -> None:
    """Verify intent vector empty level is mostly zero."""
    vector = compute_intent_vector(_flat_level(catalog), catalog, config)
    names = dict(zip(INTENT_KEYS, vector.tolist(), strict=True))
    assert names["enemy_density"] == 0.0
    assert names["verticality"] == 0.0
    assert names["platform_usage"] == 0.0
    assert names["path_smoothness"] == 1.0


def test_compute_verticality_band_renormalisation() -> None:
    """The viability band stretches the compressed raw range over [0, 1]."""
    from pcgrl.runtime.metrics.intent import compute_verticality

    # Raw ratio 6/12 = 0.5 with band [0.41, 0.92] -> (0.5-0.41)/0.51.
    envelope = [2, 2, 8, 8]
    value = compute_verticality(envelope, 14, band_min=0.41, band_max=0.92)
    assert value == pytest.approx((0.5 - 0.41) / 0.51, abs=1e-6)
    # Below the band floor reads as flat; above the ceiling saturates.
    assert compute_verticality([2, 2, 4, 4], 14, band_min=0.41, band_max=0.92) == 0.0
    assert compute_verticality([0, 12], 14, band_min=0.41, band_max=0.92) == 1.0


def test_effective_target_reads_the_none_band_as_zero() -> None:
    """Count targets at or below 0.05 become zero; the rest is untouched."""
    from pcgrl.runtime.metrics.intent import effective_target

    target = np.array([0.05, 0.04, 0.06, 0.05, 0.0, 0.05, 0.95, 0.05], dtype=np.float32)
    effective = effective_target(target)
    assert effective.tolist() == pytest.approx([0.0, 0.0, 0.06, 0.0, 0.0, 0.05, 0.95, 0.05])
    assert target[0] == pytest.approx(0.05), "the input must not be modified"
    assert effective_target(effective).tolist() == effective.tolist()


def test_measure_intent_exposes_the_path_geometry(catalog, config) -> None:
    """The measurement carries the envelope the vector was derived from."""
    from pcgrl.runtime.metrics.intent import measure_intent

    level = _flat_level(catalog)
    state = measure_intent(level, catalog, config)
    assert np.array_equal(state.vector, compute_intent_vector(level, catalog, config))
    assert len(state.envelope) == level.width_tiles == len(state.ground)
    assert state.amplitude == 0
    assert state.path_columns == sum(1 for height in state.envelope if height is not None)
