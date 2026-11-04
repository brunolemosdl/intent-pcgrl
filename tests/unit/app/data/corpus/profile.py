from __future__ import annotations

import numpy as np
from pcgrl.app.data.corpus.profile import extract_corpus_profile
from pcgrl.core.domain.level import AreaType, LevelGrid


def _level(config, catalog) -> LevelGrid:
    """Synthetic overworld level: full floor with one width-3 gap, a width-4
    floating brick platform 4 rows above the floor, a coin 2 above the floor and a
    coin over the gap (arc, no support)."""
    h, w = 14, 24
    empty = catalog.id_for_name("empty")
    solid = catalog.id_for_name("solid")
    brick = catalog.id_for_name("brick")
    coin = catalog.id_for_name("coin")
    t = np.full((h, w), empty, dtype=np.int16)
    t[h - 1, :] = solid
    for c in (5, 6, 7):  # width-3 floor gap
        t[h - 1, c] = empty
    for c in range(10, 14):  # floating platform at row 9 (4 above floor)
        t[9, c] = brick
    t[11, 2] = coin  # 2 above floor, supported
    t[8, 6] = coin  # over the gap column -> arc, no support below
    return LevelGrid(
        tiles=t, width_tiles=w, height_tiles=h, tile_size_px=16, area_type=AreaType.OVERWORLD
    )


def test_extract_profile_basic_shapes(config, catalog) -> None:
    """Verify extract profile basic shapes."""
    prof = extract_corpus_profile([_level(config, catalog)], catalog, config)
    assert prof["n_levels"] == 1
    # floating platform sits 4 above the floor.
    assert prof["height_above_surface"]["floating_platform"]["pcts"]["p50"] == 4.0
    # one coin is an arc over the gap (no support).
    assert prof["height_above_surface"]["coin"]["arc_over_pit_frac"] > 0.0
    # the supported coin is 2 above the floor.
    assert 2.0 in {
        prof["height_above_surface"]["coin"]["pcts"]["p50"],
        prof["height_above_surface"]["coin"]["pcts"]["max"],
    }
    # the floor gap is width 3.
    assert prof["gaps"]["width"]["pcts"]["max"] == 3.0
    # densities are present and non-negative.
    for key in ("enemy", "coin", "pipe", "platform", "block"):
        assert prof["density"][key]["max"] >= 0.0


def test_extract_profile_skips_non_overworld(config, catalog) -> None:
    """Verify extract profile skips non overworld."""
    # A level with no solid ground row is skipped (not overworld-like).
    h, w = 14, 20
    t = np.full((h, w), catalog.id_for_name("empty"), dtype=np.int16)
    sky = LevelGrid(
        tiles=t, width_tiles=w, height_tiles=h, tile_size_px=16, area_type=AreaType.OVERWORLD
    )
    prof = extract_corpus_profile([sky], catalog, config)
    assert prof["n_levels"] == 0
