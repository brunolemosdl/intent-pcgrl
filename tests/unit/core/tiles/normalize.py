from __future__ import annotations

import pytest
from pcgrl.core.tiles.normalize import (
    CORPUS_TO_CANONICAL,
    canonical_to_engine,
    corpus_glyph_to_name,
)


def test_corpus_empty_glyph() -> None:
    """Verify corpus empty glyph."""
    assert corpus_glyph_to_name("-") == "empty"


def test_corpus_enemy_glyph() -> None:
    """Verify corpus enemy glyph."""
    assert corpus_glyph_to_name("E") == "goomba"


def test_corpus_solid_glyph() -> None:
    """Verify corpus solid glyph."""
    assert CORPUS_TO_CANONICAL["X"] == "solid"


def test_corpus_pipe_glyphs() -> None:
    """Verify corpus pipe glyphs."""
    assert corpus_glyph_to_name("<") == "pipe_top_left"
    assert corpus_glyph_to_name(">") == "pipe_top_right"
    assert corpus_glyph_to_name("[") == "pipe_body_left"
    assert corpus_glyph_to_name("]") == "pipe_body_right"


def test_corpus_coin_glyph() -> None:
    """Verify corpus coin glyph."""
    assert corpus_glyph_to_name("o") == "coin"


def test_corpus_cannon_glyphs_map_to_solid() -> None:
    """Verify corpus cannon glyphs map to solid."""
    assert corpus_glyph_to_name("B") == "solid"
    assert corpus_glyph_to_name("b") == "solid"


def test_unknown_vglc_glyph_raises() -> None:
    """Verify unknown vglc glyph raises."""
    with pytest.raises(KeyError):
        corpus_glyph_to_name("Z")


def test_engine_translation_goomba() -> None:
    """Verify engine translation goomba."""
    assert canonical_to_engine("goomba") == "g"


def test_engine_translation_solid() -> None:
    """Verify engine translation solid."""
    assert canonical_to_engine("solid") == "X"


def test_engine_translation_empty() -> None:
    """Verify canonical to engine covers corpus keys."""
    assert canonical_to_engine("empty") == "-"


def test_engine_translation_flagpole() -> None:
    """Verify engine translation flagpole."""
    assert canonical_to_engine("flagpole") == "F"


def test_engine_unknown_returns_dash() -> None:
    """Verify engine unknown returns dash."""
    assert canonical_to_engine("nonexistent_tile_xyz") == "-"


def test_canonical_to_engine_covers_corpus_keys() -> None:
    """Verify canonical to engine covers corpus keys."""
    for _glyph, canonical in CORPUS_TO_CANONICAL.items():
        result = canonical_to_engine(canonical)
        assert isinstance(result, str) and len(result) == 1
