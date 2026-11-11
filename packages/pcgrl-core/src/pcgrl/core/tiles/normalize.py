from __future__ import annotations

CORPUS_TO_CANONICAL: dict[str, str] = {
    "-": "empty",
    "X": "solid",
    "S": "brick",
    # '?' = FULL question block (holds an item) -> lucky block (reward + solid).
    "?": "question_block_coin",
    # 'Q' = EMPTY question block (already used, no item). Distinct tile, still a
    # solid question-block sprite, but NOT a reward (so it is not a lucky block).
    "Q": "question_block_empty",
    "E": "goomba",
    "<": "pipe_top_left",
    ">": "pipe_top_right",
    "[": "pipe_body_left",
    "]": "pipe_body_right",
    "o": "coin",
    "B": "solid",
    "b": "solid",
}

CANONICAL_TO_ENGINE: dict[str, str] = {
    "empty": "-",
    "solid": "X",
    "brick": "S",
    "used_block": "D",
    "coin": "o",
    "goomba": "g",
    # Engine glyphs match the corpus so the renderer shows both as question
    # blocks (full '?' -> item sprite, empty 'Q' -> used sprite).
    "question_block_coin": "?",
    "question_block_empty": "Q",
    "pipe_top_left": "<",
    "pipe_top_right": ">",
    "pipe_body_left": "[",
    "pipe_body_right": "]",
    "player": "M",
    "flagpole": "F",
}


def corpus_glyph_to_name(glyph: str) -> str:
    """Map a VGLC corpus glyph to a canonical tile name.

    Args:
        glyph: Single-character VGLC glyph.

    Returns:
        Canonical tile name.

    Raises:
        KeyError: When the glyph is not recognised.
    """
    if glyph not in CORPUS_TO_CANONICAL:
        raise KeyError(f"Unknown VGLC glyph: {glyph!r}")
    return CORPUS_TO_CANONICAL[glyph]


def canonical_to_engine(name: str) -> str:
    """Map a canonical tile name to a Mario-bridge engine glyph.

    Args:
        name: Canonical tile name.

    Returns:
        Engine glyph, defaulting to empty (``-``) for unmapped names.
    """
    return CANONICAL_TO_ENGINE.get(name, "-")
