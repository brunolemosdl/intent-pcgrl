from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pcgrl.core.library.types import ChunkFamily
from pcgrl.core.tiles.catalog import TileCatalog
from pcgrl.core.tiles.normalize import corpus_glyph_to_name


@dataclass(frozen=True)
class RawLevel:
    """Corpus level parsed at native height with provenance id.

    Attributes:
        level_id: Stable ``set/stem`` identifier for heldout splitting.
        tiles: ``(height, width)`` tile-id matrix at native corpus height.
    """

    level_id: str
    tiles: np.ndarray


@dataclass(frozen=True)
class ChunkWindow:
    """Sliding window cut from a raw level before annotation.

    Attributes:
        tiles: Window tile slice at a fixed chunk width.
        family: Structural family tag (START, MIDDLE, or ENDING).
        level_id: Provenance id of the source level.
        column_offset: Horizontal offset of the window in the source level.
    """

    tiles: np.ndarray
    family: ChunkFamily
    level_id: str
    column_offset: int


def parse_raw_level(path: Path, catalog: TileCatalog) -> RawLevel:
    """Parse a VGLC text level at native height with provenance id.

    Args:
        path: Path to a ``.txt`` VGLC level file.
        catalog: Tile catalog for glyph-to-id resolution.

    Returns:
        Parsed ``RawLevel`` preserving native row count.

    Raises:
        ValueError: When the file is empty after stripping blank lines.
        OSError: When the level file cannot be read.
    """
    lines = [
        line.rstrip("\n")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() != ""
    ]
    if not lines:
        raise ValueError(f"{path} is empty")
    width = max(len(line) for line in lines)
    tiles = np.full((len(lines), width), catalog.id_for_name("empty"), dtype=np.int16)
    for y, line in enumerate(lines):
        for x, glyph in enumerate(line.ljust(width, "-")):
            tiles[y, x] = catalog.id_for_name(corpus_glyph_to_name(glyph))
    return RawLevel(level_id=f"{path.parent.name}/{path.stem}", tiles=tiles)


def load_raw_levels(
    corpus_root: Path, level_sets: list[str], catalog: TileCatalog
) -> list[RawLevel]:
    """Parse every level of the requested corpus sets.

    Args:
        corpus_root: Root directory containing level-set subfolders.
        level_sets: Subdirectory names to scan for ``.txt`` levels.
        catalog: Tile catalog for glyph-to-id resolution.

    Returns:
        Parsed raw levels from all requested sets.

    Raises:
        FileNotFoundError: When a level set directory has no level files.
    """
    levels: list[RawLevel] = []
    for level_set in level_sets:
        set_dir = corpus_root / level_set
        files = sorted(f for f in set_dir.glob("*.txt") if f.stem != "symbols")
        if not files:
            raise FileNotFoundError(f"No corpus levels found in {set_dir}")
        levels.extend(parse_raw_level(path, catalog) for path in files)
    return levels


def split_levels_for_heldout(
    level_ids: list[str], *, heldout_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Split level ids into train and heldout sets deterministically.

    Args:
        level_ids: Unique corpus level identifiers.
        heldout_fraction: Fraction of levels reserved for heldout evaluation.
        seed: RNG seed for the shuffle.

    Returns:
        Tuple of sorted train ids and sorted heldout ids.
    """
    ids = sorted(set(level_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    heldout_count = round(len(ids) * heldout_fraction)
    if heldout_fraction > 0.0 and heldout_count == 0 and len(ids) > 1:
        heldout_count = 1
    heldout = sorted(ids[:heldout_count])
    train = sorted(ids[heldout_count:])
    return train, heldout


def extract_windows(
    level: RawLevel,
    *,
    chunk_widths: list[int],
    stride_tiles: int,
    ending_columns: int,
) -> list[ChunkWindow]:
    """Cut sliding windows from a level and tag START/MIDDLE/ENDING families.

    Args:
        level: Source raw level to window.
        chunk_widths: Chunk widths to extract.
        stride_tiles: Horizontal stride between window offsets.
        ending_columns: Trailing columns reserved for the ENDING family.

    Returns:
        Chunk windows with structural family tags.
    """
    level_width = int(level.tiles.shape[1])
    windows: list[ChunkWindow] = []
    for width in chunk_widths:
        if width > level_width:
            continue
        final_offset = level_width - width
        ending_start = max(0, level_width - ending_columns)
        offsets = [
            offset
            for offset in range(0, final_offset + 1, stride_tiles)
            if offset == 0 or offset + width <= ending_start
        ]
        if final_offset not in offsets:
            offsets.append(final_offset)
        for offset in offsets:
            windows.append(
                ChunkWindow(
                    tiles=level.tiles[:, offset : offset + width].copy(),
                    family=_family_for_offset(offset, final_offset),
                    level_id=level.level_id,
                    column_offset=offset,
                )
            )
    return windows


def _family_for_offset(offset: int, final_offset: int) -> ChunkFamily:
    """Map a window offset to its structural family."""
    if offset == final_offset:
        return ChunkFamily.ENDING
    if offset == 0:
        return ChunkFamily.START
    return ChunkFamily.MIDDLE


def deduplicate_windows(windows: list[ChunkWindow]) -> list[ChunkWindow]:
    """Deduplicate windows by family, width, and exact tile content.

    Args:
        windows: Candidate chunk windows before deduplication.

    Returns:
        First-seen window for each unique content key.
    """
    seen: set[bytes] = set()
    unique: list[ChunkWindow] = []
    for window in windows:
        key = (
            window.family.value.encode()
            + window.tiles.shape[1].to_bytes(2, "little")
            + window.tiles.tobytes()
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(window)
    return unique


def cap_windows(windows: list[ChunkWindow], *, max_chunks: int, seed: int) -> list[ChunkWindow]:
    """Cap library size by subsampling MIDDLE windows while keeping endpoints.

    Args:
        windows: Candidate windows after deduplication.
        max_chunks: Maximum total windows to retain.
        seed: RNG seed for uniform middle subsampling.

    Returns:
        Protected START/ENDING windows plus a random middle subset.
    """
    if len(windows) <= max_chunks:
        return windows
    protected = [w for w in windows if w.family != ChunkFamily.MIDDLE]
    middles = [w for w in windows if w.family == ChunkFamily.MIDDLE]
    budget = max(0, max_chunks - len(protected))
    rng = np.random.default_rng(seed)
    keep_indices = sorted(rng.choice(len(middles), size=budget, replace=False).tolist())
    kept = {id(middles[index]) for index in keep_indices}
    return [w for w in windows if w.family != ChunkFamily.MIDDLE or id(w) in kept]
