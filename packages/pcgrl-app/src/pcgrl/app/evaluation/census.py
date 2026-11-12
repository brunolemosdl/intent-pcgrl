from __future__ import annotations

import itertools
from collections import Counter
from typing import cast

import numpy as np

from pcgrl.core.domain.level import LevelGrid
from pcgrl.core.settings.schema import RuntimeConfig
from pcgrl.core.tiles.catalog import TileCatalog
from pcgrl.runtime.metrics.intent import gap_widths, ground_height_profile
from pcgrl.runtime.metrics.usefulness import block_names, solid_like_ids

_HISTOGRAM_KEYS = ("gap_widths", "platform_lengths", "pipe_heights", "content_thirds")


def _platform_run_lengths(
    level: LevelGrid, catalog: TileCatalog, config: RuntimeConfig
) -> list[int]:
    """Lengths of horizontal solid-like runs NOT on the bottom-contiguous floor."""
    solid_ids = solid_like_ids(catalog, config)
    ground = ground_height_profile(level, catalog, config)
    height = level.height_tiles
    lengths: list[int] = []
    for row in range(height - 1):
        run = 0
        for column in range(level.width_tiles):
            in_floor = row >= height - ground[column]
            solid = int(level.tiles[row, column]) in solid_ids
            if solid and not in_floor:
                run += 1
            else:
                if run:
                    lengths.append(run)
                run = 0
        if run:
            lengths.append(run)
    return lengths


def _pipe_heights(level: LevelGrid, catalog: TileCatalog) -> list[int]:
    """Height of every pipe, measured from its top-left tile downwards."""
    top_left = catalog.id_for_name("pipe_top_left")
    body_left = catalog.id_for_name("pipe_body_left")
    heights: list[int] = []
    for row, column in np.argwhere(level.tiles == top_left):
        height = 1
        r = int(row) + 1
        while r < level.height_tiles and int(level.tiles[r, int(column)]) == body_left:
            height += 1
            r += 1
        heights.append(height)
    return heights


def _isolated_block_rate(level: LevelGrid, catalog: TileCatalog, config: RuntimeConfig) -> float:
    """Fraction of aerial solid/interactive tiles with no 8-neighbours.

    Fully isolated aerial tiles are the confetti signature of scatter that hits
    densities without structure; corpus levels score ~0 here.
    """
    solid_ids = solid_like_ids(catalog, config)
    block_ids = {
        catalog.id_for_name(name) for name in block_names(config) if name in catalog.name_to_id
    }
    interesting = solid_ids | block_ids
    tiles = level.tiles
    height, width = tiles.shape
    ground = ground_height_profile(level, catalog, config)
    total = 0
    isolated = 0
    for row in range(height):
        for column in range(width):
            if int(tiles[row, column]) not in interesting:
                continue
            if row >= height - ground[column]:
                continue
            total += 1
            if not _has_interesting_neighbour(tiles, row, column, interesting):
                isolated += 1
    return isolated / total if total else 0.0


def _has_interesting_neighbour(
    tiles: np.ndarray, row: int, column: int, interesting: set[int]
) -> bool:
    """True when any of the 8 neighbours belongs to the interesting tile set."""
    height, width = tiles.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == dc == 0:
                continue
            rr, cc = row + dr, column + dc
            if 0 <= rr < height and 0 <= cc < width and int(tiles[rr, cc]) in interesting:
                return True
    return False


def _content_thirds(level: LevelGrid, catalog: TileCatalog) -> list[int]:
    """Non-empty tile counts per level third (front-loading detector)."""
    empty_id = catalog.id_for_name("empty")
    width = level.width_tiles
    bounds = [0, width // 3, 2 * width // 3, width]
    return [int((level.tiles[:, bounds[i] : bounds[i + 1]] != empty_id).sum()) for i in range(3)]


def structural_census(
    level: LevelGrid, catalog: TileCatalog, config: RuntimeConfig
) -> dict[str, object]:
    """Compute structural histograms for one level.

    Args:
        level: Level grid to profile.
        catalog: Tile catalog for category and pipe identification.
        config: Runtime settings for semantic and spatial thresholds.

    Returns:
        Histograms and isolated-block rate for designer-style comparison.
    """
    ground = ground_height_profile(level, catalog, config)
    return {
        "gap_widths": dict(Counter(gap_widths(ground))),
        "platform_lengths": dict(Counter(_platform_run_lengths(level, catalog, config))),
        "pipe_heights": dict(Counter(_pipe_heights(level, catalog))),
        "content_thirds": dict(enumerate(_content_thirds(level, catalog))),
        "isolated_block_rate": _isolated_block_rate(level, catalog, config),
    }


def merge_censuses(censuses: list[dict[str, object]]) -> dict[str, object]:
    """Merge per-level census histograms into one aggregate profile.

    Args:
        censuses: Per-level census dicts from ``structural_census``.

    Returns:
        Combined histograms with mean isolated-block rate.
    """
    merged: dict[str, Counter[int]] = {key: Counter() for key in _HISTOGRAM_KEYS}
    rates: list[float] = []
    for census in censuses:
        for key in _HISTOGRAM_KEYS:
            merged[key].update(cast("dict[int, int]", census[key]))
        rates.append(float(cast(float, census["isolated_block_rate"])))
    result: dict[str, object] = {key: dict(merged[key]) for key in _HISTOGRAM_KEYS}
    result["isolated_block_rate"] = float(np.mean(rates)) if rates else 0.0
    return result


def _total_variation(hist_a: dict[int, int], hist_b: dict[int, int]) -> float:
    """Total-variation distance between two count histograms in [0, 1]."""
    keys = set(hist_a) | set(hist_b)
    total_a = float(sum(hist_a.values())) or 1.0
    total_b = float(sum(hist_b.values())) or 1.0
    return 0.5 * float(
        sum(abs(hist_a.get(key, 0) / total_a - hist_b.get(key, 0) / total_b) for key in keys)
    )


def census_distance(census_a: dict[str, object], census_b: dict[str, object]) -> dict[str, float]:
    """Compute per-histogram total-variation distances between two censuses.

    Args:
        census_a: Reference census profile.
        census_b: Candidate census profile.

    Returns:
        Per-key distances in ``[0, 1]`` plus isolated-rate delta.
    """
    distances = {
        key: round(_total_variation(census_a[key], census_b[key]), 4)  # type: ignore[arg-type]
        for key in _HISTOGRAM_KEYS
    }
    distances["isolated_block_rate_delta"] = round(
        abs(
            float(census_a["isolated_block_rate"])  # type: ignore[arg-type]
            - float(census_b["isolated_block_rate"])  # type: ignore[arg-type]
        ),
        4,
    )
    return distances


def diversity_metrics(levels: list[LevelGrid]) -> dict[str, float]:
    """Measure intra-intent diversity across a batch of generated levels.

    Args:
        levels: Generated levels sharing the same intent target.

    Returns:
        Mean pairwise tile distance and duplicate chunk-sequence rate.
    """
    if len(levels) < 2:
        return {"pairwise_tile_distance": 0.0, "duplicate_sequence_rate": 0.0}
    distances = [float((a.tiles != b.tiles).mean()) for a, b in itertools.combinations(levels, 2)]
    sequences = [tuple(level.metadata.get("chunk_ids", ())) for level in levels]
    counted = Counter(seq for seq in sequences if seq)
    duplicates = sum(count - 1 for count in counted.values())
    denominator = max(1, sum(1 for seq in sequences if seq))
    return {
        "pairwise_tile_distance": round(float(np.mean(distances)), 4),
        "duplicate_sequence_rate": round(duplicates / denominator, 4),
    }
