from __future__ import annotations

from typing import Any, cast

import numpy as np

from pcgrl.core.domain.level import LevelGrid
from pcgrl.core.settings.schema import RuntimeConfig
from pcgrl.core.tiles.catalog import TileCatalog

# Percentiles persisted for every scalar metric, so the policy config can pick any
# of them (e.g. density at p95, gap width at p90) without re-running prepare.
_PERCENTILES = (5, 10, 25, 50, 75, 90, 95)
# Histograms of height-above-surface are stored up to this many tiles.
_MAX_HEIGHT = 14
_GROUND_SOLID_FRACTION = 0.5  # a row is "ground" if >= this fraction is solid-like


def _ids(catalog: TileCatalog, names: set[str] | list[str]) -> set[int]:
    """Resolve catalog tile names that exist in the catalog."""
    return {catalog.id_for_name(n) for n in names if n in catalog.name_to_id}


def _category(config: RuntimeConfig, key: str) -> set[str]:
    """Return the configured tile-name set for a category key."""
    return set(config.categories.get(key, []))


def _percentiles(values: list[float]) -> dict[str, float]:
    """Compute configured percentiles plus mean, max, and count."""
    if not values:
        return {f"p{p}": 0.0 for p in _PERCENTILES} | {"mean": 0.0, "max": 0.0, "n": 0}
    arr = np.asarray(values, dtype=float)
    out: dict[str, float] = {f"p{p}": float(np.percentile(arr, p)) for p in _PERCENTILES}
    out["mean"] = float(arr.mean())
    out["max"] = float(arr.max())
    out["n"] = int(arr.size)
    return out


def _normalized_histogram(heights: list[int]) -> list[float]:
    """Histogram of integer heights in [0, _MAX_HEIGHT], normalised so the mode is
    1.0. Index = height above surface. Used directly as a reward-shaping curve."""
    counts = np.zeros(_MAX_HEIGHT + 1, dtype=float)
    for h in heights:
        if 0 <= h <= _MAX_HEIGHT:
            counts[h] += 1
    peak = counts.max()
    if peak <= 0:
        return cast(list[float], counts.tolist())
    return cast(list[float], (counts / peak).tolist())


def _ground_row(tiles: np.ndarray, solid_ids: set[int]) -> int | None:
    """Lowest row that is mostly solid-like (the floor); None if no such row."""
    h, _w = tiles.shape
    solid_arr = np.isin(tiles, list(solid_ids)) if solid_ids else np.zeros_like(tiles, bool)
    for r in range(h - 1, -1, -1):
        if solid_arr[r].mean() >= _GROUND_SOLID_FRACTION:
            return r
    return None


def _surface_distance_below(solid_below: np.ndarray, r: int, c: int, ground_row: int) -> int | None:
    """Rows from (r, c) down to the nearest solid-like tile at/above the ground
    row. None if the column is open to the floor (a coin/arc over a pit)."""
    for d in range(1, ground_row - r + 1):
        if solid_below[r + d, c]:
            return d
    return None


def _level_stats(
    level: LevelGrid,
    catalog: TileCatalog,
    config: RuntimeConfig,
    *,
    solid_ids: set[int],
    ground_id: int,
    brick_ids: set[int],
    question_ids: set[int],
    coin_ids: set[int],
    enemy_ids: set[int],
    pipe_ids: set[int],
    pipe_top_left: int,
    pipe_top_right: int,
    pipe_body_left: int,
    seg_count: int,
    min_platform_width: int,
) -> dict[str, Any] | None:
    """Collect per-level structural stats when a ground row is detectable."""
    tiles = level.tiles
    h, w = tiles.shape
    gr = _ground_row(tiles, solid_ids)
    if gr is None:
        return None  # not overworld-like
    # Density over the *content* rows (sky..ground), excluding the parser's bottom
    # padding, so it matches the 14-row game area instead of being diluted by /16.
    area = float((gr + 1) * w)
    is_solid = np.isin(tiles, list(solid_ids))

    def dens(ids: set[int]) -> float:
        """Return the fraction of cells whose tile id is in ``ids``."""
        return float(np.isin(tiles, list(ids)).sum() / area) if ids else 0.0

    elevated_solid = int((tiles[:gr] == ground_id).sum()) if gr > 0 else 0
    # platform_cell_count / brick_in_runs are filled by the run loop below so the
    # platform & block densities match the runtime's run-length split (a brick in a
    # run >= min_platform_width is a platform tile; shorter brick groups are blocks).
    stats: dict[str, Any] = {
        "_area": area,
        "_elevated_solid": elevated_solid,
        "_total_brick": int(np.isin(tiles, list(brick_ids)).sum()) if brick_ids else 0,
        "_total_question": int(np.isin(tiles, list(question_ids)).sum()) if question_ids else 0,
        "_platform_cells": 0,
        "_brick_in_runs": 0,
        "density": {
            "enemy": dens(enemy_ids),
            "coin": dens(coin_ids),
            "pipe": dens(pipe_ids),
            "brick": dens(brick_ids),
            "question": dens(question_ids),
            "elevated_solid": elevated_solid / area,
            "solid_like": dens(solid_ids),
        },
        "coin_h": [],
        "coin_arc": 0,
        "coin_total": 0,
        "block_h": [],
        "floating_platform_h": [],
        "platform_run_width": [],
        "gap_widths": [],
        "pipe_heights": [],
        "pipe_cols": [],
        "ground_hole_frac": 0.0,
        "width": w,
        "ground_row": gr,
        "seg_counts": {k: [0] * seg_count for k in ("enemy", "coin", "pipe", "block", "platform")},
    }
    edges = np.linspace(0, w, seg_count + 1, dtype=int)

    def seg_of(x: int) -> int:
        """Map a scalar value to its histogram segment index."""
        return max(0, min(seg_count - 1, int(np.searchsorted(edges, x, side="right") - 1)))

    # Coins / blocks: height above local surface.
    for r in range(h):
        for c in range(w):
            tid = int(tiles[r, c])
            if tid in coin_ids:
                stats["coin_total"] += 1
                d = _surface_distance_below(is_solid, r, c, gr)
                if d is None:
                    stats["coin_arc"] += 1
                else:
                    stats["coin_h"].append(d)
                stats["seg_counts"]["coin"][seg_of(c)] += 1
            elif tid in brick_ids or tid in question_ids:
                d = _surface_distance_below(is_solid, r, c, gr)
                if d is not None:
                    stats["block_h"].append(d)
                stats["seg_counts"]["block"][seg_of(c)] += 1
            elif tid in enemy_ids:
                stats["seg_counts"]["enemy"][seg_of(c)] += 1

    # Elevated solid-like runs (row < ground): widths + floating-platform heights.
    plat_ids = brick_ids | {ground_id}
    for r in range(gr):
        c = 0
        while c < w:
            if int(tiles[r, c]) in plat_ids:
                start = c
                while c < w and int(tiles[r, c]) in plat_ids:
                    c += 1
                run_w = c - start
                stats["platform_run_width"].append(run_w)
                stats["seg_counts"]["platform"][seg_of((start + c - 1) // 2)] += 1
                if run_w >= min_platform_width:
                    stats["_platform_cells"] += run_w
                    stats["_brick_in_runs"] += sum(
                        1 for cc in range(start, c) if int(tiles[r, cc]) in brick_ids
                    )
                if run_w >= min_platform_width and r + 1 <= gr:
                    floating = all(not is_solid[r + 1, cc] for cc in range(start, c))
                    if floating:
                        hs = [
                            d
                            for cc in range(start, c)
                            if (d := _surface_distance_below(is_solid, r, cc, gr))
                        ]
                        if hs:
                            stats["floating_platform_h"].append(int(np.median(hs)))
            else:
                c += 1

    # Ground-row gaps.
    holes = ~is_solid[gr]
    stats["ground_hole_frac"] = float(holes.mean())
    run = 0
    for hole in holes:
        if hole:
            run += 1
        elif run > 0:
            stats["gap_widths"].append(run)
            run = 0
    if run > 0:
        stats["gap_widths"].append(run)

    # Pipes: height (cap + body) and column positions for spacing.
    for c in range(w - 1):
        for r in range(h):
            if int(tiles[r, c]) == pipe_top_left and int(tiles[r, c + 1]) == pipe_top_right:
                ph = 1
                rr = r + 1
                while rr < h and int(tiles[rr, c]) == pipe_body_left:
                    ph += 1
                    rr += 1
                stats["pipe_heights"].append(ph)
                stats["pipe_cols"].append(c)
                stats["seg_counts"]["pipe"][seg_of(c)] += 1
                break

    # Platform / block densities via the run-length split (match the runtime).
    block_cells = stats["_total_brick"] - stats["_brick_in_runs"] + stats["_total_question"]
    stats["density"]["platform"] = stats["_platform_cells"] / area
    stats["density"]["block"] = block_cells / area
    return stats


def extract_corpus_profile(
    levels: list[LevelGrid], catalog: TileCatalog, config: RuntimeConfig
) -> dict[str, Any]:
    """Aggregate structural statistics across overworld-like corpus levels.

    Args:
        levels: Parsed corpus levels to profile.
        catalog: Tile catalog for category and tile identification.
        config: Runtime settings for semantic and spatial thresholds.

    Returns:
        JSON-serializable corpus profile with densities, gaps, pipes, and
        spatial segment shares.
    """
    solid_ids = _ids(catalog, _category(config, "solid_like"))
    pipe_ids = _ids(catalog, _category(config, "pipes"))
    enemy_ids = _ids(catalog, _category(config, "enemies"))
    coin_ids = _ids(catalog, _category(config, "rewards")) - solid_ids
    question_ids = _ids(catalog, {"question_block_coin", "question_block_empty"})
    brick_ids = _ids(catalog, {"brick"})
    ground_id = catalog.id_for_name(config.layout.ground_tile)
    seg_count = config.metrics.spatial.segment_count
    min_platform_width = config.metrics.semantic.min_platform_width

    pipe_top_left = catalog.id_for_name("pipe_top_left")
    pipe_top_right = catalog.id_for_name("pipe_top_right")
    pipe_body_left = catalog.id_for_name("pipe_body_left")

    per_level: list[dict[str, Any]] = []
    for level in levels:
        s = _level_stats(
            level,
            catalog,
            config,
            solid_ids=solid_ids,
            ground_id=ground_id,
            brick_ids=brick_ids,
            question_ids=question_ids,
            coin_ids=coin_ids,
            enemy_ids=enemy_ids,
            pipe_ids=pipe_ids,
            pipe_top_left=pipe_top_left,
            pipe_top_right=pipe_top_right,
            pipe_body_left=pipe_body_left,
            seg_count=seg_count,
            min_platform_width=min_platform_width,
        )
        if s is not None:
            per_level.append(s)

    if not per_level:
        return {"n_levels": 0, "area_type": config.game.area_type}

    # Densities: distribution over levels (so caps can use a percentile).
    density: dict[str, dict[str, float]] = {}
    for key in (
        "enemy",
        "coin",
        "pipe",
        "brick",
        "question",
        "elevated_solid",
        "solid_like",
        "platform",
        "block",
    ):
        density[key] = _percentiles([s["density"][key] for s in per_level])

    coin_h = [v for s in per_level for v in s["coin_h"]]
    block_h = [v for s in per_level for v in s["block_h"]]
    float_h = [v for s in per_level for v in s["floating_platform_h"]]
    coin_total = sum(s["coin_total"] for s in per_level)
    coin_arc = sum(s["coin_arc"] for s in per_level)
    run_widths = [v for s in per_level for v in s["platform_run_width"]]
    gap_widths = [v for s in per_level for v in s["gap_widths"]]
    # Gap count normalised per the generated width (game.width_tiles) so the cap is
    # comparable across corpus levels of different widths.
    gen_width = float(config.game.width_tiles)
    gap_count_norm = [len(s["gap_widths"]) * gen_width / max(1.0, s["width"]) for s in per_level]
    hole_frac = [s["ground_hole_frac"] for s in per_level]
    pipe_heights = [v for s in per_level for v in s["pipe_heights"]]
    pipe_spacing = [
        cols[i + 1] - cols[i]
        for s in per_level
        for cols in [sorted(set(s["pipe_cols"]))]
        for i in range(len(cols) - 1)
    ]

    def seg_share(key: str) -> list[float]:
        """Return normalised segment shares for histogram key ``key``."""
        total = np.zeros(seg_count, dtype=float)
        for s in per_level:
            total += np.asarray(s["seg_counts"][key], dtype=float)
        ssum = total.sum()
        return (total / ssum).tolist() if ssum > 0 else total.tolist()

    return {
        "n_levels": len(per_level),
        "area_type": config.game.area_type,
        "density": density,
        "height_above_surface": {
            "floating_platform": {
                "hist": _normalized_histogram(float_h),
                "pcts": _percentiles([float(v) for v in float_h]),
            },
            "coin": {
                "hist": _normalized_histogram(coin_h),
                "pcts": _percentiles([float(v) for v in coin_h]),
                "arc_over_pit_frac": float(coin_arc / max(1, coin_total)),
            },
            "block": {
                "hist": _normalized_histogram(block_h),
                "pcts": _percentiles([float(v) for v in block_h]),
            },
        },
        "platform_run_width": {"pcts": _percentiles([float(v) for v in run_widths])},
        "gaps": {
            "width": {"pcts": _percentiles([float(v) for v in gap_widths])},
            "count_per_level_norm": {"pcts": _percentiles(gap_count_norm)},
            "ground_hole_frac": {"pcts": _percentiles(hole_frac)},
        },
        "pipe": {
            "height": {"pcts": _percentiles([float(v) for v in pipe_heights])},
            "spacing": {"pcts": _percentiles([float(v) for v in pipe_spacing])},
        },
        "spatial": {
            "per_segment_share": {
                key: seg_share(key) for key in ("enemy", "coin", "pipe", "block", "platform")
            }
        },
    }
