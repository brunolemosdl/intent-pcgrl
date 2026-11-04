from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from pcgrl.bridge.bridge import GameplayBridge
from pcgrl.core.domain.level import AreaType, LevelGrid
from pcgrl.core.helpers.path import ensure_dir, resolve_path
from pcgrl.core.settings.schema import RuntimeConfig
from pcgrl.core.tiles.catalog import TileCatalog, load_catalog
from pcgrl.runtime.layout import resolve_layout
from pcgrl.runtime.validators.reachability import reachable_cells

logger = logging.getLogger(__name__)


def _stamp_endpoints(tiles: np.ndarray, catalog: TileCatalog, config: RuntimeConfig) -> np.ndarray:
    """Stamp player, flag and reserved-region ground onto a tile grid."""
    height, width = tiles.shape
    resolved = resolve_layout(config.layout, height=height, width=width)
    solid_id = catalog.id_for_name(config.layout.ground_tile)
    empty_id = catalog.id_for_name("empty")
    for y, x in resolved.reserved:
        if y == resolved.ground_row:
            tiles[y, x] = solid_id
        elif (y, x) not in (resolved.player, resolved.flagpole):
            tiles[y, x] = empty_id
    tiles[resolved.player] = catalog.id_for_name("player")
    tiles[resolved.flagpole] = catalog.id_for_name("flagpole")
    return tiles


def _as_level(tiles: np.ndarray, catalog: TileCatalog, config: RuntimeConfig, origin: str) -> LevelGrid:
    """Pad/trim a raw tile grid to game dimensions and stamp the endpoints."""
    height, width = tiles.shape
    if height < config.game.height_tiles:
        padded = np.full(
            (config.game.height_tiles, width), catalog.id_for_name("empty"), dtype=np.int16
        )
        padded[config.game.height_tiles - height :, :] = tiles
        tiles = padded
        height = config.game.height_tiles
    if width > config.game.width_tiles:
        tiles = tiles[:, : config.game.width_tiles]
        width = config.game.width_tiles
    tiles = _stamp_endpoints(tiles.copy(), catalog, config)
    return LevelGrid(
        tiles=tiles,
        width_tiles=width,
        height_tiles=height,
        tile_size_px=config.game.tile_size_px,
        area_type=AreaType.OVERWORLD,
        metadata={"calibration_origin": origin},
    )


def build_calibration_pool(
    config: RuntimeConfig, catalog: TileCatalog, *, num_levels: int, seed: int
) -> list[LevelGrid]:
    """Build a mixed calibration pool of corpus, perturbed, and composed levels.

    Args:
        config: Runtime settings for game dimensions and library paths.
        catalog: Tile catalog for endpoint stamping and composition.
        num_levels: Total pool size to assemble.
        seed: RNG seed for perturbations and composed targets.

    Returns:
        List of levels sized to ``num_levels`` for bridge-vs-flood calibration.
    """
    from pcgrl.app.data.library.extract import load_raw_levels
    from pcgrl.core.library.types import ChunkLibrary
    from pcgrl.runtime.environment.composer_policies import rollout_random_legal
    from pcgrl.runtime.environment.composer_state import ComposerCore

    rng = np.random.default_rng(seed)
    raw = load_raw_levels(
        resolve_path(config.dataset.corpus_root), config.library.level_sets, catalog
    )
    pool: list[LevelGrid] = []
    per_bucket = max(1, num_levels // 3)
    for index in range(per_bucket):
        level = raw[index % len(raw)]
        pool.append(_as_level(level.tiles, catalog, config, "corpus"))
    empty_id = catalog.id_for_name("empty")
    for index in range(per_bucket):
        level = raw[(index + 7) % len(raw)]
        tiles = level.tiles.copy()
        for _ in range(int(rng.integers(1, 4))):
            start = int(rng.integers(4, tiles.shape[1] - 12))
            width = int(rng.integers(3, 10))
            tiles[-4:, start : start + width] = empty_id
        pool.append(_as_level(tiles, catalog, config, "perturbed"))
    library = ChunkLibrary.load(resolve_path(config.library.output_root))
    core = ComposerCore(library, config, catalog)
    while len(pool) < num_levels:
        target = rng.uniform(0.0, 1.0, size=8).astype(np.float32)
        state = rollout_random_legal(core, target, rng)
        pool.append(core.stitch_level(state))
    return pool[:num_levels]


def _flood_playable(
    level: LevelGrid, catalog: TileCatalog, *, jump_height: int, jump_reach: int
) -> bool:
    """Flood prediction: can the flag column be reached under these physics?"""
    flag_id = catalog.id_for_name("flagpole")
    flags = np.argwhere(level.tiles == flag_id)
    if flags.size == 0:
        return False
    flag_col = int(flags[0][1])
    reach = reachable_cells(level, catalog, jump_height=jump_height, jump_reach=jump_reach)
    if not reach.standable:
        return False
    return max(col for _, col in reach.standable) >= flag_col


def _bridge_truths(
    config: RuntimeConfig, pool: list[LevelGrid], agent: str
) -> list[bool]:
    """Ground-truth playability per level, judged by the bridge A* agent."""
    bridge = GameplayBridge(config)
    bridge.ensure_available()
    truths: list[bool] = []
    for level in pool:
        result = bridge.evaluate(
            level,
            agent=agent,
            timer_seconds=config.evaluation.timer_seconds,
            mario_state=config.evaluation.mario_state,
            num_runs=config.evaluation.num_runs,
        )
        truths.append(bool(result.won))
    return truths


def _grid_entry(
    pool: list[LevelGrid],
    truths: list[bool],
    catalog: TileCatalog,
    *,
    jump_height: int,
    jump_reach: int,
    max_false_playable: float,
) -> dict[str, object]:
    """Confusion-matrix entry of the flood predictor for one physics pair."""
    predictions = [
        _flood_playable(level, catalog, jump_height=jump_height, jump_reach=jump_reach)
        for level in pool
    ]
    tp = sum(1 for p, t in zip(predictions, truths, strict=True) if p and t)
    fp = sum(1 for p, t in zip(predictions, truths, strict=True) if p and not t)
    tn = sum(1 for p, t in zip(predictions, truths, strict=True) if not p and not t)
    fn = sum(1 for p, t in zip(predictions, truths, strict=True) if not p and t)
    accuracy = (tp + tn) / len(pool)
    false_playable = fp / max(1, fp + tn)
    return {
        "jump_height": jump_height,
        "jump_reach": jump_reach,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": round(accuracy, 4),
        "false_playable_rate": round(false_playable, 4),
        "meets_cap": false_playable <= max_false_playable,
    }


def run_calibration(
    config: RuntimeConfig,
    *,
    num_levels: int | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Calibrate flood jump physics against the bridge A* agent.

    Args:
        config: Runtime settings for calibration grid and false-playable cap.
        num_levels: Optional pool size override.
        output_dir: Directory for the calibration JSON report.

    Returns:
        Path to the written calibration report.

    Raises:
        FileNotFoundError: When the gameplay bridge JAR is missing.
        OSError: When the report file cannot be written.

    Description:
        Sweeps candidate ``(jump_height, jump_reach)`` pairs and recommends the
        accuracy-maximising pair subject to the false-playable rate cap.
    """
    catalog = load_catalog(config)
    settings = config.evaluation.calibration
    pool = build_calibration_pool(
        config,
        catalog,
        num_levels=num_levels or settings.num_levels,
        seed=config.experiment.seed,
    )
    agent = config.evaluation.agents[0]
    truths = _bridge_truths(config, pool, agent)
    report: dict[str, object] = {
        "agent": agent,
        "num_levels": len(pool),
        "bridge_win_rate": float(np.mean(truths)),
        "grid": [],
    }
    best: tuple[float, dict[str, object]] | None = None
    for jump_height in settings.jump_height_candidates:
        for jump_reach in settings.jump_reach_candidates:
            entry = _grid_entry(
                pool,
                truths,
                catalog,
                jump_height=jump_height,
                jump_reach=jump_reach,
                max_false_playable=settings.max_false_playable,
            )
            report["grid"].append(entry)  # type: ignore[attr-defined]
            accuracy = float(entry["accuracy"])  # type: ignore[arg-type]
            if entry["meets_cap"] and (best is None or accuracy > best[0]):
                best = (accuracy, entry)
    report["recommended"] = best[1] if best else None
    if best:
        logger.info(
            "calibration: recommended jump_height=%s jump_reach=%s (accuracy=%.3f)",
            best[1]["jump_height"],
            best[1]["jump_reach"],
            best[0],
        )
    else:
        logger.warning(
            "calibration: NO candidate met false_playable <= %.2f",
            settings.max_false_playable,
        )
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ensure_dir(
        output_dir or resolve_path("artifacts/reports/calibration") / timestamp
    )
    path = out_dir / "calibration.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
