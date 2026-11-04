from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from pcgrl.app.data.library.annotate import annotate_window
from pcgrl.app.data.library.extract import (
    cap_windows,
    deduplicate_windows,
    extract_windows,
    load_raw_levels,
    split_levels_for_heldout,
)
from pcgrl.core.helpers.path import resolve_path
from pcgrl.core.library.types import ChunkFamily, ChunkLibrary
from pcgrl.core.settings.schema import RuntimeConfig
from pcgrl.core.tiles.catalog import load_catalog

logger = logging.getLogger(__name__)


def library_config_hash(config: RuntimeConfig) -> str:
    """Fingerprint the config blocks that shape the chunk library.

    Description:
        Written into the library metadata and checked by the run pipeline so a
        stale library (built under different ``library``/``game`` settings) is
        rebuilt instead of silently reused.

    Args:
        config: Runtime settings.

    Returns:
        Short hex digest of the ``library`` and ``game`` blocks plus the seed.
    """
    payload = {
        "library": config.library.model_dump(mode="json"),
        "game": config.game.model_dump(mode="json"),
        "seed": config.experiment.seed,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    )
    return digest.hexdigest()[:16]


def build_chunk_library(config: RuntimeConfig) -> ChunkLibrary:
    """Build the annotated chunk library from the raw corpus.

    Args:
        config: Runtime settings for library extraction and heldout fraction.

    Returns:
        Annotated ``ChunkLibrary`` with START/MIDDLE/ENDING families populated.

    Raises:
        FileNotFoundError: When requested corpus level sets are missing.
        ValueError: When any chunk family ends up empty after extraction.
    """
    catalog = load_catalog(config)
    settings = config.library
    corpus_root = resolve_path(config.dataset.corpus_root)
    levels = load_raw_levels(corpus_root, settings.level_sets, catalog)
    train_ids, heldout_ids = split_levels_for_heldout(
        [level.level_id for level in levels],
        heldout_fraction=settings.heldout_level_fraction,
        seed=config.experiment.seed,
    )
    windows = []
    for level in levels:
        if level.level_id not in train_ids:
            continue
        windows.extend(
            extract_windows(
                level,
                chunk_widths=settings.chunk_widths,
                stride_tiles=settings.stride_tiles,
                ending_columns=settings.ending_columns,
            )
        )
    if settings.deduplicate:
        windows = deduplicate_windows(windows)
    windows = cap_windows(
        windows, max_chunks=settings.max_chunks, seed=config.experiment.seed
    )
    chunks = [
        annotate_window(window, chunk_id, catalog)
        for chunk_id, window in enumerate(windows)
    ]
    library = ChunkLibrary(
        chunks,
        train_level_ids=train_ids,
        heldout_level_ids=heldout_ids,
        tile_names=[tile.name for tile in catalog.tiles],
        metadata={
            "level_sets": settings.level_sets,
            "chunk_widths": settings.chunk_widths,
            "stride_tiles": settings.stride_tiles,
            "ending_columns": settings.ending_columns,
            "seed": config.experiment.seed,
            "library_config_hash": library_config_hash(config),
        },
    )
    for family in ChunkFamily:
        if not library.by_family(family):
            raise ValueError(
                f"chunk library has no '{family.value}' chunks; "
                "check library.chunk_widths/ending_columns against the corpus"
            )
    return library


def run_library(config: RuntimeConfig) -> Path:
    """Persist the chunk library and viability map for composer training.

    Args:
        config: Runtime settings for library output paths and viability rollouts.

    Returns:
        Path to the library output root directory.

    Raises:
        ValueError: When library extraction yields an incomplete family set.
        OSError: When library or viability artifacts cannot be written.
    """
    library = build_chunk_library(config)
    output_root = resolve_path(config.library.output_root)
    library.save(output_root)
    from pcgrl.runtime.environment.composer_state import ComposerCore
    from pcgrl.runtime.environment.viability import build_viability_map

    core = ComposerCore(library, config, load_catalog(config))
    viability = build_viability_map(
        core, episodes=config.library.viability_episodes, seed=config.experiment.seed
    )
    viability.save(output_root)
    correlated = viability.correlated_pairs()
    if correlated:
        logger.warning("intent dims with |corr| > 0.8 (merge candidates): %s", correlated)
    logger.info(
        "chunk library saved: %d chunks (%d start / %d middle / %d ending), "
        "%d train levels, %d heldout levels -> %s",
        len(library),
        len(library.by_family(ChunkFamily.START)),
        len(library.by_family(ChunkFamily.MIDDLE)),
        len(library.by_family(ChunkFamily.ENDING)),
        len(library.train_level_ids),
        len(library.heldout_level_ids),
        output_root,
    )
    return output_root
