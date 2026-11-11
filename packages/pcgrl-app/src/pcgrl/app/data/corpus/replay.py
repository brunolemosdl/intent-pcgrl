from __future__ import annotations

import numpy as np

from pcgrl.app.data.library.extract import load_raw_levels, split_levels_for_heldout
from pcgrl.core.domain.level import AreaType, LevelGrid
from pcgrl.core.helpers.path import resolve_path
from pcgrl.core.settings.schema import RuntimeConfig
from pcgrl.core.tiles.catalog import load_catalog
from pcgrl.runtime.layout import resolve_layout


def corpus_replay_level(config: RuntimeConfig, seed: int) -> LevelGrid:
    """Replay a random raw-corpus window as a baseline level.

    Description:
        Reads the RAW VGLC parse (the same source as the chunk library) and
        copies a bottom-aligned window, so the ground lands on the grid's last
        rows. Only library train-split levels are eligible: heldout levels feed
        the anti-Goodhart style evaluation and must never appear in any system's
        output. Endpoints are stamped through the shared layout resolution.

    Args:
        config: Runtime settings for corpus location, split seed, and layout.
        seed: RNG seed picking the source level and window offset.

    Returns:
        Level grid with provenance metadata and stamped endpoints.

    Raises:
        FileNotFoundError: When the configured corpus level sets are missing.
    """
    catalog = load_catalog(config)
    corpus_root = resolve_path(config.dataset.corpus_root)
    levels = load_raw_levels(corpus_root, config.library.level_sets, catalog)
    train_ids, _ = split_levels_for_heldout(
        [level.level_id for level in levels],
        heldout_fraction=config.library.heldout_level_fraction,
        seed=config.experiment.seed,
    )
    pool = [level for level in levels if level.level_id in train_ids]
    rng = np.random.default_rng(seed)
    source = pool[int(rng.integers(0, len(pool)))]
    height = config.game.height_tiles
    width = config.game.width_tiles
    empty_id = catalog.id_for_name("empty")
    tiles = np.full((height, width), empty_id, dtype=np.int16)
    rows = min(height, source.tiles.shape[0])
    cols = min(width, source.tiles.shape[1])
    start = int(rng.integers(0, max(0, source.tiles.shape[1] - cols) + 1))
    tiles[height - rows :, :cols] = source.tiles[
        source.tiles.shape[0] - rows :, start : start + cols
    ]
    resolved = resolve_layout(config.layout, height=height, width=width)
    solid_id = catalog.id_for_name(config.layout.ground_tile)
    if config.layout.fill_full_ground:
        tiles[resolved.ground_row, :] = solid_id
    for y, x in resolved.reserved:
        if y == resolved.ground_row:
            tiles[y, x] = solid_id
        elif (y, x) not in (resolved.player, resolved.flagpole):
            tiles[y, x] = empty_id
    tiles[resolved.player] = catalog.id_for_name("player")
    tiles[resolved.flagpole] = catalog.id_for_name("flagpole")
    return LevelGrid(
        tiles=tiles,
        width_tiles=width,
        height_tiles=height,
        tile_size_px=config.game.tile_size_px,
        area_type=AreaType(config.game.area_type),
        metadata={
            "baseline": "corpus_replay",
            "source_level_id": source.level_id,
            "column_offset": start,
        },
    )
