from __future__ import annotations

from pathlib import Path

import numpy as np
import orjson

from pcgrl.app.data.corpus.parser import parse_level_file
from pcgrl.app.data.corpus.patterns import run_patterns
from pcgrl.app.data.corpus.profile import extract_corpus_profile
from pcgrl.core.helpers.path import ensure_dir, resolve_path
from pcgrl.core.settings.schema import RuntimeConfig
from pcgrl.core.tiles.catalog import load_catalog


def run_prepare(config: RuntimeConfig) -> Path:
    """Build the corpus profile and TPKL pattern distributions.

    Description:
        The composer pipeline derives everything from the raw VGLC parse: the
        chunk library (``library`` stage), the corpus profile, and the pattern
        distributions written here.

    Args:
        config: Runtime settings for corpus paths and TPKL sizes.

    Returns:
        Path to the processed corpus root directory.

    Raises:
        FileNotFoundError: When no VGLC level files exist in the corpus root.
        OSError: When processed artifacts cannot be written.
    """
    catalog = load_catalog(config)
    corpus_root = resolve_path(config.dataset.corpus_root)
    processed_root = ensure_dir(config.dataset.processed_root)
    level_files = sorted(f for f in corpus_root.rglob("*.txt") if f.stem != "symbols")
    if not level_files:
        raise FileNotFoundError(f"No VGLC level files found in {corpus_root}")
    parsed_levels = [
        parse_level_file(path, catalog, tile_size_px=config.game.tile_size_px)
        for path in level_files
    ]
    all_tiles = np.concatenate([level.tiles.flatten() for level in parsed_levels])
    counts = np.bincount(all_tiles.astype(np.int64), minlength=catalog.size).astype(np.float64)
    freq = (counts / counts.sum()).tolist()
    meta = {
        "level_count": len(parsed_levels),
        "tile_distribution_train": freq,
        "tile_catalog": {tile.name: tile.glyph for tile in catalog.tiles},
    }
    meta_path = processed_root / "meta.json"
    meta_path.write_bytes(orjson.dumps(meta, option=orjson.OPT_INDENT_2))
    profile = extract_corpus_profile(parsed_levels, catalog, config)
    profile_path = processed_root / "corpus_profile.json"
    profile_path.write_bytes(orjson.dumps(profile, option=orjson.OPT_INDENT_2))
    run_patterns(config, catalog, processed_root)
    return processed_root
