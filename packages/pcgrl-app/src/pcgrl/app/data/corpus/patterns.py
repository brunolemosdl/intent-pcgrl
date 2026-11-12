from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from pcgrl.app.data.library.extract import (
    RawLevel,
    load_raw_levels,
    split_levels_for_heldout,
)
from pcgrl.core.settings.schema import RuntimeConfig
from pcgrl.core.tiles.catalog import TileCatalog
from pcgrl.runtime.metrics.tpkl import PATTERNS_FILENAME, count_patterns


def _stack_counter(counts: Counter[bytes], size: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert a pattern counter into aligned (patterns, frequencies) arrays."""
    patterns = np.zeros((len(counts), size, size), dtype=np.int16)
    frequencies = np.zeros(len(counts), dtype=np.int64)
    for index, (key, value) in enumerate(sorted(counts.items())):
        patterns[index] = np.frombuffer(key, dtype=np.int16).reshape(size, size)
        frequencies[index] = value
    return patterns, frequencies


def extract_pattern_distributions(
    levels: list[RawLevel],
    *,
    sizes: list[int],
    heldout_level_ids: set[str],
) -> dict[str, np.ndarray]:
    """Count k x k tile patterns per train/heldout split for npz export.

    Args:
        levels: Raw corpus levels with provenance ids.
        sizes: Pattern window sizes to extract.
        heldout_level_ids: Level ids assigned to the heldout split.

    Returns:
        Numpy arrays keyed for ``pattern_distributions.npz`` serialization.
    """
    arrays: dict[str, np.ndarray] = {}
    for split_name in ("train", "heldout"):
        selected = [
            level
            for level in levels
            if (level.level_id in heldout_level_ids) == (split_name == "heldout")
        ]
        for size in sizes:
            counts: Counter[bytes] = Counter()
            for level in selected:
                counts.update(count_patterns(level.tiles, size))
            patterns, frequencies = _stack_counter(counts, size)
            arrays[f"{split_name}_patterns_k{size}"] = patterns
            arrays[f"{split_name}_counts_k{size}"] = frequencies
    return arrays


def run_patterns(config: RuntimeConfig, catalog: TileCatalog, processed_root: Path) -> Path:
    """Persist ``pattern_distributions.npz`` beside the processed corpus.

    Args:
        config: Runtime settings for corpus paths and TPKL sizes.
        catalog: Tile catalog used when loading raw levels.
        processed_root: Directory for processed corpus artifacts.

    Returns:
        Path to the written ``pattern_distributions.npz``.

    Raises:
        FileNotFoundError: When corpus level sets are missing.
        OSError: When the npz file cannot be written.
    """
    from pcgrl.core.helpers.path import resolve_path

    corpus_root = resolve_path(config.dataset.corpus_root)
    levels = load_raw_levels(corpus_root, config.library.level_sets, catalog)
    _, heldout_ids = split_levels_for_heldout(
        [level.level_id for level in levels],
        heldout_fraction=config.library.heldout_level_fraction,
        seed=config.experiment.seed,
    )
    sizes = sorted(set(config.evaluation.tpkl_sizes) | {config.reward.composer.style_pattern_size})
    arrays = extract_pattern_distributions(levels, sizes=sizes, heldout_level_ids=set(heldout_ids))
    arrays["sizes"] = np.array(sizes, dtype=np.int64)
    path = processed_root / PATTERNS_FILENAME
    np.savez_compressed(path, **arrays)  # type: ignore[arg-type]
    return path
