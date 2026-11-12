from __future__ import annotations

import numpy as np
import pytest
from pcgrl.app.data.library.extract import RawLevel
from pcgrl.core.helpers.path import resolve_path
from pcgrl.core.settings.loader import load_config
from pcgrl.runtime.metrics.tpkl import PatternReference


def _level(level_id: str, seed: int) -> RawLevel:
    """level."""
    rng = np.random.default_rng(seed)
    tiles = np.zeros((14, 40), dtype=np.int16)
    tiles[-2:, :] = 1
    tiles[8, rng.integers(0, 37)] = 2
    return RawLevel(level_id=level_id, tiles=tiles)


def test_extract_pattern_distributions_splits_by_level() -> None:
    """Verify extract pattern distributions splits by level."""
    from pcgrl.app.data.corpus.patterns import extract_pattern_distributions

    levels = [_level(f"mario-1/{i}-1", i) for i in range(4)]
    arrays = extract_pattern_distributions(levels, sizes=[2, 3], heldout_level_ids={"mario-1/3-1"})
    for size in (2, 3):
        for split in ("train", "heldout"):
            patterns = arrays[f"{split}_patterns_k{size}"]
            counts = arrays[f"{split}_counts_k{size}"]
            assert patterns.shape[0] == counts.shape[0] > 0
            assert patterns.shape[1:] == (size, size)
    # 3 train levels vs 1 heldout level -> more train pattern mass.
    assert arrays["train_counts_k2"].sum() > arrays["heldout_counts_k2"].sum()


def test_run_patterns_creates_loadable_reference(tmp_path) -> None:
    """Verify run patterns creates loadable reference."""
    config = load_config(training="initial", apply_profile=False)
    corpus_root = resolve_path(config.dataset.corpus_root)
    if not corpus_root.is_dir() or not any(corpus_root.rglob("*.txt")):
        pytest.skip("Corpus files not present")
    from pcgrl.app.data.corpus.patterns import run_patterns
    from pcgrl.core.tiles.catalog import load_catalog

    catalog = load_catalog(config)
    path = run_patterns(config, catalog, tmp_path)
    assert path.is_file()
    for split in ("train", "heldout"):
        reference = PatternReference.load(tmp_path, split=split)
        assert set(reference.sizes) >= set(config.evaluation.tpkl_sizes)
        assert config.reward.composer.style_pattern_size in reference.sizes
