from __future__ import annotations

import numpy as np
import pytest
from pcgrl.runtime.metrics.tpkl import (
    PatternReference,
    count_patterns,
    style_score,
    tile_pattern_kl,
)


def _corpus_like_tiles(width: int = 60, seed: int = 0) -> np.ndarray:
    """Corpus like tiles."""
    # Flat floor with sparse, structured content: low pattern entropy.
    rng = np.random.default_rng(seed)
    tiles = np.zeros((14, width), dtype=np.int16)
    tiles[-2:, :] = 1
    for col in rng.choice(width - 4, size=4, replace=False):
        tiles[8, col : col + 3] = 2
    return tiles


def _confetti_tiles(width: int = 60, seed: int = 1) -> np.ndarray:
    """confetti tiles."""
    rng = np.random.default_rng(seed)
    tiles = np.zeros((14, width), dtype=np.int16)
    tiles[-2:, :] = 1
    scatter = rng.random((12, width)) < 0.25
    tiles[:12][scatter] = rng.integers(1, 5, size=int(scatter.sum()), dtype=np.int16)
    return tiles


def _reference(size: int = 3) -> PatternReference:
    """Reference."""
    counts = count_patterns(_corpus_like_tiles(seed=0), size)
    counts.update(count_patterns(_corpus_like_tiles(seed=2), size))
    return PatternReference({size: dict(counts)})


def test_count_patterns_window_count() -> None:
    """Verify count patterns window count."""
    tiles = np.zeros((4, 6), dtype=np.int16)
    counts = count_patterns(tiles, 3)
    assert sum(counts.values()) == (4 - 2) * (6 - 2)
    assert len(counts) == 1


def test_count_patterns_too_small_grid_is_empty() -> None:
    """Verify tpkl near zero for reference material."""
    assert count_patterns(np.zeros((2, 2), dtype=np.int16), 3) == {}


def test_tpkl_near_zero_for_reference_material() -> None:
    """Verify tpkl orders corpus below confetti."""
    reference = _reference()
    kl = tile_pattern_kl(_corpus_like_tiles(seed=0), reference, 3)
    assert kl == pytest.approx(0.0, abs=0.2)


def test_tpkl_orders_corpus_below_confetti() -> None:
    """Verify style score in unit interval and monotonic."""
    reference = _reference()
    corpus_kl = tile_pattern_kl(_corpus_like_tiles(seed=3), reference, 3)
    confetti_kl = tile_pattern_kl(_confetti_tiles(), reference, 3)
    assert corpus_kl < confetti_kl


def test_style_score_in_unit_interval_and_monotonic() -> None:
    """Verify pattern reference load roundtrip."""
    reference = _reference()
    corpus_style = style_score(_corpus_like_tiles(seed=3), reference, 3)
    confetti_style = style_score(_confetti_tiles(), reference, 3)
    assert 0.0 < confetti_style < corpus_style <= 1.0


def test_pattern_reference_load_roundtrip(tmp_path) -> None:
    """Verify pattern reference load roundtrip."""
    counts = count_patterns(_corpus_like_tiles(seed=0), 3)
    patterns = np.stack(
        [np.frombuffer(key, dtype=np.int16).reshape(3, 3) for key in sorted(counts)]
    )
    frequencies = np.array([counts[key] for key in sorted(counts)], dtype=np.int64)
    np.savez_compressed(
        tmp_path / "pattern_distributions.npz",
        sizes=np.array([3], dtype=np.int64),
        train_patterns_k3=patterns,
        train_counts_k3=frequencies,
        heldout_patterns_k3=patterns,
        heldout_counts_k3=frequencies,
    )
    for split in ("train", "heldout"):
        reference = PatternReference.load(tmp_path, split=split)
        assert reference.sizes == (3,)
        kl = tile_pattern_kl(_corpus_like_tiles(seed=0), reference, 3)
        assert kl == pytest.approx(0.0, abs=1e-3)


def test_pattern_reference_load_missing_file_raises(tmp_path) -> None:
    """Verify pattern reference load missing file raises."""
    with pytest.raises(FileNotFoundError, match="prepare"):
        PatternReference.load(tmp_path)


def test_pattern_reference_unknown_size_raises() -> None:
    """Verify pattern reference unknown size raises."""
    reference = _reference(3)
    with pytest.raises(KeyError, match="k=4"):
        tile_pattern_kl(_corpus_like_tiles(), reference, 4)
