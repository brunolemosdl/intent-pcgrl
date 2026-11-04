from __future__ import annotations

import numpy as np
import pytest
from pcgrl.core.library.types import Chunk, ChunkFamily, ChunkLibrary
from pcgrl.runtime.environment.composer_state import ComposerCore
from pcgrl.runtime.environment.viability import (
    IntentViabilityMap,
    build_viability_map,
)
from pcgrl.runtime.metrics.intent import INTENT_KEYS


def _flat_chunk(chunk_id: int, catalog, family=ChunkFamily.MIDDLE, enemies: int = 0) -> Chunk:
    """ flat chunk."""
    tiles = np.full((14, 16), catalog.id_for_name("empty"), dtype=np.int16)
    tiles[-2:, :] = catalog.id_for_name("solid")
    counts: dict[str, int] = {}
    if enemies:
        goomba = catalog.id_for_name("goomba")
        tiles[-3, 2 : 2 + enemies] = goomba
        counts["enemy"] = enemies
    return Chunk(
        chunk_id=chunk_id,
        tiles=tiles,
        family=family,
        level_id="1-1",
        column_offset=0,
        entry_height=2,
        exit_height=2,
        gap_at_left=False,
        gap_at_right=False,
        category_counts=counts,
    )


@pytest.fixture
def core(config, catalog):
    """Core."""
    chunks = [
        _flat_chunk(0, catalog, family=ChunkFamily.START),
        _flat_chunk(1, catalog),
        _flat_chunk(2, catalog, enemies=4),
        _flat_chunk(3, catalog, family=ChunkFamily.ENDING),
    ]
    library = ChunkLibrary(
        chunks,
        train_level_ids=["1-1"],
        heldout_level_ids=["8-1"],
        tile_names=[t.name for t in catalog.tiles],
    )
    return ComposerCore(library, config, catalog)


def test_build_viability_map_shapes(core) -> None:
    """Verify build viability map shapes."""
    viability = build_viability_map(core, episodes=8, seed=1)
    assert viability.samples.shape[1] == len(INTENT_KEYS)
    assert viability.samples.shape[0] >= 2
    assert ((viability.samples >= 0.0) & (viability.samples <= 1.0)).all()


def test_quantiles_and_correlations(core) -> None:
    """Verify quantiles and correlations."""
    viability = build_viability_map(core, episodes=8, seed=1)
    quantiles = viability.quantiles()
    assert set(quantiles) == set(INTENT_KEYS)
    for stats in quantiles.values():
        assert stats["p05"] <= stats["p50"] <= stats["p95"]
    matrix = viability.correlations()
    assert matrix.shape == (8, 8)
    for a, b, corr in viability.correlated_pairs(threshold=0.0):
        assert a != b
        assert -1.0 <= corr <= 1.0


def test_sample_target_in_unit_cube_and_deterministic(core) -> None:
    """Verify sample target in unit cube and deterministic."""
    viability = build_viability_map(core, episodes=6, seed=3)
    rng_a = np.random.default_rng(9)
    rng_b = np.random.default_rng(9)
    for _ in range(20):
        sample_a = viability.sample_target(rng_a)
        sample_b = viability.sample_target(rng_b)
        assert sample_a.shape == (8,)
        assert ((sample_a >= 0.0) & (sample_a <= 1.0)).all()
        assert np.allclose(sample_a, sample_b)


def test_save_load_roundtrip(core, tmp_path) -> None:
    """Verify save load roundtrip."""
    viability = build_viability_map(core, episodes=6, seed=3)
    viability.save(tmp_path)
    loaded = IntentViabilityMap.load(tmp_path)
    assert np.allclose(viability.samples, loaded.samples)


def test_load_missing_raises(tmp_path) -> None:
    """Verify load missing raises."""
    with pytest.raises(FileNotFoundError, match="library"):
        IntentViabilityMap.load(tmp_path)
