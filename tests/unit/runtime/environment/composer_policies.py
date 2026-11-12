from __future__ import annotations

import numpy as np
import pytest
from pcgrl.core.library.types import Chunk, ChunkFamily, ChunkLibrary
from pcgrl.runtime.environment.composer_policies import (
    BASELINE_NAMES,
    BigramMarkov,
    rollout_baseline,
    rollout_greedy_intent,
    rollout_random_legal,
)
from pcgrl.runtime.environment.composer_state import ComposerCore


def _chunk(chunk_id, catalog, family=ChunkFamily.MIDDLE, enemies=0, level_id="1-1", offset=0):
    """chunk."""
    tiles = np.full((14, 16), catalog.id_for_name("empty"), dtype=np.int16)
    tiles[-2:, :] = catalog.id_for_name("solid")
    counts: dict[str, int] = {}
    if enemies:
        tiles[-3, 2 : 2 + enemies] = catalog.id_for_name("goomba")
        counts["enemy"] = enemies
    return Chunk(
        chunk_id=chunk_id,
        tiles=tiles,
        family=family,
        level_id=level_id,
        column_offset=offset,
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
        _chunk(0, catalog, family=ChunkFamily.START, offset=0),
        _chunk(1, catalog, offset=8),
        _chunk(2, catalog, enemies=6, level_id="2-1", offset=8),
        _chunk(3, catalog, family=ChunkFamily.ENDING, offset=24),
    ]
    library = ChunkLibrary(
        chunks,
        train_level_ids=["1-1", "2-1"],
        heldout_level_ids=["8-1"],
        tile_names=[t.name for t in catalog.tiles],
    )
    return ComposerCore(library, config, catalog)


def test_random_legal_closes_episode(core) -> None:
    """Verify random legal closes episode."""
    state = rollout_random_legal(core, np.full(8, 0.5), np.random.default_rng(0))
    assert state.closed
    assert state.chunk_ids[0] == 0
    assert state.chunk_ids[-1] == 3


def test_greedy_matches_enemy_target_direction(core) -> None:
    """Verify greedy matches enemy target direction."""
    rng = np.random.default_rng(0)
    rich = rollout_greedy_intent(core, np.array([1.0] + [0.5] * 7, dtype=np.float32), rng)
    poor = rollout_greedy_intent(core, np.zeros(8, dtype=np.float32), np.random.default_rng(0))
    enemy_rich = rich.counts[0]
    enemy_poor = poor.counts[0]
    assert enemy_rich > enemy_poor


def test_markov_fit_orders_by_offset(core) -> None:
    """Verify markov fit orders by offset."""
    model = BigramMarkov.fit(core.library)
    # level 1-1 sequence: chunk 0 (offset 0) -> chunk 1 (offset 8) -> 3 (24)
    assert 0 in model.initial
    assert 1 in model.transitions.get(0, {})
    state = model.rollout(core, np.full(8, 0.5), np.random.default_rng(2))
    assert state.closed


def test_all_baselines_produce_closed_levels(core, config, catalog) -> None:
    """Verify all baselines produce closed levels."""
    from pcgrl.runtime.metrics.tpkl import PatternReference, count_patterns
    from pcgrl.runtime.reward.composer_terminal import ComposerTerminalReward

    size = config.reward.composer.style_pattern_size
    counts = count_patterns(core.library[0].tiles, size)
    reference = PatternReference({size: dict(counts)})
    terminal = ComposerTerminalReward(config, catalog, reference)
    for name in BASELINE_NAMES:
        state = rollout_baseline(
            name, core, np.full(8, 0.5), np.random.default_rng(7), terminal=terminal
        )
        assert state.closed, name
        level = core.stitch_level(state)
        assert level.width_tiles == config.game.width_tiles


def test_unknown_baseline_raises(core) -> None:
    """Verify unknown baseline raises."""
    with pytest.raises(ValueError, match="unknown baseline"):
        rollout_baseline("nope", core, np.full(8, 0.5), np.random.default_rng(0))
