# Objective sanity harness: the composer reward must order
# corpus material >> confetti >> empty sky, and shuffling a good level must
# strictly reduce its reward. These orderings are the precondition for ANY
# training run — if they fail, the optimum is degenerate again.
# Built on the real VGLC corpus (skipped when data/raw is absent).

from __future__ import annotations

import numpy as np
import pytest
from pcgrl.core.domain.level import AreaType, LevelGrid
from pcgrl.core.helpers.path import resolve_path
from pcgrl.runtime.metrics.intent import compute_intent_vector
from pcgrl.runtime.metrics.tpkl import PatternReference, count_patterns
from pcgrl.runtime.reward.composer_terminal import ComposerTerminalReward


@pytest.fixture(scope="module")
def corpus_levels(request):
    """Corpus levels."""
    config = request.getfixturevalue("config")
    catalog = request.getfixturevalue("catalog")
    corpus_root = resolve_path(config.dataset.corpus_root)
    if not corpus_root.is_dir() or not any(corpus_root.rglob("*.txt")):
        pytest.skip("Corpus files not present")
    from pcgrl.app.data.library.extract import load_raw_levels

    return load_raw_levels(corpus_root, ["mario-1"], catalog)


@pytest.fixture(scope="module")
def reference(corpus_levels, request):
    """Reference."""
    config = request.getfixturevalue("config")
    size = config.reward.composer.style_pattern_size
    counts: dict[bytes, float] = {}
    for level in corpus_levels[:10]:
        for key, value in count_patterns(level.tiles, size).items():
            counts[key] = counts.get(key, 0.0) + value
    return PatternReference({size: counts})


def _as_grid(tiles: np.ndarray, config, catalog) -> LevelGrid:
    """as grid."""
    height, width = tiles.shape
    grid = tiles.copy()
    # Stamp spawn + flag so playability and the intent flood have endpoints.
    grid[height - 3, 1] = catalog.id_for_name("player")
    grid[height - 3, width - 2] = catalog.id_for_name("flagpole")
    return LevelGrid(
        tiles=grid,
        width_tiles=width,
        height_tiles=height,
        tile_size_px=config.game.tile_size_px,
        area_type=AreaType.OVERWORLD,
        metadata={},
    )


@pytest.fixture(scope="module")
def harness(config, catalog, corpus_levels, reference):
    """Harness."""
    corpus = _as_grid(corpus_levels[0].tiles, config, catalog)
    rng = np.random.default_rng(0)

    confetti_tiles = corpus_levels[0].tiles.copy()
    sky = slice(0, confetti_tiles.shape[0] - 3)
    scatter = rng.random(confetti_tiles[sky].shape) < 0.25
    names = ["solid", "brick", "question_block_coin", "coin"]
    ids = np.array([catalog.id_for_name(n) for n in names], dtype=np.int16)
    region = confetti_tiles[sky]
    region[scatter] = rng.choice(ids, size=int(scatter.sum()))
    confetti_tiles[sky] = region
    confetti = _as_grid(confetti_tiles, config, catalog)

    empty_tiles = np.full_like(corpus_levels[0].tiles, catalog.id_for_name("empty"))
    empty_tiles[-2:, :] = catalog.id_for_name("solid")
    empty = _as_grid(empty_tiles, config, catalog)

    shuffled_tiles = corpus_levels[0].tiles.copy()
    flat = shuffled_tiles.flatten()
    rng.shuffle(flat)
    shuffled = _as_grid(flat.reshape(shuffled_tiles.shape), config, catalog)

    reward = ComposerTerminalReward(config, catalog, reference)
    return {
        "reward": reward,
        # Self-target: the corpus level asked for exactly what it is.
        "corpus_target": compute_intent_vector(corpus, catalog, config),
        # Generic mid-range intent: the ordering confetti > empty must hold for
        # a typical request, not for one tuned to either degenerate.
        "mid_target": np.full(8, 0.5, dtype=np.float32),
        "corpus": corpus,
        "confetti": confetti,
        "empty": empty,
        "shuffled": shuffled,
    }


def _score(harness, name: str, target_name: str = "corpus_target") -> float:
    """Verify corpus beats confetti decisively."""
    total, _ = harness["reward"].compute(harness[name], harness[target_name])
    return total


def test_corpus_beats_confetti_decisively(harness) -> None:
    """Verify corpus beats both under generic intent."""
    corpus = _score(harness, "corpus")
    confetti = _score(harness, "confetti")
    assert corpus > confetti + 1.0  # ">>", not just ">"


def test_corpus_beats_both_under_generic_intent(harness) -> None:
    """Verify corpus beats both under generic intent."""
    corpus = _score(harness, "corpus", "mid_target")
    assert corpus > _score(harness, "confetti", "mid_target")
    assert corpus > _score(harness, "empty", "mid_target")


def test_confetti_beats_empty(harness) -> None:
    """Verify confetti beats empty."""
    # Empty sky is a degenerate optimum; with the empty-pattern exclusion in
    # the style term it must rank BELOW even confetti for a generic mid-range
    # intent.
    confetti = _score(harness, "confetti", "mid_target")
    empty = _score(harness, "empty", "mid_target")
    assert confetti > empty


def test_shuffle_strictly_reduces_reward(harness) -> None:
    """Verify shuffle strictly reduces reward."""
    assert _score(harness, "corpus") > _score(harness, "shuffled")


def test_component_ranges(harness) -> None:
    """Verify duplicate penalty counts window repeats."""
    total, components = harness["reward"].compute(harness["corpus"], harness["corpus_target"])
    for name, value in components.items():
        assert 0.0 <= value <= 1.0, f"{name}={value}"
    settings_max = 4.0 + 2.0 + 3.0
    assert -1.0 <= total <= settings_max


def test_edited_flat_empty_wins_its_own_corner_target(harness, config, catalog) -> None:
    """For the flat-empty corner intent, a clean edited-flat level must beat
    the unedited corpus level.

    This is the edit-phase incentive: when the user asks for "flat, empty,
    smooth", alignment (weight 4) must dominate the style drop so the policy
    is rewarded for editing toward the request instead of shipping a generic
    corpus level. The classic orderings above still pin the opposite case
    (for corpus-typical targets the corpus material wins)."""
    corner = np.array([0, 0, 0, 0, 0, 0, 1.0, 0], dtype=np.float32)
    flat_score, _ = harness["reward"].compute(harness["empty"], corner)
    corpus_score, _ = harness["reward"].compute(harness["corpus"], corner)
    assert flat_score > corpus_score


def test_corpus_beats_its_flattened_self_on_corpus_target(harness, config, catalog) -> None:
    """For the corpus self-target, flattening the level must reduce reward.

    The mirror incentive: edits that destroy requested structure (alignment)
    and corpus style must never pay off, so the policy learns to edit only
    when the target demands it."""
    corpus_score, _ = harness["reward"].compute(harness["corpus"], harness["corpus_target"])
    flattened_score, _ = harness["reward"].compute(harness["empty"], harness["corpus_target"])
    assert corpus_score > flattened_score


def test_duplicate_penalty_counts_window_repeats(config, catalog, reference) -> None:
    """Verify duplicate penalty counts window repeats."""
    reward = ComposerTerminalReward(config, catalog, reference)
    assert reward._duplicate_penalty(()) == 0.0
    assert reward._duplicate_penalty((1, 2, 3, 4)) == 0.0
    assert reward._duplicate_penalty((1, 1, 1, 1)) == 1.0
    half = reward._duplicate_penalty((1, 1, 2, 3, 4, 4))
    assert 0.0 < half < 1.0
