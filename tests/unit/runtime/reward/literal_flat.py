from __future__ import annotations

import numpy as np
import pytest
from pcgrl.core.helpers.path import resolve_path
from pcgrl.runtime.metrics.intent import INTENT_KEYS, compute_intent_vector
from pcgrl.runtime.metrics.tpkl import PatternReference
from pcgrl.runtime.refine.context import build_context
from pcgrl.runtime.refine.corrections import relief_fraction
from pcgrl.runtime.refine.gates import grid_from_tiles
from pcgrl.runtime.reward.composer_terminal import ComposerTerminalReward

_FLAT = np.array([0, 0, 0, 0, 0, 0, 1.0, 0], dtype=np.float32)


@pytest.fixture
def ctx(config, catalog):
    """Shared refine context for the session config."""
    return build_context(config, catalog)


@pytest.fixture
def terminal(config, catalog):
    """Terminal reward over the processed corpus patterns."""
    root = resolve_path(config.dataset.processed_root)
    if not root.is_dir():
        pytest.skip("Processed corpus not present")
    return ComposerTerminalReward(config, catalog, PatternReference.load(root, split="train"))


def _floor(config, catalog, ctx) -> np.ndarray:
    """One-row floor with the endpoints stamped."""
    height, width = config.game.height_tiles, config.game.width_tiles
    tiles = np.full((height, width), ctx.empty_id, dtype=np.int16)
    tiles[height - ctx.base_height :, :] = ctx.solid_id
    tiles[ctx.player] = catalog.id_for_name("player")
    tiles[ctx.flagpole] = catalog.id_for_name("flagpole")
    return tiles


def _mound(tiles: np.ndarray, ctx) -> np.ndarray:
    """The floor with a low six-column mound, inside the flat band."""
    raised = tiles.copy()
    raised[tiles.shape[0] - 3 :, 60:66] = ctx.solid_id
    return raised


def test_relief_fraction_counts_columns_off_the_floor(config, catalog, ctx) -> None:
    """A level floor has no relief; a mound counts its columns."""
    floor = _floor(config, catalog, ctx)
    assert relief_fraction(floor, ctx) == 0.0
    counted = config.game.width_tiles - len(ctx.protected_columns)
    assert relief_fraction(_mound(floor, ctx), ctx) == pytest.approx(6 / counted)
    floating = floor.copy()
    floating[5, 80:83] = ctx.solid_id
    assert relief_fraction(floating, ctx) == pytest.approx(3 / counted)


def test_flat_request_is_scored_literally(config, catalog, ctx, terminal) -> None:
    """Relief the banded metric cannot see lowers the reward of a flat request.

    Both levels measure the same intent vector; only the literal reading of
    the flat-and-smooth request tells them apart. A request that is not
    literally flat scores them by the vector alone."""
    floor = _floor(config, catalog, ctx)
    mound = _mound(floor, ctx)
    flat_level = grid_from_tiles(floor, config)
    mound_level = grid_from_tiles(mound, config)
    np.testing.assert_allclose(
        compute_intent_vector(flat_level, catalog, config),
        compute_intent_vector(mound_level, catalog, config),
    )
    _, flat_parts = terminal.compute(flat_level, _FLAT)
    _, mound_parts = terminal.compute(mound_level, _FLAT)
    assert mound_parts["alignment"] < flat_parts["alignment"]
    hilly = _FLAT.copy()
    hilly[INTENT_KEYS.index("verticality")] = 0.3
    _, flat_parts = terminal.compute(flat_level, hilly)
    _, mound_parts = terminal.compute(mound_level, hilly)
    assert mound_parts["alignment"] == pytest.approx(flat_parts["alignment"])
