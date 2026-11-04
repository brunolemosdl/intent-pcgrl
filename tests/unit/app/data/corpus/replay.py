from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.parametrize("seed", [0, 7, 123])
def test_corpus_replay_is_bottom_aligned(config, catalog, seed) -> None:
    """The raw-parse replay puts the ground on the bottom rows.

    The ground must land on the grid's last rows; this regression pins the
    bottom alignment.
    """
    from pcgrl.app.data.corpus.replay import corpus_replay_level

    level = corpus_replay_level(config, seed)
    assert level.height_tiles == config.game.height_tiles
    assert level.width_tiles == config.game.width_tiles
    solid_names = set(config.categories.get("solid_like", [])) | {"solid"}
    solid_ids = {
        catalog.id_for_name(name)
        for name in solid_names
        if name in catalog.name_to_id
    }
    ground = level.tiles[-1, :]
    solid_fraction = float(np.isin(ground, list(solid_ids)).mean())
    assert solid_fraction >= 0.8, f"ground row solid fraction {solid_fraction:.2f}"


def test_corpus_replay_has_provenance_and_endpoints(config, catalog) -> None:
    """Replay levels carry provenance metadata and stamped endpoints."""
    from pcgrl.app.data.corpus.replay import corpus_replay_level

    level = corpus_replay_level(config, 42)
    assert level.metadata["baseline"] == "corpus_replay"
    assert "source_level_id" in level.metadata
    player_id = catalog.id_for_name("player")
    flag_id = catalog.id_for_name("flagpole")
    assert np.count_nonzero(level.tiles == player_id) == 1
    assert np.count_nonzero(level.tiles == flag_id) >= 1
