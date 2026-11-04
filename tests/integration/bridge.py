from __future__ import annotations

import pytest
from pcgrl.core.helpers.path import resolve_path
from pcgrl.core.settings.loader import load_config
from pcgrl.core.tiles.catalog import load_catalog

_JAR = resolve_path("vendor/mario/mario-bridge.jar")
_LEVEL = resolve_path("data/raw/corpus/mario/mario-1/1-1.txt")


@pytest.mark.skipif(not _JAR.is_file(), reason="mario-bridge.jar not built")
@pytest.mark.skipif(not _LEVEL.is_file(), reason="Corpus level file not present")
def test_gameplay_bridge_runs_on_vglc_level() -> None:
    """Verify gameplay bridge runs on vglc level."""
    from pcgrl.app.data.corpus.parser import parse_level_file
    from pcgrl.bridge.bridge import GameplayBridge

    config = load_config(training="initial")
    catalog = load_catalog(config)
    level = parse_level_file(_LEVEL, catalog, tile_size_px=config.game.tile_size_px)
    bridge = GameplayBridge(config)
    result = bridge.evaluate(level, num_runs=1)
    assert result.game_status in {"WIN", "LOSE", "TIME_OUT", "RUNNING"}
