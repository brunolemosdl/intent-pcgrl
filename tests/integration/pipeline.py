from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pcgrl.core.settings.loader import load_config


def test_generate_pipeline_end_to_end(tmp_path) -> None:
    """Verify generate pipeline end to end."""
    from types import SimpleNamespace

    from pcgrl.core.domain.level import AreaType, LevelGrid
    from pcgrl.core.tiles.catalog import load_catalog
    from pcgrl.learning.generate import run_generate

    config = load_config(training="initial")
    catalog = load_catalog(config)
    h, w = (config.game.height_tiles, config.game.width_tiles)
    empty_id = catalog.id_for_name("empty")
    tiles = np.full((h, w), empty_id, dtype=np.int16)
    mock_level = LevelGrid(
        tiles=tiles,
        width_tiles=w,
        height_tiles=h,
        tile_size_px=config.game.tile_size_px,
        area_type=AreaType.OVERWORLD,
    )
    fake_ckpt = tmp_path / "model.zip"
    fake_ckpt.write_bytes(b"fake")
    mock_env = MagicMock()
    obs = np.zeros(10, dtype=np.float32)
    mock_env.reset.return_value = (obs, {})
    mock_env.action_masks.return_value = np.ones(10, dtype=bool)
    # An empty level has no flag: the flood gate must fail every candidate.
    mock_env.step.return_value = (
        obs,
        0.0,
        True,
        False,
        {"reward_components": {"playability": 0.0}, "chunk_ids": [0]},
    )
    mock_env.get_level.return_value = mock_level
    terminal = MagicMock()
    terminal.compute.return_value = (0.0, {"playability": 0.0})
    context = SimpleNamespace(catalog=catalog, composer_terminal=terminal)
    with (
        patch("pcgrl.learning.generate.MaskablePPO") as mock_ppo_cls,
        patch("pcgrl.learning.generate.build_env_context", return_value=context),
        patch("pcgrl.learning.generate.make_env", return_value=mock_env),
    ):
        mock_ppo = MagicMock()
        mock_ppo.predict.return_value = (0, None)
        mock_ppo_cls.load.return_value = mock_ppo
        with pytest.raises(RuntimeError, match="Generation failed"):
            run_generate(config, fake_ckpt, output_dir=tmp_path)
        out = next(tmp_path.glob("level_*.json"))
    assert out.is_file()
    data = json.loads(out.read_text())
    assert "target" in data
    assert "ascii" in data
