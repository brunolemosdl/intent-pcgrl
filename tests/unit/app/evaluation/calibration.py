from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pcgrl.core.helpers.path import resolve_path
from pcgrl.core.settings.loader import load_config


@pytest.fixture(scope="module")
def calibration_config():
    """Calibration config."""
    config = load_config(training="initial", apply_profile=False)
    corpus_root = resolve_path(config.dataset.corpus_root)
    if not corpus_root.is_dir() or not any(corpus_root.rglob("*.txt")):
        pytest.skip("Corpus files not present")
    if not (resolve_path(config.library.output_root) / "chunk_library.npz").is_file():
        pytest.skip("Chunk library not built")
    return config


def test_build_calibration_pool_mixes_origins(calibration_config) -> None:
    """Verify build calibration pool mixes origins."""
    from pcgrl.app.evaluation.calibration import build_calibration_pool
    from pcgrl.core.tiles.catalog import load_catalog

    catalog = load_catalog(calibration_config)
    pool = build_calibration_pool(calibration_config, catalog, num_levels=9, seed=1)
    assert len(pool) == 9
    origins = {level.metadata.get("calibration_origin", "composed") for level in pool}
    assert {"corpus", "perturbed"} <= origins
    player = catalog.id_for_name("player")
    flag = catalog.id_for_name("flagpole")
    for level in pool:
        assert (level.tiles == player).sum() == 1
        assert (level.tiles == flag).sum() == 1


def test_run_calibration_with_mocked_bridge(calibration_config, tmp_path) -> None:
    """Verify run calibration with mocked bridge."""
    from pcgrl.app.evaluation import calibration as module

    mock_bridge = MagicMock()
    mock_bridge.evaluate.return_value = SimpleNamespace(won=True)
    with patch.object(module, "GameplayBridge", return_value=mock_bridge):
        report_path = module.run_calibration(
            calibration_config, num_levels=6, output_dir=tmp_path
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["num_levels"] == 6
    assert report["bridge_win_rate"] == 1.0
    grid = report["grid"]
    expected = len(
        calibration_config.evaluation.calibration.jump_height_candidates
    ) * len(calibration_config.evaluation.calibration.jump_reach_candidates)
    assert len(grid) == expected
    for entry in grid:
        assert entry["tp"] + entry["fp"] + entry["tn"] + entry["fn"] == 6
        # with all-true ground truth, false_playable is undefined-but-zero
        assert entry["false_playable_rate"] == 0.0
    assert report["recommended"] is not None
