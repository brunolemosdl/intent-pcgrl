from __future__ import annotations

import pytest
from pcgrl.core.helpers.path import resolve_path
from pcgrl.core.settings.loader import load_config


@pytest.fixture
def prepare_config(tmp_path):
    """Prepare config."""

    override = tmp_path / "override.yaml"
    override.write_text(f"dataset:\n  processed_root: {tmp_path}/processed\n")
    return load_config(training="initial", override_path=override)


def test_prepare_writes_artifacts(prepare_config) -> None:
    """Verify prepare writes artifacts."""
    corpus_root = resolve_path(prepare_config.dataset.corpus_root)
    if not corpus_root.is_dir() or not any(corpus_root.rglob("*.txt")):
        pytest.skip("Corpus files not present")
    from pcgrl.app.pipeline.prepare import run_prepare

    out = run_prepare(prepare_config)
    assert (out / "meta.json").is_file()
    assert (out / "corpus_profile.json").is_file()
    assert (out / "pattern_distributions.npz").is_file()


def test_prepare_meta_json_structure(prepare_config) -> None:
    """Verify prepare meta json structure."""
    corpus_root = resolve_path(prepare_config.dataset.corpus_root)
    if not corpus_root.is_dir() or not any(corpus_root.rglob("*.txt")):
        pytest.skip("Corpus files not present")
    import json

    from pcgrl.app.pipeline.prepare import run_prepare

    out = run_prepare(prepare_config)
    meta = json.loads((out / "meta.json").read_text())
    assert "level_count" in meta
    assert "tile_distribution_train" in meta
    assert "tile_catalog" in meta
