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


def _require_corpus(config) -> None:
    """Skip when the raw corpus is not on disk."""
    corpus_root = resolve_path(config.dataset.corpus_root)
    if not corpus_root.is_dir() or not any(corpus_root.rglob("*.txt")):
        pytest.skip("Corpus files not present")


def test_run_prepare_creates_outputs(prepare_config) -> None:
    """Verify run prepare creates the composer artifacts."""
    _require_corpus(prepare_config)
    from pcgrl.app.pipeline.prepare import run_prepare

    result = run_prepare(prepare_config)
    assert result.is_dir()
    assert (result / "meta.json").is_file()
    assert (result / "corpus_profile.json").is_file()
    assert (result / "pattern_distributions.npz").is_file()
    # The composer pipeline never writes segment/label npz artifacts.
    assert not list(result.glob("*_segments.npz"))
    assert not list(result.glob("*_labels.npz"))


def test_run_prepare_meta_json_structure(prepare_config) -> None:
    """Verify run prepare meta json structure."""
    _require_corpus(prepare_config)
    import json

    from pcgrl.app.pipeline.prepare import run_prepare

    result = run_prepare(prepare_config)
    meta = json.loads((result / "meta.json").read_text())
    assert "level_count" in meta
    assert "tile_distribution_train" in meta
    assert "tile_catalog" in meta
    assert len(meta["tile_distribution_train"]) > 0


def test_run_prepare_no_corpus_raises(tmp_path) -> None:
    """Verify run prepare no corpus raises."""

    override = tmp_path / "override.yaml"
    override.write_text(
        f"dataset:\n  corpus_root: {tmp_path}/nonexistent\n  processed_root: {tmp_path}/processed\n"
    )
    config = load_config(training="initial", override_path=override)
    from pcgrl.app.pipeline.prepare import run_prepare

    with pytest.raises(FileNotFoundError):
        run_prepare(config)
