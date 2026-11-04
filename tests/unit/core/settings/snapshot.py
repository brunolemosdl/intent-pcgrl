from __future__ import annotations

import tempfile
from pathlib import Path

import yaml
from pcgrl.core.settings.loader import resolve_and_snapshot


def test_resolve_and_snapshot_creates_files(config) -> None:
    """Verify resolve and snapshot creates files."""
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        resolve_and_snapshot(config, run_dir=run_dir)
        assert (run_dir / "config.snapshot.yaml").is_file()
        assert (run_dir / "seed.txt").is_file()


def test_resolve_and_snapshot_returns_path(config) -> None:
    """Verify resolve and snapshot returns path."""
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        path = resolve_and_snapshot(config, run_dir=run_dir)
        assert path == run_dir / "config.snapshot.yaml"


def test_resolve_and_snapshot_seed_matches(config) -> None:
    """Verify resolve and snapshot seed matches."""
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        resolve_and_snapshot(config, run_dir=run_dir)
        seed_text = (run_dir / "seed.txt").read_text()
        assert int(seed_text) == config.experiment.seed


def test_resolve_and_snapshot_yaml_has_config(config) -> None:
    """Verify resolve and snapshot yaml has config."""
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        resolve_and_snapshot(config, run_dir=run_dir)
        data = yaml.safe_load((run_dir / "config.snapshot.yaml").read_text())
        assert "config" in data
        assert "provenance" in data


def test_resolve_and_snapshot_creates_run_dir(config) -> None:
    """Verify resolve and snapshot creates run dir."""
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "new_nested_dir"
        resolve_and_snapshot(config, run_dir=run_dir)
        assert run_dir.is_dir()
