from __future__ import annotations

import json
import tempfile
from pathlib import Path

from pcgrl.learning.training.checkpoint import write_manifest


def test_write_manifest_creates_file() -> None:
    """Verify write manifest creates file."""
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        write_manifest(run_dir, {"total_steps": 1000})
        assert (run_dir / "manifest.json").is_file()


def test_write_manifest_returns_path() -> None:
    """Verify write manifest returns path."""
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        path = write_manifest(run_dir, {"key": "value"})
        assert path == run_dir / "manifest.json"


def test_write_manifest_content_is_json() -> None:
    """Verify write manifest content is json."""
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        payload = {"total_steps": 2000, "run_id": "abc123", "score": 0.95}
        write_manifest(run_dir, payload)
        data = json.loads((run_dir / "manifest.json").read_text())
        assert data["total_steps"] == 2000
        assert data["run_id"] == "abc123"
        assert data["score"] == 0.95


def test_write_manifest_nested_payload() -> None:
    """Verify write manifest nested payload."""
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        payload = {"metrics": {"auc": 0.92, "f1": 0.88}, "config": {"lr": 0.0001}}
        write_manifest(run_dir, payload)
        data = json.loads((run_dir / "manifest.json").read_text())
        assert data["metrics"]["auc"] == 0.92


def test_write_manifest_overwrites() -> None:
    """Verify write manifest overwrites."""
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        write_manifest(run_dir, {"v": 1})
        write_manifest(run_dir, {"v": 2})
        data = json.loads((run_dir / "manifest.json").read_text())
        assert data["v"] == 2
