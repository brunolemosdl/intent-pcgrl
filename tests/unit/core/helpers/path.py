from __future__ import annotations

import tempfile
from pathlib import Path

from pcgrl.core.helpers.path import PROJECT_ROOT, ensure_dir, resolve_path


def test_project_root_is_directory() -> None:
    """Verify project root is directory."""
    assert PROJECT_ROOT.is_dir()


def test_project_root_contains_pyproject() -> None:
    """Verify project root contains pyproject."""
    assert (PROJECT_ROOT / "pyproject.toml").is_file()


def test_resolve_path_absolute() -> None:
    """Verify resolve path absolute."""
    abs_path = Path("/tmp/some/path")
    result = resolve_path(abs_path)
    assert result.is_absolute()
    assert str(result).startswith("/tmp")


def test_resolve_path_relative_anchors_to_project() -> None:
    """Verify resolve path relative anchors to project."""
    result = resolve_path("pyproject.toml")
    assert result == PROJECT_ROOT / "pyproject.toml"


def test_resolve_path_string_input() -> None:
    """Verify resolve path string input."""
    result = resolve_path("configs/mario.yaml")
    assert result.is_absolute()
    assert result.is_file()


def test_ensure_dir_creates_directory() -> None:
    """Verify ensure dir creates directory."""
    with tempfile.TemporaryDirectory() as tmp:
        new_dir = Path(tmp) / "a" / "b" / "c"
        result = ensure_dir(new_dir)
        assert result.is_dir()


def test_ensure_dir_returns_path() -> None:
    """Verify ensure dir idempotent."""
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "nested"
        returned = ensure_dir(target)
        assert returned == target.resolve()


def test_ensure_dir_idempotent() -> None:
    """Verify ensure dir idempotent."""
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "existing"
        ensure_dir(target)
        ensure_dir(target)
        assert target.is_dir()
