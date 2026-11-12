from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Locate the project root from an explicit start path or environment override.

    Args:
        start: Optional path to begin the upward search (defaults to this module).

    Returns:
        Resolved repository root directory.

    Raises:
        FileNotFoundError: When no directory contains ``pyproject.toml`` and
            ``configs/mario.yaml``.
    """
    env_root = os.environ.get("PCGRL_REPO_ROOT")
    if env_root:
        return Path(env_root).resolve()
    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "configs" / "mario.yaml"
        ).is_file():
            return candidate
    raise FileNotFoundError("Could not locate repository root.")


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """Return the cached repository root path."""
    return find_repo_root()


def resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to the repository root unless already absolute.

    Args:
        path: Relative or absolute filesystem path.

    Returns:
        Absolute resolved path.
    """
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (repo_root() / candidate).resolve()


def ensure_dir(path: str | Path) -> Path:
    """Create a directory tree when missing and return the resolved path.

    Args:
        path: Directory path relative to the repo root or absolute.

    Returns:
        Resolved directory path after creation.
    """
    resolved = resolve_path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def asset_path(*parts: str) -> Path:
    """Resolve a path under the project ``assets/`` directory.

    Args:
        *parts: Path components appended after ``assets/``.

    Returns:
        Absolute path to the requested asset location.
    """
    return resolve_path(Path("assets").joinpath(*parts))


def vendor_path(*parts: str) -> Path:
    """Resolve a path under the project ``vendor/`` directory.

    Args:
        *parts: Path components appended after ``vendor/``.

    Returns:
        Absolute path to the requested vendor resource.
    """
    return resolve_path(Path("vendor").joinpath(*parts))
