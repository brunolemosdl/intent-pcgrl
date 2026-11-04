from __future__ import annotations

import os
from pathlib import Path

from pcgrl.core.helpers.path import PROJECT_ROOT


def load_env_file(path: Path | None = None) -> int:
    """Load ``KEY=VALUE`` pairs from a ``.env`` file into the environment.

    Description:
        Dependency-free stand-in for python-dotenv: blank lines and ``#``
        comments are skipped, surrounding quotes are stripped, and variables
        already present in the environment are NEVER overridden (the shell
        wins over the file).

    Args:
        path: Env file location; defaults to ``<repo root>/.env``.

    Returns:
        Number of variables actually set.
    """
    env_path = path or PROJECT_ROOT / ".env"
    if not env_path.is_file():
        return 0
    loaded = 0
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if not key or key in os.environ or not value:
            continue
        os.environ[key] = value
        loaded += 1
    return loaded
