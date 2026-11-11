from __future__ import annotations

import json
from pathlib import Path


def write_manifest(run_dir: Path, payload: dict[str, object]) -> Path:
    """Write a JSON manifest summarizing a training run.

    Args:
        run_dir: Artifact directory for the run.
        payload: Serializable manifest fields (checkpoint path, step count, etc.).

    Returns:
        Path to the written ``manifest.json``.

    Raises:
        OSError: When the manifest file cannot be written.
    """
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
