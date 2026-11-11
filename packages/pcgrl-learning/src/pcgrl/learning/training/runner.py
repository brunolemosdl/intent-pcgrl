from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from pcgrl.core.helpers.log import setup_logging
from pcgrl.core.helpers.path import ensure_dir, resolve_path
from pcgrl.core.settings.loader import resolve_and_snapshot
from pcgrl.core.settings.schema import RuntimeConfig
from pcgrl.learning.training.trainer import PcgrlTrainer


def run_train(config: RuntimeConfig, *, run_dir: Path | None = None) -> Path:
    """Train a policy and return the final checkpoint path.

    Args:
        config: Runtime settings for the training run.
        run_dir: Existing or new artifact directory; auto-created when omitted.

    Returns:
        Path to ``policy_final.zip`` inside the run directory.

    Raises:
        OSError: When run artifacts cannot be written.
    """
    if run_dir is None:
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
        run_dir = ensure_dir(
            resolve_path(config.experiment.artifacts_root) / config.experiment.name / timestamp
        )
    else:
        ensure_dir(run_dir)
    resolve_and_snapshot(config, run_dir=run_dir)
    setup_logging(config.log.level, run_dir / "logs" / "run.log")
    trainer = PcgrlTrainer(config, run_dir)
    result = trainer.train()
    return result.checkpoint_path
