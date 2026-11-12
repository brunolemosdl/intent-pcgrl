from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_run_train_creates_run_dir(config, tmp_path) -> None:
    """Verify run train creates run dir."""
    from pcgrl.core.domain.contract import TrainResult
    from pcgrl.learning.training.runner import run_train

    expected_checkpoint = tmp_path / "model.zip"
    mock_result = TrainResult(
        run_dir=tmp_path, checkpoint_path=expected_checkpoint, total_steps=1000
    )
    with (
        patch("pcgrl.learning.training.runner.PcgrlTrainer") as mock_trainer_cls,
        patch("pcgrl.learning.training.runner.resolve_and_snapshot"),
        patch("pcgrl.learning.training.runner.resolve_path", return_value=tmp_path),
        patch(
            "pcgrl.learning.training.runner.ensure_dir",
            side_effect=lambda p: p.mkdir(parents=True, exist_ok=True) or p,
        ),
    ):
        instance = MagicMock()
        instance.train.return_value = mock_result
        mock_trainer_cls.return_value = instance
        result = run_train(config)
    assert result == expected_checkpoint


def test_run_train_with_explicit_run_dir(config, tmp_path) -> None:
    """Verify run train with explicit run dir."""
    from pcgrl.core.domain.contract import TrainResult
    from pcgrl.learning.training.runner import run_train

    run_dir = tmp_path / "my_run"
    expected_checkpoint = run_dir / "models" / "final.zip"
    mock_result = TrainResult(
        run_dir=run_dir, checkpoint_path=expected_checkpoint, total_steps=1000
    )
    with (
        patch("pcgrl.learning.training.runner.PcgrlTrainer") as mock_trainer_cls,
        patch("pcgrl.learning.training.runner.resolve_and_snapshot"),
    ):
        instance = MagicMock()
        instance.train.return_value = mock_result
        mock_trainer_cls.return_value = instance
        result = run_train(config, run_dir=run_dir)
    assert result == expected_checkpoint
