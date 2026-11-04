from __future__ import annotations

from unittest.mock import patch


def test_make_env_returns_callable(config) -> None:
    """Verify make env returns callable."""
    from pcgrl.learning.training.trainer import _make_env

    factory = _make_env(config, seed=42)
    assert callable(factory)


def test_trainer_train_mocked(config, tmp_path) -> None:
    """Verify trainer train mocked."""
    from pcgrl.core.domain.contract import TrainResult
    from pcgrl.learning.training.trainer import PcgrlTrainer

    trainer = PcgrlTrainer(config, tmp_path)
    mock_result = TrainResult(
        run_dir=tmp_path, checkpoint_path=tmp_path / "model.zip", total_steps=1000
    )
    with patch.object(PcgrlTrainer, "train", return_value=mock_result) as mock_train:
        result = trainer.train()
        mock_train.assert_called_once()
    assert result.run_dir == tmp_path


def test_trainer_train_minimal_steps(tmp_path) -> None:
    """Train the composer policy for a few real steps against the library."""
    from pcgrl.core.settings.loader import load_config
    from pcgrl.learning.training.trainer import PcgrlTrainer

    override = tmp_path / "override.yaml"
    override.write_text(
        "training:\n"
        "  total_steps: 64\n"
        "  rollout_steps: 64\n"
        "  num_environments: 1\n"
        "  batch_size: 64\n"
        "  num_epochs: 1\n"
    )
    cfg = load_config(training="initial", override_path=override)
    trainer = PcgrlTrainer(cfg, tmp_path)
    result = trainer.train()
    assert result.checkpoint_path.is_file()
