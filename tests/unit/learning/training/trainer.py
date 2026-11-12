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


def test_latest_checkpoint_picks_most_steps(tmp_path) -> None:
    """Verify the periodic checkpoint with the most steps is selected."""
    from pcgrl.learning.training.trainer import latest_checkpoint

    assert latest_checkpoint(tmp_path) is None
    for name in ("policy_step_900_steps.zip", "policy_step_10000_steps.zip", "policy_final.zip"):
        (tmp_path / name).touch()
    assert latest_checkpoint(tmp_path) == (tmp_path / "policy_step_10000_steps.zip", 10000)


def test_trainer_continues_from_latest_checkpoint(tmp_path) -> None:
    """A resumed run trains only the steps its latest checkpoint is missing."""
    from pcgrl.core.settings.loader import load_config
    from pcgrl.learning.training.trainer import PcgrlTrainer
    from sb3_contrib import MaskablePPO

    def train(total_steps: int, *, resume: bool) -> MaskablePPO:
        override = tmp_path / "override.yaml"
        override.write_text(
            "training:\n"
            f"  total_steps: {total_steps}\n"
            "  rollout_steps: 64\n"
            "  num_environments: 1\n"
            "  batch_size: 64\n"
            "  num_epochs: 1\n"
            "  save_interval_steps: 64\n"
        )
        cfg = load_config(training="initial", override_path=override)
        result = PcgrlTrainer(cfg, tmp_path / "run", resume=resume).train()
        return MaskablePPO.load(str(result.checkpoint_path), device="cpu")

    assert train(64, resume=False).num_timesteps == 64
    assert train(192, resume=True).num_timesteps == 192
    checkpoints = sorted(path.name for path in (tmp_path / "run" / "models").glob("policy_step_*"))
    assert checkpoints == [
        "policy_step_128_steps.zip",
        "policy_step_192_steps.zip",
        "policy_step_64_steps.zip",
    ]
