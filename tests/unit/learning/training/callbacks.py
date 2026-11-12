from __future__ import annotations

from unittest.mock import MagicMock, patch

from pcgrl.learning.training.callbacks import PeriodicEvalCallback, RewardComponentsCallback


def _make_callback() -> RewardComponentsCallback:
    """make callback."""
    callback = RewardComponentsCallback()
    callback.model = MagicMock()
    callback.model.logger = MagicMock()
    callback.locals = {}
    return callback


def test_callback_on_step_no_infos() -> None:
    """Verify callback on step no infos."""
    callback = _make_callback()
    callback.locals = {"infos": []}
    result = callback._on_step()
    assert result is True
    callback.model.logger.record.assert_not_called()


def test_callback_on_step_with_reward_components() -> None:
    """Verify callback on step with reward components."""
    callback = _make_callback()
    callback.locals = {"infos": [{"reward_components": {"legal": 0.01, "progress": 0.5}}]}
    result = callback._on_step()
    assert result is True
    callback.model.logger.record.assert_any_call("reward/step/legal", 0.01)
    callback.model.logger.record.assert_any_call("reward/step/progress", 0.5)


def test_callback_on_step_with_terminal_components() -> None:
    """Verify callback on step with terminal components."""
    callback = _make_callback()
    callback.locals = {"infos": [{"terminal_components": {"valid": 5.0, "play": 3.0}}]}
    result = callback._on_step()
    assert result is True
    callback.model.logger.record.assert_any_call("reward/terminal/valid", 5.0)
    callback.model.logger.record.assert_any_call("reward/terminal/play", 3.0)


def test_callback_on_step_multiple_infos() -> None:
    """Verify callback on step multiple infos."""
    callback = _make_callback()
    callback.locals = {
        "infos": [{"reward_components": {"legal": 0.01}}, {"reward_components": {"legal": 0.02}}]
    }
    result = callback._on_step()
    assert result is True
    assert callback.model.logger.record.call_count == 2


def test_callback_on_step_empty_locals() -> None:
    """Verify callback on step empty locals."""
    callback = _make_callback()
    callback.locals = {}
    result = callback._on_step()
    assert result is True


def test_callback_on_step_info_without_components() -> None:
    """Verify callback on step info without components."""
    callback = _make_callback()
    callback.locals = {"infos": [{"action_mask": [True, False]}]}
    result = callback._on_step()
    assert result is True
    callback.model.logger.record.assert_not_called()


def test_periodic_eval_bridge_stats_logged() -> None:
    """Verify periodic eval bridge stats logged."""
    env = MagicMock()
    env.reset.return_value = ("obs", {})
    env.action_masks.return_value = [True, True]
    # One step episode.
    env.step.return_value = ("obs2", 1.0, True, False, {})
    raw_env = MagicMock()
    raw_env.get_level.return_value = "level"
    env.env = raw_env

    bridge = MagicMock()
    gameplay = MagicMock()
    gameplay.won = True
    gameplay.bridge_win_rate = 1.0
    gameplay.completion_ratio = 1.0
    bridge.evaluate.return_value = gameplay

    cb = PeriodicEvalCallback(
        eval_every_steps=1,
        make_env=lambda: env,
        n_eval_episodes=1,
        bridge=bridge,
    )
    cb.model = MagicMock()
    cb.model.predict.return_value = (1, None)

    stats = cb._evaluate_with_bridge()
    assert stats is not None
    assert stats["win_rate"] == 1.0
    assert stats["completion_mean"] == 1.0


def test_periodic_eval_interval_counts_from_the_starting_step() -> None:
    """Verify a run that starts at a later step waits a full interval."""
    cb = PeriodicEvalCallback(eval_every_steps=100, make_env=MagicMock())
    cb.num_timesteps = 600
    cb._on_training_start()
    cb.num_timesteps = 650
    with patch("pcgrl.learning.training.callbacks.evaluate_policy") as evaluate:
        assert cb._on_step() is True
        evaluate.assert_not_called()
