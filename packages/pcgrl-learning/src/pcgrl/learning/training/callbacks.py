from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from pcgrl.bridge.bridge import GameplayBridge


class RewardComponentsCallback(BaseCallback):
    """Log per-step and terminal reward components to TensorBoard."""

    def _on_step(self) -> bool:
        """Record step and terminal reward components from rollout infos."""
        infos = self.locals.get("infos", [])
        if not infos:
            return True
        for info in infos:
            if "reward_components" in info:
                for key, value in info["reward_components"].items():
                    self.logger.record(f"reward/step/{key}", value)
            if "terminal_components" in info:
                for key, value in info["terminal_components"].items():
                    self.logger.record(f"reward/terminal/{key}", value)
        return True


class PeriodicEvalCallback(BaseCallback):
    """Periodically evaluate the policy and optionally log bridge win rates.

    Attributes:
        eval_every_steps: Minimum timesteps between evaluation passes.
        make_env: Factory returning a masked evaluation environment.
        n_eval_episodes: Episodes per evaluation pass.
        bridge: Optional gameplay bridge for win-rate logging.
        bridge_agent: Agent name passed to the bridge evaluator.
        bridge_timer_seconds: Per-run bridge time limit.
        bridge_mario_state: Initial Mario power-up state for bridge runs.
        bridge_num_runs: Stochastic bridge runs aggregated per episode.
    """

    def __init__(
        self,
        eval_every_steps: int,
        make_env: Callable[[], ActionMasker],
        n_eval_episodes: int = 1,
        bridge: GameplayBridge | None = None,
        bridge_agent: str = "robinBaumgarten",
        bridge_timer_seconds: int = 30,
        bridge_mario_state: int = 0,
        bridge_num_runs: int = 3,
        verbose: int = 0,
    ) -> None:
        """Store evaluation schedule, environment factory, and bridge settings."""
        super().__init__(verbose)
        self.eval_every_steps = max(0, eval_every_steps)
        self.make_env = make_env
        self.n_eval_episodes = max(1, n_eval_episodes)
        self.bridge = bridge
        self.bridge_agent = bridge_agent
        self.bridge_timer_seconds = bridge_timer_seconds
        self.bridge_mario_state = bridge_mario_state
        self.bridge_num_runs = max(1, bridge_num_runs)
        self._last_eval_step = 0

    def _on_step(self) -> bool:
        """Run periodic policy evaluation and log reward/bridge metrics."""
        if self.eval_every_steps <= 0:
            return True
        if self.num_timesteps - self._last_eval_step < self.eval_every_steps:
            return True
        self._last_eval_step = self.num_timesteps
        eval_env = DummyVecEnv([self.make_env])
        try:
            episode_rewards, episode_lengths = cast(
                tuple[list[float], list[int]],
                evaluate_policy(
                    cast(Any, self.model),
                    eval_env,
                    n_eval_episodes=self.n_eval_episodes,
                    deterministic=True,
                    return_episode_rewards=True,
                ),
            )
        finally:
            eval_env.close()
        if episode_rewards:
            self.logger.record(
                "eval/mean_episode_reward", float(sum(episode_rewards) / len(episode_rewards))
            )
        if episode_lengths:
            self.logger.record(
                "eval/mean_episode_length", float(sum(episode_lengths) / len(episode_lengths))
            )
        if self.bridge is not None:
            bridge_stats = self._evaluate_with_bridge()
            if bridge_stats is not None:
                self.logger.record("eval_bridge/win_rate", bridge_stats["win_rate"])
                self.logger.record("eval_bridge/completion_mean", bridge_stats["completion_mean"])
                self.logger.record("eval_bridge/completion_std", bridge_stats["completion_std"])
        return True

    def _evaluate_with_bridge(self) -> dict[str, float] | None:
        """Roll out eval episodes and aggregate bridge win/completion statistics."""
        if self.bridge is None:
            return None
        wins: list[float] = []
        completions: list[float] = []
        for _ in range(self.n_eval_episodes):
            env = self.make_env()
            try:
                obs, _ = env.reset()
                done = False
                while not done:
                    action_masks = env.action_masks()
                    action, _ = cast(Any, self.model).predict(
                        obs, action_masks=action_masks, deterministic=True
                    )
                    obs, _, terminated, truncated, _ = env.step(int(action))
                    done = bool(terminated or truncated)
                level = cast(Any, env.unwrapped).get_level()
                gameplay = self.bridge.evaluate(
                    level,
                    agent=self.bridge_agent,
                    timer_seconds=self.bridge_timer_seconds,
                    mario_state=self.bridge_mario_state,
                    num_runs=self.bridge_num_runs,
                )
                wins.append(
                    float(getattr(gameplay, "bridge_win_rate", 1.0 if gameplay.won else 0.0))
                )
                completions.append(float(gameplay.completion_ratio))
            finally:
                env.close()
        if not wins:
            return None
        return {
            "win_rate": float(np.mean(wins)),
            "completion_mean": float(np.mean(completions)),
            "completion_std": float(np.std(completions)),
        }
