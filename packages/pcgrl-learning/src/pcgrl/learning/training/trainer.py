from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from pcgrl.bridge.bridge import GameplayBridge
from pcgrl.core.domain.contract import TrainResult
from pcgrl.core.helpers.path import ensure_dir
from pcgrl.core.settings.schema import RuntimeConfig
from pcgrl.learning.env_factory import build_env_context, make_env
from pcgrl.learning.training.callbacks import PeriodicEvalCallback, RewardComponentsCallback
from pcgrl.learning.training.checkpoint import write_manifest


def _make_env(config: RuntimeConfig, seed: int) -> Callable[[], ActionMasker]:
    """Return a vec-env factory that builds a masked, monitored composer env."""

    def _init() -> ActionMasker:
        """Construct a seeded, masked, monitored environment instance."""
        context = build_env_context(config)
        env_seed = seed * config.environment.parallel_env_seed_stride % 2**31
        base_env = make_env(config, context, seed=env_seed)
        return ActionMasker(Monitor(base_env), lambda _: base_env.action_masks())

    return _init


class PcgrlTrainer:
    """Train a MaskablePPO policy for the composer environment.

    Description:
        Wraps Stable-Baselines3 training with the project env factory,
        callbacks, and checkpointing. The composer episode is ~10-19 terminal-
        rewarded decisions, so the default config trains with ``gamma = 1.0``
        and a flat MLP policy.

    Attributes:
        _config: Frozen runtime configuration for the run.
        _run_dir: Artifact directory for logs, TensorBoard, and checkpoints.
    """

    def __init__(self, config: RuntimeConfig, run_dir: Path) -> None:
        """Store configuration and the run artifact directory."""
        self._config = config
        self._run_dir = run_dir

    def train(self) -> TrainResult:
        """Run MaskablePPO training and persist the final checkpoint.

        Returns:
            Paths and step count for the completed training run.
        """
        config = self._config
        models_dir = ensure_dir(self._run_dir / "models")
        num_environments = config.training.num_environments
        vec_env: VecEnv
        if num_environments == 1:
            vec_env = DummyVecEnv([_make_env(config, config.experiment.seed)])
        else:
            vec_env = SubprocVecEnv(
                [
                    _make_env(config, config.experiment.seed + index)
                    for index in range(num_environments)
                ]
            )
        policy_kwargs = {
            "net_arch": {
                "pi": [config.model.actor.get("hidden_dim", 256)] * 2,
                "vf": [config.model.critic.get("value_hidden_dim", 256)] * 2,
            },
        }
        model = MaskablePPO(
            MaskableActorCriticPolicy,
            vec_env,
            learning_rate=config.training.learning_rate,
            n_steps=config.training.rollout_steps,
            batch_size=config.training.batch_size,
            n_epochs=config.training.num_epochs,
            gamma=config.training.gamma,
            gae_lambda=config.training.gae_lambda,
            clip_range=config.training.clip_range,
            ent_coef=config.training.entropy_coefficient,
            vf_coef=config.training.value_function_coefficient,
            max_grad_norm=config.training.max_grad_norm,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(self._run_dir / "tensorboard"),
            verbose=1,
        )
        callbacks = [
            CheckpointCallback(
                save_freq=max(1, config.training.save_interval_steps // max(1, num_environments)),
                save_path=str(models_dir),
                name_prefix="policy_step",
            ),
            RewardComponentsCallback(),
        ]
        if config.evaluation.eval_every_steps > 0:
            bridge = None
            if config.evaluation.require_gameplay_bridge:
                candidate = GameplayBridge(config)
                if candidate.jar_path.is_file():
                    bridge = candidate
            callbacks.append(
                PeriodicEvalCallback(
                    eval_every_steps=config.evaluation.eval_every_steps,
                    make_env=_make_env(config, config.experiment.seed + 10_000),
                    n_eval_episodes=config.evaluation.eval_episodes,
                    bridge=bridge,
                    bridge_agent=config.evaluation.agents[0],
                    bridge_timer_seconds=config.evaluation.timer_seconds,
                    bridge_mario_state=config.evaluation.mario_state,
                    bridge_num_runs=config.evaluation.num_runs,
                )
            )
        model.learn(
            total_timesteps=config.training.total_steps,
            callback=CallbackList(callbacks),
            log_interval=config.training.log_interval,
        )
        final_path = models_dir / "policy_final.zip"
        model.save(str(final_path))
        write_manifest(
            self._run_dir, {"checkpoint": str(final_path), "steps": config.training.total_steps}
        )
        vec_env.close()
        return TrainResult(
            run_dir=self._run_dir,
            checkpoint_path=final_path,
            total_steps=config.training.total_steps,
        )
