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


def latest_checkpoint(models_dir: Path) -> tuple[Path, int] | None:
    """Return the periodic checkpoint with the most steps and its step count."""
    found: list[tuple[int, Path]] = []
    for path in models_dir.glob("policy_step_*_steps.zip"):
        steps = path.stem.removeprefix("policy_step_").removesuffix("_steps")
        if steps.isdigit():
            found.append((int(steps), path))
    if not found:
        return None
    steps_done, checkpoint = max(found)
    return checkpoint, steps_done


class PcgrlTrainer:
    """Train a MaskablePPO policy for the composer environment.

    Description:
        Wraps Stable-Baselines3 training with the project env factory,
        callbacks, and checkpointing. The composer episode is ~10-19 terminal-
        rewarded decisions, so the default config trains with ``gamma = 1.0``
        and a flat MLP policy. A resumed run continues from the latest
        periodic checkpoint in its directory up to the configured total.

    Attributes:
        _config: Frozen runtime configuration for the run.
        _run_dir: Artifact directory for logs, TensorBoard, and checkpoints.
        _resume: Whether to continue from the latest periodic checkpoint.
    """

    def __init__(self, config: RuntimeConfig, run_dir: Path, *, resume: bool = False) -> None:
        """Store configuration, the run artifact directory and the resume flag."""
        self._config = config
        self._run_dir = run_dir
        self._resume = resume

    def train(self) -> TrainResult:
        """Run MaskablePPO training and persist the final checkpoint.

        Returns:
            Paths and step count for the completed training run.
        """
        config = self._config
        models_dir = ensure_dir(self._run_dir / "models")
        resumed = latest_checkpoint(models_dir) if self._resume else None
        # A resumed run draws new episodes instead of replaying the first ones.
        env_seed = config.experiment.seed + (resumed[1] if resumed else 0)
        num_environments = config.training.num_environments
        vec_env: VecEnv
        if num_environments == 1:
            vec_env = DummyVecEnv([_make_env(config, env_seed)])
        else:
            vec_env = SubprocVecEnv(
                [_make_env(config, env_seed + index) for index in range(num_environments)]
            )
        policy_kwargs = {
            "net_arch": {
                "pi": [config.model.actor.get("hidden_dim", 256)] * 2,
                "vf": [config.model.critic.get("value_hidden_dim", 256)] * 2,
            },
        }
        tensorboard_log = str(self._run_dir / "tensorboard")
        if resumed is None:
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
                # A two-layer MLP steps faster on the CPU than through per-step
                # GPU transfers; rollouts are bound by the environments anyway.
                device="cpu",
                tensorboard_log=tensorboard_log,
                verbose=1,
            )
        else:
            # The checkpoint carries the weights, the optimizer state, the
            # step counter and the hyperparameters the run started with.
            model = MaskablePPO.load(
                str(resumed[0]), env=vec_env, device="cpu", tensorboard_log=tensorboard_log
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
            total_timesteps=max(0, config.training.total_steps - model.num_timesteps),
            callback=CallbackList(callbacks),
            log_interval=config.training.log_interval,
            reset_num_timesteps=resumed is None,
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
