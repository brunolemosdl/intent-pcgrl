from __future__ import annotations

import numpy as np
from pcgrl.core.settings.loader import load_config
from pcgrl.learning.env_factory import build_env_context, make_env


def test_env_reset_and_step() -> None:
    """Verify composer env reset and step against the real library artifacts."""
    config = load_config(training="initial")
    context = build_env_context(config)
    env = make_env(config, context, seed=0)
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert info["action_mask"].any()
    rng = np.random.default_rng(0)
    done = False
    step_count = 0
    max_steps = env._max_decisions + config.environment.composer.edit_budget
    while not done and step_count < max_steps:
        mask = env.action_masks()
        assert mask.any(), "At least one action must be valid"
        action = int(rng.choice(np.flatnonzero(mask)))
        obs, _reward, terminated, truncated, _info = env.step(action)
        done = terminated or truncated
        step_count += 1
    assert done, f"Episode should complete within {max_steps} decisions (ran {step_count})"
    level = env.get_level()
    assert level.width_tiles == config.game.width_tiles
    assert level.height_tiles == config.game.height_tiles
    env.close()


def test_env_target_override() -> None:
    """Verify composer env intent target override."""
    config = load_config(training="initial")
    context = build_env_context(config)
    env = make_env(config, context, seed=1)
    target = np.zeros(config.target_vector_dim, dtype=np.float32)
    obs, _info = env.reset(seed=1, options={"target": target})
    assert obs.shape == env.observation_space.shape
    assert np.allclose(env._intent_target, target)
    env.close()
