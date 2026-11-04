from __future__ import annotations

import numpy as np
import pytest
from pcgrl.core.helpers.path import resolve_path
from pcgrl.core.settings.loader import load_config


@pytest.fixture(scope="module")
def composer_setup(tmp_path_factory):
    """Composer setup."""
    config = load_config(training="initial", apply_profile=False)
    corpus_root = resolve_path(config.dataset.corpus_root)
    if not corpus_root.is_dir() or not any(corpus_root.rglob("*.txt")):
        pytest.skip("Corpus files not present")
    tmp = tmp_path_factory.mktemp("composer")
    override = tmp / "override.yaml"
    override.write_text(
        f"library:\n  output_root: {tmp}/library\n"
        f"dataset:\n  processed_root: {tmp}/processed\n"
    )
    config = load_config(training="initial", override_path=override, apply_profile=False)
    from pcgrl.app.data.corpus.patterns import run_patterns
    from pcgrl.app.pipeline.library import run_library
    from pcgrl.core.tiles.catalog import load_catalog

    run_library(config)
    (tmp / "processed").mkdir(exist_ok=True)
    run_patterns(config, load_catalog(config), tmp / "processed")
    return config


def test_make_env_selects_composer(composer_setup) -> None:
    """Verify make env selects composer."""
    from pcgrl.learning.env_factory import build_env_context, make_env
    from pcgrl.runtime.environment.composer import MarioComposerEnv

    context = build_env_context(composer_setup)
    assert context.chunk_library is not None
    assert context.pattern_reference is not None
    assert context.composer_terminal is not None
    env = make_env(composer_setup, context, seed=11)
    assert isinstance(env, MarioComposerEnv)


def test_composer_episode_end_to_end(composer_setup) -> None:
    """Verify composer episode end to end."""
    from pcgrl.learning.env_factory import build_env_context, make_env

    context = build_env_context(composer_setup)
    env = make_env(composer_setup, context, seed=5)
    target = np.full(8, 0.5, dtype=np.float32)
    obs, info = env.reset(seed=5, options={"target": target})
    assert obs.shape == env.observation_space.shape
    assert info["action_mask"].shape == (env.action_space.n,)
    rng = np.random.default_rng(5)
    total_reward = 0.0
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated):
        mask = env.action_masks()
        assert mask.any()
        action = int(rng.choice(np.flatnonzero(mask)))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        assert steps <= 25
    assert terminated
    assert "reward_components" in info
    components = info["reward_components"]
    assert set(components) == {"alignment", "style", "playability", "dup_penalty"}
    assert total_reward > 0.0
    level = env.get_level()
    assert level.width_tiles == composer_setup.game.width_tiles
    # Episodes are short by construction (the whole point of the redesign).
    assert 5 <= steps <= 20


def test_composer_set_intent_roundtrip(composer_setup) -> None:
    """Verify composer set intent roundtrip."""
    from pcgrl.learning.env_factory import build_env_context, make_env

    context = build_env_context(composer_setup)
    env = make_env(composer_setup, context, seed=3)
    env.reset(seed=3)
    custom = np.linspace(0.1, 0.9, 8).astype(np.float32)
    env.set_intent(custom)
    assert np.allclose(env._intent_target, custom)
    # reset without options resamples a full-range target
    env.reset()
    assert env._intent_target.shape == (8,)
