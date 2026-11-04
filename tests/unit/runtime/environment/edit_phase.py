from __future__ import annotations

import numpy as np
import pytest
from pcgrl.core.helpers.path import resolve_path
from pcgrl.runtime.environment.edit_actions import (
    N_SEGMENTS,
    EditAction,
    decode_edit_action,
    edit_action_count,
    stop_action_index,
)


def test_codec_roundtrip_covers_every_index() -> None:
    """Every edit index decodes to a unique (op, segment, param) triple."""
    seen: set[tuple[str, int, int]] = set()
    for index in range(stop_action_index()):
        action = decode_edit_action(index)
        assert isinstance(action, EditAction)
        assert 0 <= action.segment < N_SEGMENTS
        seen.add((action.op, action.segment, action.param))
    assert len(seen) == stop_action_index()
    assert decode_edit_action(stop_action_index()) is None
    with pytest.raises(IndexError):
        decode_edit_action(edit_action_count())


@pytest.fixture
def env(config, catalog):
    """Two-phase env over the real library (skipped when absent)."""
    library_root = resolve_path(config.library.output_root)
    if not (library_root / "chunk_library.npz").is_file():
        pytest.skip("Chunk library not built")
    from pcgrl.learning.env_factory import build_env_context, make_env

    context = build_env_context(config)
    return make_env(config, context, seed=5)


def test_observation_shape_and_phase_features(env, config) -> None:
    """Obs is 39-d; phase flag flips when the edit phase starts."""
    obs, _info = env.reset(seed=5)
    assert obs.shape == (39,)
    assert obs[-2] == 0.0
    rng = np.random.default_rng(5)
    while not env.in_edit_phase:
        legal = np.flatnonzero(env.action_masks())
        obs, _, done, trunc, _ = env.step(int(rng.choice(legal)))
        assert not (done or trunc), "episode ended before the edit phase"
    assert obs[-2] == 1.0
    assert obs[-1] == pytest.approx(1.0)


def test_phase_masks_are_disjoint(env) -> None:
    """Compose phase masks all edits; edit phase masks all chunks."""
    env.reset(seed=6)
    num_chunks = len(env._core.library)
    mask = env.action_masks()
    assert not mask[num_chunks:].any()
    rng = np.random.default_rng(6)
    while not env.in_edit_phase:
        legal = np.flatnonzero(env.action_masks())
        env.step(int(rng.choice(legal)))
    mask = env.action_masks()
    assert not mask[:num_chunks].any()
    assert mask[num_chunks + stop_action_index()]


def test_stop_action_terminates_with_terminal_reward(env) -> None:
    """STOP ends the episode and pays the terminal reward exactly once."""
    env.reset(seed=7, options={"target": np.full(8, 0.5, dtype=np.float32)})
    rng = np.random.default_rng(7)
    while not env.in_edit_phase:
        legal = np.flatnonzero(env.action_masks())
        _, reward, done, trunc, _ = env.step(int(rng.choice(legal)))
        assert reward == 0.0 and not (done or trunc)
    num_chunks = len(env._core.library)
    _, reward, done, _, info = env.step(num_chunks + stop_action_index())
    assert done
    assert reward != 0.0
    assert "reward_components" in info
    assert "edit_log" in info


def test_edit_budget_exhaustion_terminates(env, config) -> None:
    """Spending the whole budget ends the episode without STOP."""
    env.reset(seed=8)
    rng = np.random.default_rng(8)
    while not env.in_edit_phase:
        legal = np.flatnonzero(env.action_masks())
        env.step(int(rng.choice(legal)))
    num_chunks = len(env._core.library)
    budget = config.environment.composer.edit_budget
    done = False
    steps = 0
    while not done and steps <= budget:
        mask = env.action_masks()
        legal = np.flatnonzero(mask[num_chunks:-1])
        action = num_chunks + int(rng.choice(legal)) if legal.size else (
            num_chunks + stop_action_index()
        )
        _, _, done, _, _ = env.step(action)
        steps += 1
    assert done
    assert steps <= budget


def test_edit_budget_zero_is_compose_only(config, catalog) -> None:
    """edit_budget=0 reproduces the single-phase episode (ablation cell)."""
    library_root = resolve_path(config.library.output_root)
    if not (library_root / "chunk_library.npz").is_file():
        pytest.skip("Chunk library not built")
    from pcgrl.learning.env_factory import build_env_context, make_env

    composer = config.environment.composer.model_copy(update={"edit_budget": 0})
    environment = config.environment.model_copy(update={"composer": composer})
    compose_only = config.model_copy(update={"environment": environment})
    context = build_env_context(compose_only)
    env = make_env(compose_only, context, seed=9)
    env.reset(seed=9)
    rng = np.random.default_rng(9)
    done = trunc = False
    reward = 0.0
    while not (done or trunc):
        legal = np.flatnonzero(env.action_masks())
        _, reward, done, trunc, _ = env.step(int(rng.choice(legal)))
        assert not env.in_edit_phase
    assert reward != 0.0
