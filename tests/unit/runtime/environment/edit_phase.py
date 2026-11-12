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
        action = (
            num_chunks + int(rng.choice(legal))
            if legal.size
            else (num_chunks + stop_action_index())
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


def _enter_edit_phase(env, seed: int, target: np.ndarray | None = None) -> None:
    """Compose with random legal chunks until the edit phase starts."""
    options = {"target": target} if target is not None else None
    env.reset(seed=seed, options=options)
    rng = np.random.default_rng(seed)
    while not env.in_edit_phase:
        env.step(int(rng.choice(np.flatnonzero(env.action_masks()))))


def test_none_band_targets_are_read_as_zero(env) -> None:
    """A count target of 0.05, the bridge's "none", conditions the episode on zero."""
    target = np.array([0.05, 0.05, 0.5, 0.05, 0.05, 0.05, 0.95, 0.05], dtype=np.float32)
    obs, _ = env.reset(seed=3, options={"target": target})
    assert obs[:8].tolist() == pytest.approx([0.0, 0.0, 0.5, 0.0, 0.0, 0.05, 0.95, 0.05])


def test_satisfied_level_leaves_only_stop(env) -> None:
    """When the level already measures its target no edit is offered."""
    _enter_edit_phase(env, seed=10)
    env.set_intent(env._measurement().vector.copy())
    env._cached_edit_mask = None
    num_chunks = len(env._core.library)
    mask = env.action_masks()
    assert mask.sum() == 1
    assert mask[num_chunks + stop_action_index()]


def test_edits_are_offered_only_for_dimensions_that_need_them(env) -> None:
    """Only the enemy actions, and only in the needed direction, are unmasked."""
    _enter_edit_phase(env, seed=10)
    measured = env._measurement().vector.copy()
    target = measured.copy()
    target[0] = 0.0 if measured[0] > 0.2 else 1.0
    env.set_intent(target)
    env._cached_edit_mask = None
    num_chunks = len(env._core.library)
    offered = {
        decode_edit_action(int(index)).op
        for index in np.flatnonzero(env.action_masks()[num_chunks:-1])
    }
    assert offered == ({"remove_enemies"} if target[0] == 0.0 else {"add_enemies"})


def test_edit_never_overshoots_its_dimension(env) -> None:
    """Every accepted edit leaves its own dimension no farther from the target."""
    _enter_edit_phase(env, seed=12, target=np.full(8, 0.5, dtype=np.float32))
    rng = np.random.default_rng(12)
    num_chunks = len(env._core.library)
    keys = list(env._context.config.target_vector_field_names())
    done = False
    checked = 0
    while not done:
        legal = np.flatnonzero(env.action_masks()[num_chunks:-1])
        if not legal.size:
            break
        index = int(rng.choice(legal))
        action = decode_edit_action(index)
        dim = keys.index(action.dim)
        before = float(env._measurement().vector[dim])
        _, _, done, _, _ = env.step(num_chunks + index)
        after = float(env._measurement().vector[dim]) if env.in_edit_phase else before
        target = float(env._intent_target[dim])
        if action.dim in ("enemy_density", "coin_density", "interactive_block_density"):
            assert abs(after - target) <= abs(before - target) + 1e-6
            checked += 1
    assert checked > 0


def test_void_action_stays_masked_until_its_segment_changes(env, monkeypatch) -> None:
    """An edit that changed nothing is not offered again while nothing moved.

    The gate is made to refuse one edit; the action stays masked after the
    gate is back, and is released once an accepted edit touches its segment."""
    _enter_edit_phase(env, seed=14, target=np.full(8, 0.5, dtype=np.float32))
    num_chunks = len(env._core.library)
    legal = [int(index) for index in np.flatnonzero(env.action_masks()[num_chunks:-1])]
    by_segment: dict[int, list[int]] = {}
    for index in legal:
        by_segment.setdefault(decode_edit_action(index).segment, []).append(index)
    offered = next(ids for ids in by_segment.values() if len(ids) > 1)
    index = offered[0]
    with monkeypatch.context() as patched:
        patched.setattr(
            env._gate, "check", lambda tiles, baseline=frozenset(), min_reach=None: (False, "x")
        )
        env.step(num_chunks + index)
    assert env._edit_log[-1]["accepted"] is False
    assert index in env._void_edits
    assert not env.action_masks()[num_chunks + index]
    for other in offered[1:]:
        if not env.action_masks()[num_chunks + other]:
            continue
        env.step(num_chunks + other)
        if env._edit_log[-1]["accepted"]:
            assert index not in env._void_edits
            break


def test_edit_that_does_not_help_its_dimension_is_undone(env, monkeypatch) -> None:
    """An action whose tiles change without serving its dimension leaves no trace.

    The planner is made to answer an enemy removal with a stray coin: the
    gates accept it, the enemy count is no closer to its target, so the
    tiles are restored and the action is recorded as void."""
    from pcgrl.runtime.environment import composer
    from pcgrl.runtime.refine.ops import Edit

    _enter_edit_phase(env, seed=21, target=np.zeros(8, dtype=np.float32))
    num_chunks = len(env._core.library)
    offered = [
        int(index)
        for index in np.flatnonzero(env.action_masks()[num_chunks:-1])
        if decode_edit_action(int(index)).op == "remove_enemies"
    ]
    assert offered
    ctx = env._refine_ctx
    stray = Edit(op="remove_enemies", cells=((1, 60, ctx.coin_id),))
    monkeypatch.setattr(composer, "plan_edit_action", lambda *args, **kwargs: [stray])
    tiles_before = env._tiles.copy()
    env.step(num_chunks + offered[0])
    assert env._edit_log[-1]["accepted"] is False
    assert np.array_equal(env._tiles, tiles_before)
    assert offered[0] in env._void_edits


def _mask_of(mask: np.ndarray, op: str, segment: int, param: int = 0) -> bool:
    """Mask entry of one named edit action."""
    for index in range(stop_action_index()):
        action = decode_edit_action(index)
        if (action.op, action.segment, action.param) == (op, segment, param):
            return bool(mask[index])
    raise AssertionError(f"no such action: {op} {segment} {param}")


def _plain_floor(config, catalog):
    """One-row floor with the endpoints stamped, plus its refine context."""
    from pcgrl.runtime.refine.context import build_context

    ctx = build_context(config, catalog)
    height, width = config.game.height_tiles, config.game.width_tiles
    tiles = np.full((height, width), ctx.empty_id, dtype=np.int16)
    tiles[height - ctx.base_height :, :] = ctx.solid_id
    tiles[ctx.player] = catalog.id_for_name("player")
    tiles[ctx.flagpole] = catalog.id_for_name("flagpole")
    return tiles, ctx


def _edit_mask(tiles, ctx, config, catalog, **overrides: float) -> np.ndarray:
    """Edit mask for the level's own measurement with some targets overridden."""
    from pcgrl.runtime.environment.edit_actions import edit_action_masks
    from pcgrl.runtime.metrics.intent import INTENT_KEYS, measure_intent
    from pcgrl.runtime.refine.gates import grid_from_tiles

    state = measure_intent(grid_from_tiles(tiles, config), catalog, config)
    target = state.vector.astype(np.float32).copy()
    for key, value in overrides.items():
        target[INTENT_KEYS.index(key)] = value
    return edit_action_masks(tiles, ctx, target, state, tolerance=0.05)


def test_summit_raise_is_offered_only_where_the_summit_is(config, catalog) -> None:
    """Level zero of the hill action names the segment holding the summit."""
    tiles, ctx = _plain_floor(config, catalog)
    height = tiles.shape[0]
    for offset, stack in enumerate((2, 3, 4, 5, 6, 7, 8, 9, 9)):
        tiles[height - stack :, 100 + offset] = ctx.solid_id
    mask = _edit_mask(tiles, ctx, config, catalog, verticality=0.83)
    summit_segment = 100 // (ctx.width // N_SEGMENTS)
    offered = [segment for segment in range(N_SEGMENTS) if _mask_of(mask, "raise_hill", segment)]
    assert offered == [summit_segment]
    assert _mask_of(mask, "raise_hill", 2, 1)
    flat, flat_ctx = _plain_floor(config, catalog)
    flat_mask = _edit_mask(flat, flat_ctx, config, catalog, verticality=0.83)
    assert not any(_mask_of(flat_mask, "raise_hill", segment) for segment in range(N_SEGMENTS))
    assert _mask_of(flat_mask, "raise_hill", 2, 1)


def test_smooth_is_offered_for_bumps_and_liftable_blocks(config, catalog) -> None:
    """Smoothing is unmasked where it has something to level or to lift.

    A ground spike makes it available in its segment; so does a lone
    mountable block when platform usage can spare its column. A segment
    with neither stays masked."""
    tiles, ctx = _plain_floor(config, catalog)
    height = tiles.shape[0]
    per_segment = ctx.width // N_SEGMENTS
    row = height - ctx.base_height - 4
    for start in (20, 60, 80, 100):
        tiles[row, start : start + 8] = ctx.solid_id
    tiles[height - 3 :, 45] = ctx.solid_id
    tiles[row, 125] = ctx.block_ids[0]
    for column in range(134, 146, 2):
        tiles[height - 4 :, column] = ctx.solid_id
    mask = _edit_mask(tiles, ctx, config, catalog, path_smoothness=1.0)
    assert _mask_of(mask, "smooth", 45 // per_segment)
    assert _mask_of(mask, "smooth", 125 // per_segment)
    assert not _mask_of(mask, "smooth", 0)


def test_edit_phase_starts_from_repaired_pipes(env, catalog) -> None:
    """Pipes clipped by a chunk seam are removed before the first edit."""
    env.reset(seed=15)
    rng = np.random.default_rng(15)
    while env._state.num_decisions < 2:
        env.step(int(rng.choice(np.flatnonzero(env.action_masks()))))
    original = env._core.stitch_level

    def clipped(state, **kwargs):
        level = original(state, **kwargs)
        tiles = level.tiles.copy()
        column = 40
        floor = tiles.shape[0] - env._refine_ctx.base_height
        tiles[floor:, column] = catalog.id_for_name("solid")
        tiles[:floor, column] = catalog.id_for_name("empty")
        tiles[floor - 2, column] = catalog.id_for_name("pipe_top_left")
        tiles[floor - 1, column] = catalog.id_for_name("pipe_body_left")
        tiles[floor - 3, column] = catalog.id_for_name("goomba")
        return level.copy_with_tiles(tiles)

    env._core.stitch_level = clipped
    try:
        env._enter_edit_phase()
    finally:
        env._core.stitch_level = original
    tiles = env.get_level().tiles
    pipe_ids = [catalog.id_for_name(name) for name in ("pipe_top_left", "pipe_body_left")]
    assert not np.isin(tiles[:, 40], pipe_ids).any()
    assert tiles[tiles.shape[0] - env._refine_ctx.base_height - 3, 40] != catalog.id_for_name(
        "goomba"
    )
    assert "pipe.width" not in env._gate_baseline


def test_clipped_pipe_the_path_climbs_on_becomes_ground(env, catalog) -> None:
    """A half pipe that is the only step up a wall stays as plain ground."""
    env.reset(seed=15)
    rng = np.random.default_rng(15)
    while env._state.num_decisions < 2:
        env.step(int(rng.choice(np.flatnonzero(env.action_masks()))))
    original = env._core.stitch_level
    ctx = env._refine_ctx
    solid, empty = catalog.id_for_name("solid"), catalog.id_for_name("empty")

    def walled(state, **kwargs):
        level = original(state, **kwargs)
        tiles = np.full_like(level.tiles, empty)
        floor = tiles.shape[0] - ctx.base_height
        tiles[floor:, :] = solid
        tiles[floor - 8 : floor, 60:65] = solid
        tiles[floor - 4, 59] = catalog.id_for_name("pipe_top_left")
        tiles[floor - 3 : floor, 59] = catalog.id_for_name("pipe_body_left")
        tiles[ctx.player] = catalog.id_for_name("player")
        tiles[ctx.flagpole] = catalog.id_for_name("flagpole")
        return level.copy_with_tiles(tiles)

    env._core.stitch_level = walled
    try:
        stitched = walled(env._state).tiles.astype(np.int16)
        env._enter_edit_phase()
    finally:
        env._core.stitch_level = original
    tiles = env.get_level().tiles
    floor = tiles.shape[0] - ctx.base_height
    assert (tiles[floor - 4 : floor, 59] == solid).all()
    assert "pipe.width" not in env._gate_baseline
    assert env._reach == env._gate.reach(stitched)
