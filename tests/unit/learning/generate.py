from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pcgrl.core.domain.level import AreaType, LevelGrid

_COMPONENTS_OK = {
    "alignment": 0.9,
    "style": 0.8,
    "playability": 1.0,
    "dup_penalty": 0.0,
}


def _make_mock_level(config, catalog) -> LevelGrid:
    """A small but structurally valid, playable level: continuous ground to the
    flag plus a little reachable content so the intent metrics are non-trivial."""
    h, w = (config.game.height_tiles, config.game.width_tiles)
    empty_id = catalog.id_for_name("empty")
    solid_id = catalog.id_for_name("solid")
    tiles = np.full((h, w), empty_id, dtype=np.int16)
    tiles[h - 1, :] = solid_id
    tiles[h - 2, 1] = catalog.id_for_name("player")
    tiles[h - 2, w - 2] = catalog.id_for_name("flagpole")
    tiles[h - 4, 10:13] = solid_id
    tiles[h - 2, 20] = catalog.id_for_name("goomba")
    tiles[h - 3, 30] = catalog.id_for_name("coin")
    tiles[h - 3, 40:43] = catalog.id_for_name("brick")
    return LevelGrid(
        tiles=tiles,
        width_tiles=w,
        height_tiles=h,
        tile_size_px=config.game.tile_size_px,
        area_type=AreaType.OVERWORLD,
    )


class _StubComposerEnv:
    """Composer-shaped env whose episodes terminate after one step with a
    scripted terminal transition, cycling through ``episodes`` on each reset."""

    def __init__(self, level: LevelGrid, episodes: list[dict]) -> None:
        """Bind the level every episode stitches and the scripted transitions."""
        self._level = level
        self._episodes = episodes
        self._index = -1

    def reset(self, *, seed=None, options=None):
        """Advance to the next scripted episode."""
        self._index = (self._index + 1) % len(self._episodes)
        return np.zeros(4, dtype=np.float32), {"action_mask": self.action_masks()}

    def action_masks(self) -> np.ndarray:
        """All actions legal."""
        return np.ones(4, dtype=bool)

    def step(self, action: int):
        """Return the scripted terminal transition for the current episode."""
        episode = self._episodes[self._index]
        info = dict(episode.get("info", {}))
        return (
            np.zeros(4, dtype=np.float32),
            float(episode.get("reward", 0.0)),
            not episode.get("truncated", False),
            bool(episode.get("truncated", False)),
            info,
        )

    def get_level(self) -> LevelGrid:
        """Return the fixed stitched level."""
        return self._level


def _stub_context(catalog, terminal=None):
    """Minimal EnvContext stand-in for the composer generate path."""
    if terminal is None:
        terminal = MagicMock()
        terminal.compute.return_value = (1.0, dict(_COMPONENTS_OK))
    return SimpleNamespace(catalog=catalog, composer_terminal=terminal)


def _stub_bridge_cls(won: bool = True, completion: float = 1.0) -> MagicMock:
    """GameplayBridge class mock whose instance reports one A* outcome."""
    bridge = MagicMock()
    bridge.jar_path = MagicMock()
    bridge.jar_path.is_file.return_value = True
    bridge.evaluate.return_value = SimpleNamespace(
        won=won,
        completion_ratio=completion,
        bridge_win_rate=1.0 if won else 0.0,
        bridge_runs=3,
        bridge_completion_std=0.0,
        game_status="WIN" if won else "LOSE",
    )
    bridge_cls = MagicMock(return_value=bridge)
    return bridge_cls


def _episode(reward: float, *, playability: float = 1.0, truncated: bool = False) -> dict:
    """Scripted terminal transition with composer info fields."""
    components = dict(_COMPONENTS_OK, playability=playability)
    info = {"reward_components": components, "chunk_ids": [1, 2, 3]}
    if truncated:
        info = {}
    return {"reward": reward, "truncated": truncated, "info": info}


def _run(
    config,
    catalog,
    tmp_path,
    episodes,
    *,
    bridge_cls=None,
    terminal=None,
    target=None,
    polish=False,
):
    """Invoke run_generate against the stubbed env/context/bridge/model.

    Contract tests pin ``polish=False`` so they keep asserting the raw
    best-of-N pipeline regardless of the product config default."""
    from pcgrl.learning.generate import run_generate

    fake_ckpt = tmp_path / "model.zip"
    fake_ckpt.write_bytes(b"fake")
    env = _StubComposerEnv(_make_mock_level(config, catalog), episodes)
    model = MagicMock()
    model.predict.return_value = (0, None)
    ppo_cls = MagicMock()
    ppo_cls.load.return_value = model
    with (
        patch("pcgrl.learning.generate.MaskablePPO", ppo_cls),
        patch("pcgrl.learning.generate.build_env_context", return_value=_stub_context(catalog, terminal)),
        patch("pcgrl.learning.generate.make_env", return_value=env),
        patch("pcgrl.learning.generate.GameplayBridge", bridge_cls or _stub_bridge_cls()),
    ):
        return run_generate(
            config, fake_ckpt, output_dir=tmp_path, target=target, polish=polish
        )


def test_run_generate_success_report_contract(config, catalog, tmp_path) -> None:
    """A playable candidate yields the frozen diagnostics contract."""
    out = _run(config, catalog, tmp_path, [_episode(2.0)])
    assert out.suffix == ".json"
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "SUCCESS_PLAYABLE"
    assert payload["success"] is True
    assert payload["ascii"]
    diagnostics = payload["diagnostics"]
    for key in (
        "terminal_reward",
        "reward_components",
        "reward_recomputed",
        "truncated_episode",
        "chunk_ids",
        "num_decisions",
        "structural_valid",
        "heuristic_reachable",
        "measured_target",
        "per_dimension_error",
        "mean_target_error",
        "max_target_error",
        "intent_dimensions",
        "bridge_won",
        "bridge_completion_ratio",
        "all_attempts",
    ):
        assert key in diagnostics, key
    assert diagnostics["intent_dimensions"] == list(config.target_vector_field_names())
    assert len(diagnostics["per_dimension_error"]) == config.target_vector_dim
    assert len(diagnostics["all_attempts"]) == config.generation.max_attempts
    assert diagnostics["chunk_ids"] == [1, 2, 3]
    assert diagnostics["bridge_won"] is True


def test_run_generate_selects_best_terminal_reward(config, catalog, tmp_path) -> None:
    """Candidate ranking follows the composer terminal reward."""
    rewards = [1.0, 5.0, 2.0, 0.5, 3.0]
    out = _run(config, catalog, tmp_path, [_episode(r) for r in rewards])
    diagnostics = json.loads(out.read_text(encoding="utf-8"))["diagnostics"]
    assert diagnostics["selected_attempt"] == 1
    assert diagnostics["terminal_reward"] == pytest.approx(5.0)


def test_run_generate_recomputes_reward_on_truncation(config, catalog, tmp_path) -> None:
    """Truncated episodes re-score the stitched level instead of reading 0."""
    terminal = MagicMock()
    terminal.compute.return_value = (7.7, dict(_COMPONENTS_OK))
    out = _run(
        config,
        catalog,
        tmp_path,
        [_episode(0.0, truncated=True)],
        terminal=terminal,
    )
    diagnostics = json.loads(out.read_text(encoding="utf-8"))["diagnostics"]
    assert diagnostics["terminal_reward"] == pytest.approx(7.7)
    assert diagnostics["reward_recomputed"] is True
    assert diagnostics["truncated_episode"] is True
    assert terminal.compute.called


def test_run_generate_fails_when_flood_unreachable(config, catalog, tmp_path) -> None:
    """A playability term below 1.0 fails the flood gate for every candidate."""
    with pytest.raises(RuntimeError, match=r"FAIL_HEURISTIC_UNREACHABLE"):
        _run(config, catalog, tmp_path, [_episode(2.0, playability=0.4)])
    report = next(tmp_path.glob("level_*.json"))
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["success"] is False
    assert "FAIL_HEURISTIC_UNREACHABLE" in payload["diagnostics"]["failure_reasons"]
    assert payload["diagnostics"]["final_playable"] is False


def test_run_generate_fails_when_bridge_loses(config, catalog, tmp_path) -> None:
    """A bridge loss on every candidate resolves to FAIL_BRIDGE_NOT_WON."""
    with pytest.raises(RuntimeError, match=r"FAIL_BRIDGE_NOT_WON"):
        _run(
            config,
            catalog,
            tmp_path,
            [_episode(2.0)],
            bridge_cls=_stub_bridge_cls(won=False, completion=0.2),
        )


def test_run_generate_fails_when_bridge_is_unavailable(config, catalog, tmp_path) -> None:
    """A missing bridge JAR aborts before any rollout."""
    bridge_cls = _stub_bridge_cls()
    bridge_cls.return_value.ensure_available.side_effect = FileNotFoundError("no jar")
    with pytest.raises(RuntimeError, match="FAIL_BRIDGE_UNAVAILABLE"):
        _run(config, catalog, tmp_path, [_episode(2.0)], bridge_cls=bridge_cls)


