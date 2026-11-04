from __future__ import annotations

import pytest
from pcgrl.bridge.bridge import GameplayBridge


def test_level_to_engine_string_shape(config, catalog, blank_level) -> None:
    """Verify level to engine string shape."""
    bridge = GameplayBridge(config, jar_path="/nonexistent/mario-bridge.jar")
    engine_str = bridge.level_to_engine_string(blank_level)
    lines = engine_str.splitlines()
    assert len(lines) == config.game.height_tiles
    assert all(len(line) == config.game.width_tiles for line in lines)


def test_level_to_engine_string_uses_dash_for_empty(config, catalog, blank_level) -> None:
    """Verify level to engine string uses dash for empty."""
    bridge = GameplayBridge(config, jar_path="/nonexistent/mario-bridge.jar")
    engine_str = bridge.level_to_engine_string(blank_level)
    for char in engine_str.replace("\n", ""):
        assert char == "-"


def test_fallback_result_is_lose(config) -> None:
    """Verify fallback result is lose."""
    result = GameplayBridge._fallback_result()
    assert not result.won
    assert result.game_status == "LOSE"


def test_fallback_result_zeros(config) -> None:
    """Verify fallback result zeros."""
    result = GameplayBridge._fallback_result()
    assert result.kills_total == 0
    assert result.num_jumps == 0
    assert result.completion_ratio == 0.0


def test_aggregate_runs_majority_wins(config) -> None:
    """Verify aggregate runs majority wins."""
    from pcgrl.bridge.bridge import GameplayBridge
    from pcgrl.core.domain.contract import GameplayResult

    def make(won: bool) -> GameplayResult:
        """Make."""
        return GameplayBridge._fallback_result().model_copy(
            update={"won": won, "game_status": "WIN" if won else "LOSE"}
        )

    runs = [make(True), make(True), make(False)]
    result = GameplayBridge._aggregate_runs(runs)
    assert result.won


def test_aggregate_runs_majority_loses(config) -> None:
    """Verify aggregate runs majority loses."""
    from pcgrl.bridge.bridge import GameplayBridge

    def make(won: bool):
        """Make."""
        return GameplayBridge._fallback_result().model_copy(
            update={"won": won, "game_status": "WIN" if won else "LOSE"}
        )

    runs = [make(False), make(False), make(True)]
    result = GameplayBridge._aggregate_runs(runs)
    assert not result.won


def test_evaluate_fallback_when_no_jar(config, blank_level) -> None:
    """Verify evaluate fallback when no jar."""
    relaxed = config.model_copy(
        update={
            "generation": config.generation.model_copy(update={"require_gameplay_bridge": False}),
            "evaluation": config.evaluation.model_copy(update={"require_gameplay_bridge": False}),
        }
    )
    bridge = GameplayBridge(relaxed, jar_path="/nonexistent/mario-bridge.jar")
    result = bridge.evaluate(
        blank_level, agent="robinBaumgarten", timer_seconds=5, mario_state=0, num_runs=1
    )
    assert not result.won


def test_evaluate_cache_hit(config, blank_level, tmp_path) -> None:
    """Verify evaluate cache hit."""
    from unittest.mock import patch

    from pcgrl.bridge.cache import GameplayCache, make_cache_key

    tmp_cache = tmp_path / "gameplay.jsonl"
    with patch("pcgrl.bridge.bridge.GameplayCache", lambda: GameplayCache(str(tmp_cache))):
        relaxed = config.model_copy(
            update={
                "generation": config.generation.model_copy(
                    update={"require_gameplay_bridge": False}
                ),
                "evaluation": config.evaluation.model_copy(
                    update={"require_gameplay_bridge": False}
                ),
            }
        )
        bridge = GameplayBridge(relaxed, jar_path="/nonexistent/mario-bridge.jar")
    if bridge._cache is None:
        pytest.skip("Cache disabled in config")
    engine_str = bridge.level_to_engine_string(blank_level)
    key = make_cache_key(engine_str, "robinBaumgarten", 30, 0)
    fallback = GameplayBridge._fallback_result()
    bridge._cache.put(key, fallback)
    result = bridge.evaluate(blank_level, agent="robinBaumgarten", timer_seconds=30, mario_state=0)
    assert not result.won


def test_run_once_with_mocked_subprocess(config, blank_level) -> None:
    """Verify run once with mocked subprocess."""
    import json
    from unittest.mock import MagicMock, patch

    bridge = GameplayBridge(config, jar_path="/nonexistent/mario-bridge.jar")
    payload = {
        "level": "test",
        "agent": "test",
        "timer_seconds": 30,
        "mario_state": 0,
        "visuals": False,
        "fps": 0,
        "scale": 2.0,
        "frames_output_dir": None,
    }
    response = {
        "ok": True,
        "game_status": "WIN",
        "completion_percentage": 1.0,
        "remaining_time": 100,
        "mario_mode": 0,
        "num_jumps": 3,
        "max_x_jump": 10.0,
        "max_jump_air_time": 5,
        "kills_total": 0,
        "kills_by_fire": 0,
        "kills_by_stomp": 0,
        "kills_by_shell": 0,
        "kills_by_fall": 0,
        "current_lives": 3,
        "current_coins": 5,
        "num_collected_mushrooms": 0,
        "num_collected_fireflower": 0,
        "num_collected_tile_coins": 0,
        "num_destroyed_bricks": 0,
        "num_bump_question_block": 0,
        "num_bump_brick": 0,
        "num_hurts": 0,
    }
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = json.dumps(response) + "\n"
    with patch("pcgrl.bridge.bridge.subprocess.run", return_value=mock_proc):
        result = bridge._run_once(payload)
    assert result.won


def test_run_once_subprocess_failure(config) -> None:
    """Verify run once subprocess failure."""
    from unittest.mock import MagicMock, patch

    bridge = GameplayBridge(config, jar_path="/nonexistent/mario-bridge.jar")
    mock_proc = MagicMock()
    mock_proc.returncode = 1
    with patch("pcgrl.bridge.bridge.subprocess.run", return_value=mock_proc):
        result = bridge._run_once({})
    assert not result.won


def test_evaluate_with_jar_mocked(config, blank_level) -> None:
    """Verify evaluate with jar mocked."""
    import json
    from unittest.mock import MagicMock, patch

    bridge = GameplayBridge(config, jar_path="/nonexistent/mario-bridge.jar")
    response = {
        "ok": True,
        "game_status": "WIN",
        "completion_percentage": 1.0,
        "remaining_time": 100,
        "mario_mode": 0,
        "num_jumps": 1,
        "max_x_jump": 5.0,
        "max_jump_air_time": 2,
        "kills_total": 0,
        "current_lives": 3,
        "current_coins": 0,
        "num_collected_mushrooms": 0,
        "num_collected_fireflower": 0,
        "num_collected_tile_coins": 0,
        "num_destroyed_bricks": 0,
        "num_bump_question_block": 0,
        "num_bump_brick": 0,
        "num_hurts": 0,
    }
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = json.dumps(response) + "\n"
    mock_jar = MagicMock()
    mock_jar.is_file.return_value = True
    bridge._jar = mock_jar
    bridge._cache = None
    with patch("pcgrl.bridge.bridge.subprocess.run", return_value=mock_proc):
        result = bridge.evaluate(blank_level, num_runs=1)
    assert result.won
