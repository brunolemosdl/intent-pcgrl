from __future__ import annotations

import pytest
from pcgrl.bridge.result import parse_bridge_response


def _win_payload(**overrides) -> dict:
    """win payload."""
    base = {
        "ok": True,
        "game_status": "WIN",
        "completion_percentage": 1.0,
        "remaining_time": 5000,
        "mario_mode": 0,
        "num_jumps": 12,
        "max_x_jump": 3.5,
        "max_jump_air_time": 8,
        "kills_total": 2,
        "kills_by_stomp": 2,
        "kills_by_fire": 0,
        "kills_by_shell": 0,
        "kills_by_fall": 0,
        "current_lives": 2,
        "current_coins": 5,
        "num_collected_mushrooms": 1,
        "num_collected_fireflower": 0,
        "num_collected_tile_coins": 3,
        "num_destroyed_bricks": 1,
        "num_bump_question_block": 2,
        "num_bump_brick": 1,
        "num_hurts": 0,
    }
    base.update(overrides)
    return base


def test_parse_win_result() -> None:
    """Verify parse win result."""
    result = parse_bridge_response(_win_payload())
    assert result.won
    assert result.game_status == "WIN"
    assert result.completion_ratio == 1.0


def test_parse_lose_result() -> None:
    """Verify parse lose result."""
    payload = _win_payload(ok=True, game_status="LOSE", completion_percentage=0.4)
    result = parse_bridge_response(payload)
    assert not result.won
    assert result.game_status == "LOSE"
    assert result.completion_ratio == pytest.approx(0.4)


def test_parse_timeout_result() -> None:
    """Verify parse timeout result."""
    payload = _win_payload(ok=True, game_status="TIME_OUT", completion_percentage=0.8)
    result = parse_bridge_response(payload)
    assert not result.won
    assert result.game_status == "TIME_OUT"


def test_failed_response_raises() -> None:
    """Verify failed response default error."""
    with pytest.raises(RuntimeError, match="bridge"):
        parse_bridge_response({"ok": False, "error": "bridge failed"})


def test_failed_response_default_error() -> None:
    """Verify failed response default error."""
    with pytest.raises(RuntimeError):
        parse_bridge_response({"ok": False})


def test_kills_by_type_parsed() -> None:
    """Verify kills by type parsed."""
    result = parse_bridge_response(_win_payload(kills_by_stomp=3, kills_by_fire=1))
    assert result.kills_by_type["stomp"] == 3
    assert result.kills_by_type["fire"] == 1


def test_numeric_fields_parsed() -> None:
    """Verify numeric fields parsed."""
    result = parse_bridge_response(_win_payload(num_jumps=15, current_coins=7))
    assert result.num_jumps == 15
    assert result.current_coins == 7


def test_float_fields_parsed() -> None:
    """Verify float fields parsed."""
    result = parse_bridge_response(_win_payload(max_x_jump=4.5))
    assert result.max_x_jump == pytest.approx(4.5)


def test_missing_fields_default() -> None:
    """Verify missing fields default."""
    result = parse_bridge_response({"ok": True, "game_status": "LOSE"})
    assert result.num_jumps == 0
    assert result.current_lives == 0
