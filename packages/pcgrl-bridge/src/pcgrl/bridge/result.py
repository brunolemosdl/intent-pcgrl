from __future__ import annotations

from typing import Literal, cast

from pcgrl.core.domain.contract import GameplayResult


def _as_int(value: object, default: int = 0) -> int:
    """Coerce a bridge payload field to ``int``, falling back to ``default``."""
    return int(value) if isinstance(value, (int, float, str)) else default


def _as_float(value: object, default: float = 0.0) -> float:
    """Coerce a bridge payload field to ``float``, falling back to ``default``."""
    return float(value) if isinstance(value, (int, float, str)) else default


def parse_bridge_response(payload: dict[str, object]) -> GameplayResult:
    """Parse a JSON bridge response into a ``GameplayResult``.

    Args:
        payload: Decoded JSON object returned by the Mario bridge JAR.

    Returns:
        Normalized gameplay statistics for reward and evaluation code.

    Raises:
        RuntimeError: When the bridge reports ``ok: false`` or an error message.
    """
    if not bool(payload.get("ok", False)):
        error = payload.get("error", "Gameplay bridge failed")
        raise RuntimeError(str(error))
    kills = {
        "stomp": _as_int(payload.get("kills_by_stomp", 0)),
        "fire": _as_int(payload.get("kills_by_fire", 0)),
        "shell": _as_int(payload.get("kills_by_shell", 0)),
        "fall": _as_int(payload.get("kills_by_fall", 0)),
    }
    status = cast(
        Literal["WIN", "LOSE", "TIME_OUT", "RUNNING"], str(payload.get("game_status", "LOSE"))
    )
    completion = _as_float(payload.get("completion_percentage", 0.0))
    won = status == "WIN"
    return GameplayResult(
        won=won,
        completion_ratio=completion,
        remaining_time_ms=_as_int(payload.get("remaining_time", 0)),
        mario_mode=_as_int(payload.get("mario_mode", 0)),
        num_jumps=_as_int(payload.get("num_jumps", 0)),
        max_x_jump=_as_float(payload.get("max_x_jump", 0.0)),
        max_jump_air_time=_as_int(payload.get("max_jump_air_time", 0)),
        kills_total=_as_int(payload.get("kills_total", 0)),
        kills_by_type=kills,
        current_lives=_as_int(payload.get("current_lives", 0)),
        current_coins=_as_int(payload.get("current_coins", 0)),
        num_collected_mushrooms=_as_int(payload.get("num_collected_mushrooms", 0)),
        num_collected_fireflower=_as_int(payload.get("num_collected_fireflower", 0)),
        num_collected_tile_coins=_as_int(payload.get("num_collected_tile_coins", 0)),
        num_destroyed_bricks=_as_int(payload.get("num_destroyed_bricks", 0)),
        num_bump_question_block=_as_int(payload.get("num_bump_question_block", 0)),
        num_bump_brick=_as_int(payload.get("num_bump_brick", 0)),
        num_hurts=_as_int(payload.get("num_hurts", 0)),
        game_status=status,
        bridge_runs=1,
        bridge_win_rate=1.0 if won else 0.0,
        bridge_completion_std=0.0,
    )
