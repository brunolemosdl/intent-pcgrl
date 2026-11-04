from __future__ import annotations

import tempfile

from pcgrl.bridge.cache import GameplayCache, make_cache_key
from pcgrl.core.domain.contract import GameplayResult


def _make_result(won: bool = True) -> GameplayResult:
    """ make result."""
    return GameplayResult(
        won=won,
        completion_ratio=1.0 if won else 0.0,
        remaining_time_ms=0,
        mario_mode=0,
        num_jumps=5,
        max_x_jump=0.0,
        max_jump_air_time=0,
        kills_total=0,
        kills_by_type={},
        current_lives=3,
        current_coins=0,
        num_collected_mushrooms=0,
        num_collected_fireflower=0,
        num_collected_tile_coins=0,
        num_destroyed_bricks=0,
        num_bump_question_block=0,
        num_bump_brick=0,
        num_hurts=0,
        game_status="WIN" if won else "LOSE",
    )


def test_cache_miss_returns_none() -> None:
    """Verify cache miss returns none."""
    with tempfile.TemporaryDirectory() as tmp:
        cache = GameplayCache(cache_path=f"{tmp}/gameplay.jsonl")
        assert cache.get("nonexistent") is None


def test_cache_put_and_get() -> None:
    """Verify cache put and get."""
    with tempfile.TemporaryDirectory() as tmp:
        cache = GameplayCache(cache_path=f"{tmp}/gameplay.jsonl")
        result = _make_result(won=True)
        cache.put("key1", result)
        retrieved = cache.get("key1")
        assert retrieved is not None
        assert retrieved.won


def test_cache_persists_to_disk() -> None:
    """Verify cache persists to disk."""
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/gameplay.jsonl"
        cache = GameplayCache(cache_path=path)
        cache.put("key_disk", _make_result())
        cache2 = GameplayCache(cache_path=path)
        result = cache2.get("key_disk")
        assert result is not None
        assert result.won


def test_cache_lru_eviction() -> None:
    """Verify cache lru eviction."""
    with tempfile.TemporaryDirectory() as tmp:
        cache = GameplayCache(cache_path=f"{tmp}/gameplay.jsonl", max_size=2)
        cache.put("k1", _make_result())
        cache.put("k2", _make_result())
        cache.put("k3", _make_result())
        assert cache.get("k1") is None
        assert cache.get("k3") is not None


def test_make_cache_key_deterministic() -> None:
    """Verify make cache key different inputs."""
    key1 = make_cache_key("level", "agent", 30, 0)
    key2 = make_cache_key("level", "agent", 30, 0)
    assert key1 == key2


def test_make_cache_key_different_inputs() -> None:
    """Verify make cache key agent matters."""
    key1 = make_cache_key("level1", "agent", 30, 0)
    key2 = make_cache_key("level2", "agent", 30, 0)
    assert key1 != key2


def test_make_cache_key_agent_matters() -> None:
    """Verify make cache key agent matters."""
    key1 = make_cache_key("level", "agentA", 30, 0)
    key2 = make_cache_key("level", "agentB", 30, 0)
    assert key1 != key2


def test_make_cache_key_timer_matters() -> None:
    """Verify make cache key timer matters."""
    key1 = make_cache_key("level", "agent", 30, 0)
    key2 = make_cache_key("level", "agent", 60, 0)
    assert key1 != key2


def test_make_cache_key_returns_string() -> None:
    """Verify make cache key returns string."""
    key = make_cache_key("level", "agent", 30, 0)
    assert isinstance(key, str)
    assert len(key) > 0
