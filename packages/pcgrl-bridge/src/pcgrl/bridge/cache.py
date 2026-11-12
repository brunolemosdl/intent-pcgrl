from __future__ import annotations

import fcntl
import hashlib
import json
from collections import OrderedDict
from pathlib import Path

from pcgrl.core.domain.contract import GameplayResult
from pcgrl.core.helpers.path import ensure_dir, resolve_path


class GameplayCache:
    """LRU in-memory cache backed by an append-only JSONL file.

    Description:
        Avoids repeated bridge invocations for identical level/agent/timer
        combinations. Entries are persisted under file lock for multiprocess
        training runs.

    Attributes:
        max_size: Maximum number of entries kept in memory.
    """

    def __init__(
        self, cache_path: str = "artifacts/cache/gameplay.jsonl", max_size: int = 4096
    ) -> None:
        """Open or create the JSONL cache and hydrate memory from disk.

        Args:
            cache_path: Relative path to the append-only cache file.
            max_size: In-memory LRU capacity before evicting oldest keys.
        """
        self._path = ensure_dir(resolve_path(cache_path).parent) / Path(cache_path).name
        self._max_size = max_size
        self._memory: OrderedDict[str, GameplayResult] = OrderedDict()
        self._load()

    def get(self, key: str) -> GameplayResult | None:
        """Look up a cached gameplay result.

        Args:
            key: Cache key produced by ``make_cache_key``.

        Returns:
            Cached result, or ``None`` when the key is absent.
        """
        if key in self._memory:
            self._memory.move_to_end(key)
            return self._memory[key]
        return None

    def put(self, key: str, result: GameplayResult) -> None:
        """Store a gameplay result in memory and append it to the JSONL file.

        Args:
            key: Cache key produced by ``make_cache_key``.
            result: Gameplay outcome to persist.
        """
        self._memory[key] = result
        self._memory.move_to_end(key)
        while len(self._memory) > self._max_size:
            self._memory.popitem(last=False)
        with self._path.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.write(json.dumps({"key": key, "result": result.model_dump()}) + "\n")
            fcntl.flock(handle, fcntl.LOCK_UN)

    def _load(self) -> None:
        """Hydrate the in-memory LRU from the on-disk JSONL file."""
        if not self._path.is_file():
            return
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                self._memory[payload["key"]] = GameplayResult.model_validate(payload["result"])


def make_cache_key(level_string: str, agent: str, timer_seconds: int, mario_state: int) -> str:
    """Build a stable cache key for a bridge evaluation request.

    Args:
        level_string: Engine wire-format level text.
        agent: Bridge agent identifier.
        timer_seconds: Per-run timer limit.
        mario_state: Initial Mario power-up state code.

    Returns:
        Hex digest used as the cache lookup key.
    """
    digest = hashlib.blake2b(digest_size=16)
    digest.update(level_string.encode())
    digest.update(agent.encode())
    digest.update(str(timer_seconds).encode())
    digest.update(str(mario_state).encode())
    return digest.hexdigest()
