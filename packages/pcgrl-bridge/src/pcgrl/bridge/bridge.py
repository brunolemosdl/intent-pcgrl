from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np

from pcgrl.bridge.cache import GameplayCache, make_cache_key
from pcgrl.bridge.result import parse_bridge_response
from pcgrl.core.domain.contract import GameplayResult
from pcgrl.core.domain.level import LevelGrid
from pcgrl.core.helpers.path import resolve_path, vendor_path
from pcgrl.core.settings.schema import RuntimeConfig
from pcgrl.core.tiles.catalog import load_catalog
from pcgrl.core.tiles.engine import level_to_engine_string


class GameplayBridge:
    """Evaluate levels through the Java Mario gameplay bridge.

    Description:
        Converts ``LevelGrid`` instances to engine strings, invokes the
        subprocess bridge JAR, optionally caches results, and aggregates
        multi-run statistics into a ``GameplayResult``.

    Attributes:
        jar_path: Resolved path to ``mario-bridge.jar``.
    """

    def __init__(self, config: RuntimeConfig, jar_path: str | None = None) -> None:
        """Wire catalog, JAR path, and optional gameplay cache from config.

        Args:
            config: Runtime settings (catalog, cache toggle, bridge requirements).
            jar_path: Optional override for the bridge JAR location.
        """
        self._config = config
        self._catalog = load_catalog(config)
        self._jar = (
            vendor_path("mario", "mario-bridge.jar") if jar_path is None else resolve_path(jar_path)
        )
        self._cache = GameplayCache() if config.reward.gameplay.cache else None

    @property
    def jar_path(self) -> Path:
        """Return the resolved bridge JAR path."""
        return self._jar

    def ensure_available(self) -> None:
        """Verify the bridge JAR exists on disk.

        Raises:
            FileNotFoundError: When the JAR is missing (build with ``make bridge``).
        """
        if self._jar.is_file():
            return
        raise FileNotFoundError(
            f"Gameplay bridge jar not found at '{self._jar}'. Build it with `make bridge`."
        )

    def level_to_engine_string(self, level: LevelGrid) -> str:
        """Serialize a level grid to the Mario engine wire format.

        Args:
            level: Level to convert.

        Returns:
            Engine-compatible level string.
        """
        return level_to_engine_string(level, self._catalog)

    def evaluate(
        self,
        level: LevelGrid,
        agent: str = "robinBaumgarten",
        timer_seconds: int = 30,
        mario_state: int = 0,
        num_runs: int = 5,
    ) -> GameplayResult:
        """Run one or more bridge simulations and aggregate the outcomes.

        Args:
            level: Level grid to evaluate.
            agent: Bridge agent name (default ``robinBaumgarten``).
            timer_seconds: Per-run time limit.
            mario_state: Initial Mario power-up state code.
            num_runs: Number of stochastic runs to aggregate.

        Returns:
            Aggregated ``GameplayResult``; falls back to a loss stub when the
            JAR is unavailable and bridge is not required.

        Raises:
            FileNotFoundError: When bridge is required but the JAR is missing.
        """
        if (
            self._config.generation.require_gameplay_bridge
            or self._config.evaluation.require_gameplay_bridge
        ):
            self.ensure_available()
        engine_level = self.level_to_engine_string(level)
        cache_key = make_cache_key(engine_level, agent, timer_seconds, mario_state)
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached
        if not self._jar.is_file():
            return self._fallback_result()
        payload = {
            "level": engine_level,
            "agent": agent,
            "timer_seconds": timer_seconds,
            "mario_state": mario_state,
            "visuals": False,
            "fps": 0,
            "scale": 2.0,
            "frames_output_dir": None,
        }
        runs = [self._run_once(payload) for _ in range(max(1, num_runs))]
        aggregated = self._aggregate_runs(runs)
        if self._cache is not None:
            self._cache.put(cache_key, aggregated)
        return aggregated

    def _run_once(self, payload: dict[str, object]) -> GameplayResult:
        """Execute a single bridge subprocess invocation.

        Args:
            payload: JSON-serializable request sent to the JAR on stdin.

        Returns:
            Parsed gameplay result, or a loss stub when the process fails.
        """
        proc = subprocess.run(
            ["java", "-jar", str(self._jar)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            return self._fallback_result()
        line = proc.stdout.strip().splitlines()[-1]
        return parse_bridge_response(json.loads(line))

    @staticmethod
    def _aggregate_runs(runs: list[GameplayResult]) -> GameplayResult:
        """Merge multiple bridge runs into one majority-vote result."""
        wins = sum(1 for run in runs if run.won)
        won = wins >= math.ceil(len(runs) / 2)
        first = runs[0]
        completion_values = [float(r.completion_ratio) for r in runs]
        return GameplayResult(
            won=won,
            completion_ratio=float(sum(completion_values) / len(runs)),
            remaining_time_ms=int(sum(r.remaining_time_ms for r in runs) / len(runs)),
            mario_mode=first.mario_mode,
            num_jumps=int(sum(r.num_jumps for r in runs) / len(runs)),
            max_x_jump=float(sum(r.max_x_jump for r in runs) / len(runs)),
            max_jump_air_time=int(sum(r.max_jump_air_time for r in runs) / len(runs)),
            kills_total=int(sum(r.kills_total for r in runs) / len(runs)),
            kills_by_type={
                key: int(sum(r.kills_by_type.get(key, 0) for r in runs) / len(runs))
                for key in ("stomp", "fire", "shell", "fall")
            },
            current_lives=first.current_lives,
            current_coins=int(sum(r.current_coins for r in runs) / len(runs)),
            num_collected_mushrooms=int(sum(r.num_collected_mushrooms for r in runs) / len(runs)),
            num_collected_fireflower=int(sum(r.num_collected_fireflower for r in runs) / len(runs)),
            num_collected_tile_coins=int(sum(r.num_collected_tile_coins for r in runs) / len(runs)),
            num_destroyed_bricks=int(sum(r.num_destroyed_bricks for r in runs) / len(runs)),
            num_bump_question_block=int(sum(r.num_bump_question_block for r in runs) / len(runs)),
            num_bump_brick=int(sum(r.num_bump_brick for r in runs) / len(runs)),
            num_hurts=int(sum(r.num_hurts for r in runs) / len(runs)),
            game_status="WIN" if won else first.game_status,
            bridge_runs=len(runs),
            bridge_win_rate=float(wins / max(1, len(runs))),
            bridge_completion_std=float(np.std(completion_values)),
        )

    @staticmethod
    def _fallback_result() -> GameplayResult:
        """Return a zeroed loss result used when the bridge cannot run."""
        return GameplayResult(
            won=False,
            completion_ratio=0.0,
            remaining_time_ms=0,
            mario_mode=0,
            num_jumps=0,
            max_x_jump=0.0,
            max_jump_air_time=0,
            kills_total=0,
            kills_by_type={"stomp": 0, "fire": 0, "shell": 0, "fall": 0},
            current_lives=0,
            current_coins=0,
            num_collected_mushrooms=0,
            num_collected_fireflower=0,
            num_collected_tile_coins=0,
            num_destroyed_bricks=0,
            num_bump_question_block=0,
            num_bump_brick=0,
            num_hurts=0,
            game_status="LOSE",
            bridge_runs=1,
            bridge_win_rate=0.0,
            bridge_completion_std=0.0,
        )
