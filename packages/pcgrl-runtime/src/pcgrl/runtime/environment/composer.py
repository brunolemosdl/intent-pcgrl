from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pcgrl.core.domain.level import LevelGrid
from pcgrl.runtime.environment.composer_obs import (
    build_composer_observation,
    composer_observation_size,
    measured_partial_intent,
)
from pcgrl.runtime.environment.composer_state import ComposerCore, ComposeState
from pcgrl.runtime.environment.context import EnvContext
from pcgrl.runtime.environment.edit_actions import (
    decode_edit_action,
    edit_action_count,
    edit_action_masks,
    plan_edit_action,
)
from pcgrl.runtime.metrics.intent import compute_intent_vector
from pcgrl.runtime.refine.context import build_context
from pcgrl.runtime.refine.gates import RefineGate, grid_from_tiles
from pcgrl.runtime.refine.ops import apply_edit, revert_edit


def sample_training_target(
    config_dim: int,
    mix: Any,
    viability_map: Any,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one training target from the configured mixture.

    Description:
        Blends the corpus-anchored viability cloud (style-typical targets),
        uniform cube exploration, and corner emphasis where each dimension
        independently snaps to 0, 1, or stays uniform — the regions only the
        edit phase can satisfy. Falls back to uniform when no viability map
        exists.

    Args:
        config_dim: Intent vector dimensionality.
        mix: ``TargetMixConfig`` with the three shares.
        viability_map: Optional ``IntentViabilityMap``.
        rng: NumPy random generator.

    Returns:
        Float32 target vector in ``[0, 1]^dim``.
    """
    roll = rng.random()
    if viability_map is not None and roll < mix.viability:
        return np.asarray(viability_map.sample_target(rng), dtype=np.float32)
    if roll < mix.viability + mix.uniform or viability_map is None:
        return rng.uniform(0.0, 1.0, size=config_dim).astype(np.float32)
    choices = rng.integers(0, 3, size=config_dim)
    uniform = rng.uniform(0.0, 1.0, size=config_dim)
    target = np.where(choices == 0, 0.0, np.where(choices == 1, 1.0, uniform))
    return target.astype(np.float32)


class MarioComposerEnv(gym.Env[np.ndarray, int]):
    """Two-phase compose-then-edit environment over one Discrete action space.

    Description:
        Actions ``[0, K)`` pick library chunks; actions ``[K, K + E)`` are
        macro-edits from the shared refine-op vocabulary plus STOP. An
        episode composes until an ENDING chunk closes the level, then edits
        the stitched tiles under per-edit structural+flood gating (failed
        edits revert and consume budget) until STOP or the edit budget runs
        out. The terminal-only reward is computed on the edited level, so
        the policy learns when chunk material suffices and when targeted
        edits are needed — extremes included. ``gamma = 1.0`` still applies
        (one terminal reward over ~10-40 decisions).
    """

    def __init__(self, context: EnvContext, seed: int | None = None) -> None:
        """Validate composer context and build observation/action spaces.

        Args:
            context: Shared env context with library and terminal reward.
            seed: Optional RNG seed.

        Raises:
            ValueError: When required composer context members are missing.
        """
        if context.chunk_library is None:
            raise ValueError(
                "EnvContext.chunk_library is required for the composer env; "
                "run the 'library' stage and build the context with kind=composer"
            )
        if context.composer_terminal is None:
            raise ValueError("EnvContext.composer_terminal is required for the composer env")
        self.metadata = {"render_modes": []}
        self._context = context
        self._core = ComposerCore(context.chunk_library, context.config, context.catalog)
        self._refine_ctx = build_context(context.config, context.catalog)
        self._gate = RefineGate(context.config, context.catalog)
        self._rng = np.random.default_rng(seed)
        widths = self._core.library.widths()
        self._max_decisions = context.config.game.width_tiles // max(1, min(widths)) + 2
        self._edit_budget = context.config.environment.composer.edit_budget
        self._num_chunks = len(self._core.library)
        self._state = ComposeState()
        self._tiles: np.ndarray | None = None
        self._edits_left = 0
        self._edit_log: list[dict[str, Any]] = []
        self._gate_baseline: frozenset[str] = frozenset()
        # Rejected edits revert the tiles, so measured intent and the edit
        # mask stay valid across no-op steps; the dirty flag is the single
        # invalidation point (set on accepted edits and phase entry).
        self._edit_cache_dirty = True
        self._cached_measured: np.ndarray | None = None
        self._cached_edit_mask: np.ndarray | None = None
        self._intent_target = self._sample_target()
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(composer_observation_size(),),
            dtype=np.float32,
        )
        # gymnasium>=1.4 types Discrete as Space[np.int64]; older releases as Space[int].
        self.action_space = spaces.Discrete(  # type: ignore[assignment,unused-ignore]
            self._num_chunks + edit_action_count()
        )

    @property
    def in_edit_phase(self) -> bool:
        """Whether the episode is editing the stitched level."""
        return self._tiles is not None

    def set_intent(self, target: np.ndarray) -> None:
        """Condition the episode on an explicit intent vector.

        Args:
            target: Intent vector of shape ``(target_vector_dim,)``.
        """
        self._intent_target = np.asarray(target, dtype=np.float32)

    def action_masks(self) -> np.ndarray:
        """Legal-action mask over chunks + edits for the current phase.

        Returns:
            Boolean array of length ``num_chunks + edit_action_count()``.
        """
        mask = np.zeros(self._num_chunks + edit_action_count(), dtype=bool)
        if self._tiles is None:
            mask[: self._num_chunks] = self._core.legal_chunk_mask(self._state, self._intent_target)
        else:
            if self._edit_cache_dirty or self._cached_edit_mask is None:
                self._cached_edit_mask = edit_action_masks(self._tiles, self._refine_ctx)
            mask[self._num_chunks :] = self._cached_edit_mask
        return mask

    def get_level(self) -> LevelGrid:
        """Current level: edited tiles when editing, else the stitched state.

        Returns:
            Level grid with endpoints and provenance metadata.
        """
        if self._tiles is not None:
            return self._level_from_tiles()
        return self._core.stitch_level(self._state)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode.

        Args:
            seed: Optional RNG seed override.
            options: Gymnasium options; ``target`` overrides intent sampling.

        Returns:
            Tuple of (observation, info) with ``action_mask`` in info.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if options and "target" in options:
            self.set_intent(np.asarray(options["target"], dtype=np.float32))
        else:
            self._intent_target = self._sample_target()
        self._state = ComposeState()
        self._tiles = None
        self._edits_left = 0
        self._edit_log = []
        self._gate_baseline = frozenset()
        self._edit_cache_dirty = True
        self._cached_measured = None
        self._cached_edit_mask = None
        observation = self._build_observation()
        return observation, {"action_mask": self.action_masks()}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance one decision in whichever phase the episode is in.

        Args:
            action: Chunk index, edit-action index, or STOP.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
            Illegal actions terminate immediately with terminal reward.
        """
        mask = self.action_masks()
        info: dict[str, Any] = {}
        if not bool(mask[action]):
            reward, components = self._terminal_reward()
            info["illegal_action"] = True
            info["reward_components"] = components
            return self._build_observation(), reward, True, False, info
        if self._tiles is None:
            return self._step_compose(int(action), info)
        return self._step_edit(int(action), info)

    def _step_compose(
        self, action: int, info: dict[str, Any]
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply one chunk decision, entering the edit phase on close."""
        self._state = self._core.apply_chunk(self._state, action)
        compose_done = self._core.is_terminal(self._state)
        truncated = not compose_done and self._state.num_decisions >= self._max_decisions
        if (compose_done or truncated) and self._edit_budget > 0:
            self._enter_edit_phase()
            observation = self._build_observation()
            info["action_mask"] = self.action_masks()
            return observation, 0.0, False, False, info
        reward = 0.0
        if compose_done or truncated:
            reward, components = self._terminal_reward()
            info["reward_components"] = components
            info["chunk_ids"] = list(self._state.chunk_ids)
        observation = self._build_observation()
        info["action_mask"] = self.action_masks()
        return observation, reward, compose_done, truncated, info

    def _step_edit(
        self, action: int, info: dict[str, Any]
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply one edit action; STOP or budget exhaustion ends the episode."""
        assert self._tiles is not None
        decoded = decode_edit_action(action - self._num_chunks)
        if decoded is not None:
            edits = plan_edit_action(self._tiles, self._refine_ctx, decoded)
            accepted = False
            for edit in edits:
                if not edit.cells:
                    continue
                snapshot = apply_edit(self._tiles, edit)
                ok, _ = self._gate.check(self._tiles, baseline=self._gate_baseline)
                if ok:
                    accepted = True
                    self._edit_cache_dirty = True
                else:
                    revert_edit(self._tiles, snapshot)
            self._edit_log.append(
                {
                    "op": decoded.op,
                    "segment": decoded.segment,
                    "param": decoded.param,
                    "accepted": accepted,
                }
            )
            self._edits_left -= 1
        terminated = decoded is None or self._edits_left <= 0
        reward = 0.0
        if terminated:
            reward, components = self._terminal_reward()
            info["reward_components"] = components
            info["chunk_ids"] = list(self._state.chunk_ids)
            info["edit_log"] = list(self._edit_log)
        observation = self._build_observation()
        info["action_mask"] = self.action_masks()
        return observation, reward, terminated, False, info

    def _enter_edit_phase(self) -> None:
        """Stitch once and switch the episode to the edit phase."""
        level = self._core.stitch_level(self._state)
        self._tiles = level.tiles.astype(np.int16).copy()
        self._edits_left = self._edit_budget
        self._edit_log = []
        self._gate_baseline = self._gate.hard_rules(self._tiles)
        self._edit_cache_dirty = True

    def _level_from_tiles(self) -> LevelGrid:
        """Wrap the edited tiles in a grid carrying composition provenance."""
        assert self._tiles is not None
        config = self._context.config
        return LevelGrid(
            tiles=self._tiles.copy(),
            width_tiles=config.game.width_tiles,
            height_tiles=config.game.height_tiles,
            tile_size_px=config.game.tile_size_px,
            area_type=self._core.stitch_level(self._state).area_type,
            metadata={
                "chunk_ids": list(self._state.chunk_ids),
                "edit_log": list(self._edit_log),
            },
        )

    def _sample_target(self) -> np.ndarray:
        """Sample a training target from the configured mixture."""
        return sample_training_target(
            self._context.config.target_vector_dim,
            self._context.config.environment.composer.target_mix,
            self._context.viability_map,
            self._rng,
        )

    def _terminal_reward(self) -> tuple[float, dict[str, float]]:
        """Compute the terminal reward for the current (possibly edited) level."""
        terminal = self._context.composer_terminal
        assert terminal is not None
        level = self.get_level()
        return terminal.compute(level, self._intent_target, chunk_ids=self._state.chunk_ids)

    def _build_observation(self) -> np.ndarray:
        """Pack the current state into the flat observation vector."""
        if self._tiles is not None:
            if self._edit_cache_dirty or self._cached_measured is None:
                self._cached_measured = compute_intent_vector(
                    grid_from_tiles(self._tiles, self._context.config),
                    self._context.catalog,
                    self._context.config,
                )
                self._edit_cache_dirty = False
            measured = self._cached_measured
            budget_fraction = self._edits_left / max(1, self._edit_budget)
        else:
            measured = measured_partial_intent(self._core, self._state)
            budget_fraction = 0.0
        return build_composer_observation(
            self._core,
            self._state,
            self._intent_target,
            measured=measured,
            max_decisions=self._max_decisions,
            edit_phase=self._tiles is not None,
            edit_budget_fraction=budget_fraction,
        )
