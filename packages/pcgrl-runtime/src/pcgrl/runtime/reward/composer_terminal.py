from __future__ import annotations

import numpy as np

from pcgrl.core.domain.level import LevelGrid
from pcgrl.core.settings.schema import RuntimeConfig
from pcgrl.core.tiles.catalog import TileCatalog
from pcgrl.runtime.metrics.intent import compute_intent_vector
from pcgrl.runtime.metrics.tpkl import (
    PatternReference,
    empty_pattern_key,
    style_score,
)
from pcgrl.runtime.validators.reachability import reachable_cells

_PARTIAL_PLAY_FACTOR = 0.5


class ComposerTerminalReward:
    """Terminal-only composer reward (gamma = 1.0), every term in [0, 1].

    Description:
        R = w_align * [m * (1 - mean|d|) + M * (1 - max|d|)]
          + w_style * exp(-TPKL_k(level || corpus_train))
          + w_play  * playable_flood(spawn -> flag)
          - w_dup   * dup_penalty(chunk window)

        The max-term of the alignment blocks sacrificing single intent dims. The
        style term EXCLUDES the all-empty pattern from both distributions: corpus
        skies are mostly empty windows, and without the exclusion an empty level
        scores ~0.7 style — a degenerate optimum. Components are returned
        separately for callback logging.

    """

    def __init__(
        self,
        config: RuntimeConfig,
        catalog: TileCatalog,
        pattern_reference: PatternReference,
    ) -> None:
        """Bind config weights and precompute the excluded empty-pattern key.

        Args:
            config: Runtime configuration with composer reward weights.
            catalog: Tile catalog.
            pattern_reference: Corpus train-split pattern distributions.
        """
        self._config = config
        self._catalog = catalog
        self._reference = pattern_reference
        settings = config.reward.composer
        self._exclude = empty_pattern_key(catalog.id_for_name("empty"), settings.style_pattern_size)

    def compute(
        self,
        level: LevelGrid,
        target: np.ndarray,
        *,
        chunk_ids: tuple[int, ...] = (),
        measured: np.ndarray | None = None,
    ) -> tuple[float, dict[str, float]]:
        """Score a finished composition.

        Args:
            level: Stitched level grid.
            target: Intent target vector.
            chunk_ids: Sequence of chosen chunk ids for duplicate penalty.
            measured: Optional precomputed intent vector.

        Returns:
            Tuple of (total reward, per-term component dict).
        """
        settings = self._config.reward.composer
        if measured is None:
            measured = compute_intent_vector(level, self._catalog, self._config)
        deltas = np.abs(
            np.asarray(measured, dtype=np.float64) - np.asarray(target, dtype=np.float64)
        )
        alignment = settings.alignment_mean_weight * (1.0 - float(deltas.mean()))
        alignment += settings.alignment_max_weight * (1.0 - float(deltas.max()))
        style = style_score(
            level.tiles,
            self._reference,
            settings.style_pattern_size,
            exclude=self._exclude,
        )
        playability = self._playability(level)
        duplicate = self._duplicate_penalty(chunk_ids)
        components = {
            "alignment": float(alignment),
            "style": float(style),
            "playability": float(playability),
            "dup_penalty": float(duplicate),
        }
        total = (
            settings.weight_alignment * alignment
            + settings.weight_style * style
            + settings.weight_playability * playability
            - settings.weight_duplicate * duplicate
        )
        return float(total), components

    def _playability(self, level: LevelGrid) -> float:
        """Flood completion from spawn under the configured jump physics.

        1.0 when the flag column is reached; otherwise partial credit for the
        rightmost column reached, scaled so it never beats an actual win.
        """
        catalog = self._catalog
        flag_id = catalog.id_for_name("flagpole")
        flag_cols = np.argwhere(level.tiles == flag_id)
        if flag_cols.size == 0:
            return 0.0
        flag_col = int(flag_cols[0][1])
        reach = reachable_cells(
            level,
            catalog,
            jump_height=self._config.validation.jump_height,
            jump_reach=self._config.validation.jump_reach,
        )
        if not reach.standable:
            return 0.0
        rightmost = max(col for _, col in reach.standable)
        if rightmost >= flag_col:
            return 1.0
        return _PARTIAL_PLAY_FACTOR * rightmost / max(1, flag_col)

    def _duplicate_penalty(self, chunk_ids: tuple[int, ...]) -> float:
        """Fraction of decisions that repeat a chunk seen inside the window."""
        window = self._config.environment.composer.duplicate_window
        if window <= 0 or len(chunk_ids) < 2:
            return 0.0
        violations = sum(
            1
            for index in range(1, len(chunk_ids))
            if chunk_ids[index] in chunk_ids[max(0, index - window) : index]
        )
        return violations / (len(chunk_ids) - 1)
