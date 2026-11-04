from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from pcgrl.core.domain.level import AreaType, LevelGrid
from pcgrl.core.library.types import ChunkFamily, ChunkLibrary
from pcgrl.core.settings.schema import RuntimeConfig
from pcgrl.core.tiles.catalog import TileCatalog
from pcgrl.runtime.layout import resolve_layout
from pcgrl.runtime.metrics.usefulness import block_names

_BUDGET_SLACK_RATIO = 0.25
_BUDGET_SLACK_CELLS = 4.0
_BUDGET_KEYS = ("enemy", "coin", "interactive_block", "pipe")
_GROUND_CATEGORY_NAMES = frozenset({"solid", "pipe", "block", "platform"})


@dataclass(frozen=True)
class ComposeState:
    """Immutable composition state: the sequence of chosen chunk ids.

    Description:
        ``counts`` accumulates the per-chunk budget counts (enemy, coin,
        interactive block, pipe) so the budget mask never re-scans tiles.
        ``closed`` flips when an ENDING chunk is placed.

    """

    chunk_ids: tuple[int, ...] = ()
    used_width: int = 0
    counts: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    closed: bool = False

    @property
    def num_decisions(self) -> int:
        """Number of chunks placed so far."""
        return len(self.chunk_ids)


@dataclass(frozen=True)
class _ChunkData:
    """Annotation columns of the library, vectorised for mask computation."""

    widths: np.ndarray
    entry_heights: np.ndarray
    exit_heights: np.ndarray
    leading_gaps: np.ndarray
    trailing_gaps: np.ndarray
    is_start: np.ndarray
    is_ending: np.ndarray
    counts: np.ndarray


def _edge_gap_width(
    tiles: np.ndarray,
    ground_categories: frozenset[str],
    *,
    from_left: bool,
    catalog: TileCatalog,
) -> int:
    """Number of consecutive bottomless columns at one edge of a chunk."""
    width = tiles.shape[1]
    columns = range(width) if from_left else range(width - 1, -1, -1)
    gap = 0
    for column in columns:
        bottom = catalog.tile_for_id(int(tiles[-1, column]))
        if bottom.category.value in ground_categories:
            break
        gap += 1
    return gap


class ComposerCore:
    """Library + config bound together; owns masks, transitions and stitching.

    Description:
        Pure (no gym): shared by the composer environment, the non-RL baselines
        and the intent viability map, so every decision rule obeys exactly the
        same composition rules:

        - width arithmetic: chunks fill ``game.width_tiles`` left to right; the
          level must close with an ENDING leaving at most ``width_slack_tiles``
          of padding.
        - seam physics: |exit_height - entry_height| <= ``validation.jump_height``
          for grounded seams; a gap spanning the seam must be at most
          ``validation.jump_reach`` columns wide.
        - families: episodes start with START and end with ENDING when required.
        - the anti-duplicate window and intent budgets are SOFT rules, dropped in
          that order if they would empty the mask, because the episode must always
          be able to close.

    """

    def __init__(
        self, library: ChunkLibrary, config: RuntimeConfig, catalog: TileCatalog
    ) -> None:
        """Vectorise the library annotations and derive the budget caps."""
        if len(library) == 0:
            raise ValueError("chunk library is empty")
        self.library = library
        self.config = config
        self.catalog = catalog
        self._ground_categories = _GROUND_CATEGORY_NAMES
        self._data = self._vectorize(library, catalog)
        self._min_ending_width = int(
            self._data.widths[self._data.is_ending].min()
            if self._data.is_ending.any()
            else self._data.widths.min()
        )
        area = config.game.width_tiles * config.game.height_tiles
        ranges = config.metrics.target_density_ranges
        self._budget_caps = np.array(
            [
                ranges.enemy.max * area,
                ranges.coin.max * area,
                ranges.block.max * area,
                ranges.pipe.max * area,
            ],
            dtype=np.float64,
        )

    def _vectorize(self, library: ChunkLibrary, catalog: TileCatalog) -> _ChunkData:
        """Stack every chunk annotation into arrays indexed by chunk id."""
        size = len(library)
        widths = np.zeros(size, dtype=np.int64)
        entry = np.zeros(size, dtype=np.int64)
        exit_ = np.zeros(size, dtype=np.int64)
        lead = np.zeros(size, dtype=np.int64)
        trail = np.zeros(size, dtype=np.int64)
        starts = np.zeros(size, dtype=bool)
        endings = np.zeros(size, dtype=bool)
        counts = np.zeros((size, len(_BUDGET_KEYS)), dtype=np.float64)
        block_ids = {
            catalog.id_for_name(name)
            for name in block_names(self.config)
            if name in catalog.name_to_id
        }
        for index, chunk in enumerate(library.chunks):
            widths[index] = chunk.width
            entry[index] = chunk.entry_height
            exit_[index] = chunk.exit_height
            lead[index] = _edge_gap_width(
                chunk.tiles, self._ground_categories, from_left=True, catalog=catalog
            )
            trail[index] = _edge_gap_width(
                chunk.tiles, self._ground_categories, from_left=False, catalog=catalog
            )
            starts[index] = chunk.family == ChunkFamily.START
            endings[index] = chunk.family == ChunkFamily.ENDING
            counts[index, 0] = chunk.category_counts.get("enemy", 0)
            counts[index, 1] = chunk.category_counts.get("coin", 0)
            counts[index, 2] = int(np.isin(chunk.tiles, list(block_ids)).sum())
            counts[index, 3] = chunk.category_counts.get("pipe", 0)
        return _ChunkData(
            widths=widths,
            entry_heights=entry,
            exit_heights=exit_,
            leading_gaps=lead,
            trailing_gaps=trail,
            is_start=starts,
            is_ending=endings,
            counts=counts,
        )

    def legal_chunk_mask(
        self, state: ComposeState, target: np.ndarray | None = None
    ) -> np.ndarray:
        """Boolean mask over the library for the next decision.

        Args:
            state: Current composition state.
            target: Optional intent vector for budget masking.

        Returns:
            Boolean array over chunk ids; soft rules relax when mask is empty.
        """
        structural = self._structural_mask(state)
        if not structural.any():
            return structural
        mask = structural.copy()
        if target is not None and self.config.environment.composer.intent_budget_mask:
            budget = self._budget_mask(state, target)
            if (mask & budget).any():
                mask &= budget
        duplicate = self._duplicate_mask(state)
        if (mask & duplicate).any():
            mask &= duplicate
        return mask

    def _structural_mask(self, state: ComposeState) -> np.ndarray:
        """Hard rules: width arithmetic, families and seam physics.

        Non-ending chunks must leave room for some ending afterwards; endings
        must close the level within the padding slack. Grounded seams obey the
        jump-height rule; gap seams (either side bottomless) obey the combined
        gap-width-vs-jump-reach rule.
        """
        data = self._data
        if state.closed:
            return np.zeros(len(self.library), dtype=bool)
        env = self.config.environment.composer
        total = self.config.game.width_tiles
        remaining = total - state.used_width
        mask = data.widths <= remaining
        ending_ok = data.is_ending & (remaining - data.widths <= env.width_slack_tiles)
        middle_ok = ~data.is_ending & (
            remaining - data.widths >= self._min_ending_width
        )
        if env.require_ending_family:
            mask &= ending_ok | middle_ok
        if state.num_decisions == 0:
            if env.require_start_family and (mask & data.is_start).any():
                mask &= data.is_start
            return mask
        mask &= ~data.is_start
        previous = self.library[state.chunk_ids[-1]]
        physics = self.config.validation
        prev_exit = previous.exit_height
        prev_trail = int(self._data.trailing_gaps[state.chunk_ids[-1]])
        grounded = (data.entry_heights > 0) & (prev_exit > 0)
        jumpable = np.abs(data.entry_heights - prev_exit) <= physics.jump_height
        seam_gap = prev_trail + data.leading_gaps
        gap_seam_ok = (seam_gap <= physics.jump_reach) & (
            (data.entry_heights == 0) | (prev_exit == 0)
        )
        mask &= (grounded & jumpable) | gap_seam_ok
        return mask

    def _duplicate_mask(self, state: ComposeState) -> np.ndarray:
        """Soft rule: block chunks used within the rolling duplicate window."""
        window = self.config.environment.composer.duplicate_window
        mask = np.ones(len(self.library), dtype=bool)
        if window > 0:
            for chunk_id in state.chunk_ids[-window:]:
                mask[chunk_id] = False
        return mask

    def _budget_mask(self, state: ComposeState, target: np.ndarray) -> np.ndarray:
        """Soft rule: block chunks that grossly overshoot the intent budgets.

        Budgets derive from the corpus density bands scaled by the target;
        the slack is loose on purpose — fine alignment is the reward's job,
        the mask only stops gross overshoot.
        """
        budgets = self._budget_caps * np.clip(
            np.asarray(target[: len(_BUDGET_KEYS)], dtype=np.float64), 0.0, 1.0
        )
        limits = budgets * (1.0 + _BUDGET_SLACK_RATIO) + _BUDGET_SLACK_CELLS
        current = np.asarray(state.counts, dtype=np.float64)
        projected = current[None, :] + self._data.counts
        return (projected <= limits[None, :]).all(axis=1)  # type: ignore[no-any-return]

    def trailing_gap(self, chunk_id: int) -> int:
        """Bottomless columns at the right edge of a chunk."""
        return int(self._data.trailing_gaps[chunk_id])

    def budget_counts(self, chunk_id: int) -> np.ndarray:
        """Per-chunk (enemy, coin, interactive_block, pipe) tile counts."""
        return self._data.counts[chunk_id]  # type: ignore[no-any-return]

    def apply_chunk(self, state: ComposeState, chunk_id: int) -> ComposeState:
        """Place one chunk and return the successor state.

        Args:
            state: Current composition state.
            chunk_id: Library index of the chunk to append.

        Returns:
            New immutable ``ComposeState`` with updated counts and width.
        """
        chunk = self.library[chunk_id]
        counts = tuple(
            float(value)
            for value in (
                np.asarray(state.counts, dtype=np.float64)
                + self._data.counts[chunk_id]
            )
        )
        return replace(
            state,
            chunk_ids=(*state.chunk_ids, chunk_id),
            used_width=state.used_width + chunk.width,
            counts=counts,
            closed=bool(self._data.is_ending[chunk_id]),
        )

    def is_terminal(self, state: ComposeState) -> bool:
        """Return True when the level closed or no structural move remains.

        Args:
            state: Current composition state.
        """
        return state.closed or not self._structural_mask(state).any()

    def stitch_level(
        self, state: ComposeState, *, place_endpoints: bool = True
    ) -> LevelGrid:
        """Assemble chosen chunks into a full game grid.

        Args:
            state: Composition state with chunk sequence.
            place_endpoints: When True, stamp player/flag and reserved cells.

        Returns:
            ``LevelGrid`` with provenance metadata on the composition.
        """
        config = self.config
        height = config.game.height_tiles
        width = config.game.width_tiles
        empty_id = self.catalog.id_for_name("empty")
        solid_id = self.catalog.id_for_name(config.layout.ground_tile)
        tiles = np.full((height, width), empty_id, dtype=np.int16)
        cursor = 0
        for chunk_id in state.chunk_ids:
            chunk = self.library[chunk_id]
            chunk_width = min(chunk.width, width - cursor)
            if chunk_width <= 0:
                break
            tiles[height - chunk.height :, cursor : cursor + chunk_width] = chunk.tiles[
                :, :chunk_width
            ]
            cursor += chunk_width
        if cursor < width:
            tiles[height - 2 :, cursor:] = solid_id
        if place_endpoints:
            resolved = resolve_layout(config.layout, height=height, width=width)
            for y, x in resolved.reserved:
                if y == resolved.ground_row:
                    tiles[y, x] = solid_id
                elif (y, x) not in (resolved.player, resolved.flagpole):
                    tiles[y, x] = empty_id
            tiles[resolved.player] = self.catalog.id_for_name("player")
            tiles[resolved.flagpole] = self.catalog.id_for_name("flagpole")
        return LevelGrid(
            tiles=tiles,
            width_tiles=width,
            height_tiles=height,
            tile_size_px=config.game.tile_size_px,
            area_type=AreaType.OVERWORLD,
            metadata={
                "chunk_ids": list(state.chunk_ids),
                "provenance": [
                    {
                        "level_id": self.library[chunk_id].level_id,
                        "column_offset": self.library[chunk_id].column_offset,
                    }
                    for chunk_id in state.chunk_ids
                ],
            },
        )
