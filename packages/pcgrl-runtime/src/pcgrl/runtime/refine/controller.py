from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from pcgrl.core.domain.level import LevelGrid
from pcgrl.core.settings.schema import RuntimeConfig
from pcgrl.core.tiles.catalog import TileCatalog
from pcgrl.runtime.metrics.intent import INTENT_KEYS, compute_intent_vector
from pcgrl.runtime.refine.context import (
    build_context,
    target_amplitude,
    target_count,
    target_gap_score,
    target_mean_delta,
)
from pcgrl.runtime.refine.gates import RefineGate, grid_from_tiles
from pcgrl.runtime.refine.ops import (
    Edit,
    apply_edit,
    plan_add_blocks,
    plan_add_pipe,
    plan_add_platform,
    plan_add_tiles_on_surface,
    plan_cap_heights,
    plan_dig_gap,
    plan_fill_gaps,
    plan_flatten,
    plan_raise_blocks,
    plan_raise_hill,
    plan_remove_blocks,
    plan_remove_pipes,
    plan_remove_pipes_above,
    plan_remove_platforms,
    plan_remove_tiles,
    plan_repair_pipes,
    plan_roughen,
    plan_smooth,
    plan_trim_pipes,
    revert_edit,
)

_DIM = {key: index for index, key in enumerate(INTENT_KEYS)}
_GATELESS_OPS = frozenset({"remove_coins", "add_coins"})
_PHASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "subtract",
        (
            "enemy_density",
            "coin_density",
            "interactive_block_density",
            "pipe_density",
            "gap_difficulty",
            "verticality",
            "platform_usage",
        ),
    ),
    ("terrain", ("gap_difficulty", "verticality", "path_smoothness")),
    ("envelope_add", ("pipe_density", "interactive_block_density", "platform_usage")),
    ("inert_add", ("enemy_density", "coin_density")),
)


@dataclass(frozen=True)
class AppliedEdit:
    """Log entry for one attempted atomic edit.

    Attributes:
        op: Planner name.
        dim: Intent dimension the edit served.
        cell_count: Number of tile writes in the edit.
        accepted: Whether the gate kept the edit.
        reject_reason: Gate failure label when rejected.
    """

    op: str
    dim: str
    cell_count: int
    accepted: bool
    reject_reason: str | None


@dataclass(frozen=True)
class RefineResult:
    """Outcome of one deterministic refinement run.

    Attributes:
        level: Refined level (the input level when nothing was applied).
        applied: Whether any edit survived the gates.
        converged: Whether every dimension ended within tolerance.
        passes: Number of full passes executed.
        measured_before: Intent vector before refinement.
        measured_after: Intent vector after refinement.
        edits: Per-edit acceptance log.
    """

    level: LevelGrid
    applied: bool
    converged: bool
    passes: int
    measured_before: tuple[float, ...]
    measured_after: tuple[float, ...]
    edits: tuple[AppliedEdit, ...]


class IntentRefiner:
    """Deterministic measured-vs-target editor over a stitched level.

    Description:
        Inverts each intent-metric formula into targeted tile edits and
        applies them one atomic edit at a time, gated by the structural
        validators plus the spawn-to-flag flood (failed edits are reverted).
        Dimensions are processed in interference-minimising phases —
        subtract excess material, settle terrain, add envelope structures,
        then inert decorations — and the loop repeats until every dimension
        sits within tolerance or a pass makes no progress. The same planners
        back the environment's edit-action phase, so the polish stage and the
        learned policy share one edit semantics.
    """

    def __init__(
        self,
        config: RuntimeConfig,
        catalog: TileCatalog,
        *,
        tolerance: float = 0.05,
        max_passes: int = 3,
        max_edits_per_dim: int = 64,
    ) -> None:
        """Bind context, gate, and convergence parameters.

        Args:
            config: Runtime configuration.
            catalog: Tile catalog.
            tolerance: Per-dimension absolute error target.
            max_passes: Full refinement passes before giving up.
            max_edits_per_dim: Edit budget per dimension per pass.
        """
        self._config = config
        self._catalog = catalog
        self._ctx = build_context(config, catalog)
        self._gate = RefineGate(config, catalog)
        self._tolerance = tolerance
        self._max_passes = max_passes
        self._max_edits_per_dim = max_edits_per_dim
        self._baseline: frozenset[str] = frozenset()

    def refine(self, level: LevelGrid, target: np.ndarray) -> RefineResult:
        """Refine one level toward the target intent vector.

        Args:
            level: Stitched level to edit.
            target: Intent vector in ``[0, 1]^8``.

        Returns:
            ``RefineResult`` with the edited level and the acceptance log.
        """
        target = np.asarray(target, dtype=np.float32)
        tiles = level.tiles.astype(np.int16).copy()
        measured_before = self._measure(tiles)
        log: list[AppliedEdit] = []
        self._baseline = self._gate.hard_rules(tiles)
        for edit in plan_repair_pipes(tiles, self._ctx):
            snapshot = apply_edit(tiles, edit)
            ok, reason = self._gate.check(tiles, baseline=self._baseline)
            if not ok:
                revert_edit(tiles, snapshot)
            else:
                self._baseline = self._gate.hard_rules(tiles)
            log.append(
                AppliedEdit(
                    op=edit.op,
                    dim="repair",
                    cell_count=len(edit.cells),
                    accepted=ok,
                    reject_reason=reason,
                )
            )
        passes = 0
        for _ in range(self._max_passes):
            passes += 1
            accepted_this_pass = 0
            for phase, dims in _PHASES:
                for dim in dims:
                    accepted_this_pass += self._refine_dim(
                        tiles, target, phase, dim, log
                    )
            measured = self._measure(tiles)
            if self._within_tolerance(measured, target):
                break
            if accepted_this_pass == 0:
                break
        self._literal_flat_cleanup(tiles, target, log)
        measured_after = self._measure(tiles)
        applied = any(entry.accepted for entry in log)
        refined = level.copy_with_tiles(tiles) if applied else level
        return RefineResult(
            level=refined,
            applied=applied,
            converged=self._within_tolerance(measured_after, target),
            passes=passes,
            measured_before=tuple(float(v) for v in measured_before),
            measured_after=tuple(float(v) for v in measured_after),
            edits=tuple(log),
        )

    def _measure(self, tiles: np.ndarray) -> np.ndarray:
        """Intent vector of the working array."""
        return compute_intent_vector(
            grid_from_tiles(tiles, self._config), self._catalog, self._config
        )

    def _literal_flat_cleanup(
        self, tiles: np.ndarray, target: np.ndarray, log: list[AppliedEdit]
    ) -> None:
        """Deliver the literal reading of a flat-and-smooth request.

        The achievable bands map any envelope amplitude below their 5th
        percentile to a measured zero, so sub-band mounds, tall pipes, and
        floating platforms are metrically invisible while remaining visible
        to a player. The per-dimension loop early-stops on such targets and
        never reaches its own literal-flat branch; this target-driven pass
        runs unconditionally when the request itself is literal flat
        (verticality at most 0.05 with smoothness at least 0.9). It runs
        after the per-dimension loop so gaps are already filled: floating
        platforms over open pits are load-bearing for the flood gate and
        their removal would be rejected before that. Unlike the
        error-driven branch it uses literal caps, not the banded amplitude
        (whose zero inverts to the band's 5th-percentile raw amplitude and
        would spare exactly the floaters being removed); pipes and
        platforms survive only when their own targets explicitly ask for
        them. Each stage replans on the tiles the previous stage mutated:
        flattening lowers the ground, which changes which structures read
        as floating. It can only remove sub-band structure, so banded
        errors never worsen.
        """
        literal_flat = (
            float(target[_DIM["verticality"]]) <= 0.05
            and float(target[_DIM["path_smoothness"]]) >= 0.9
        )
        if not literal_flat:
            return
        ctx = self._ctx
        full = range(0, ctx.width)
        base = self._envelope_min(tiles)
        amplitude = target_amplitude(target[_DIM["verticality"]], self._config)
        pipe_cap = (
            base if float(target[_DIM["pipe_density"]]) <= 0.05 else base + 2
        )
        platform_cap = (
            base
            if float(target[_DIM["platform_usage"]]) <= 0.05
            else base + amplitude
        )
        stages: tuple[Callable[[], list[Edit]], ...] = (
            lambda: plan_flatten(tiles, ctx, base_height=2, columns=full),
            lambda: plan_remove_pipes_above(tiles, ctx, cap=pipe_cap, columns=full),
            lambda: plan_remove_platforms(
                tiles, ctx, columns=full, above_height=platform_cap
            ),
        )
        for stage in stages:
            for edit in stage():
                if not edit.cells:
                    continue
                snapshot = apply_edit(tiles, edit)
                ok, reason = self._gate.check(tiles, baseline=self._baseline)
                if not ok:
                    revert_edit(tiles, snapshot)
                elif self._baseline:
                    self._baseline &= self._gate.hard_rules(tiles)
                log.append(
                    AppliedEdit(
                        op=edit.op,
                        dim="literal_flat",
                        cell_count=len(edit.cells),
                        accepted=ok,
                        reject_reason=reason,
                    )
                )

    def _standing_cap(self, tiles: np.ndarray, target: np.ndarray) -> int:
        """Max standing height additive structures may create.

        Anchored on the envelope minimum plus the verticality target's
        amplitude, so envelope-raising additions can never push verticality
        past its own target.
        """
        amplitude = target_amplitude(target[_DIM["verticality"]], self._config)
        return self._envelope_min(tiles) + amplitude

    def _envelope_min(self, tiles: np.ndarray) -> int:
        """Lowest reachable standing height (the verticality baseline).

        The spawn corner is protected and often one row low, so amplitude
        targets must be anchored on the true envelope minimum, not on the
        nominal floor height.
        """
        from pcgrl.runtime.metrics.intent import traversable_path_envelope

        envelope = traversable_path_envelope(
            grid_from_tiles(tiles, self._config), self._catalog, self._config
        )
        heights = [h for h in envelope if h is not None]
        return min(heights) if heights else 2

    def _within_tolerance(self, measured: np.ndarray, target: np.ndarray) -> bool:
        """Whether every dimension error is at most the tolerance."""
        return bool(np.max(np.abs(measured - target)) <= self._tolerance)

    def _refine_dim(
        self,
        tiles: np.ndarray,
        target: np.ndarray,
        phase: str,
        dim: str,
        log: list[AppliedEdit],
    ) -> int:
        """Drive one dimension toward its target within the phase direction.

        Returns:
            Number of accepted edits.
        """
        accepted = 0
        for _ in range(self._max_edits_per_dim):
            measured = self._measure(tiles)
            delta = float(target[_DIM[dim]] - measured[_DIM[dim]])
            if abs(delta) <= self._tolerance:
                return accepted
            if phase == "subtract" and delta > 0:
                return accepted
            if phase in ("envelope_add", "inert_add") and delta < 0:
                return accepted
            edits = self._plan(tiles, target, measured, dim, delta)
            if not edits:
                return accepted
            progressed = False
            for edit in edits:
                if not edit.cells:
                    continue
                snapshot = apply_edit(tiles, edit)
                if edit.op in _GATELESS_OPS:
                    ok, reason = True, None
                else:
                    ok, reason = self._gate.check(tiles, baseline=self._baseline)
                if not ok:
                    revert_edit(tiles, snapshot)
                else:
                    progressed = True
                    accepted += 1
                    if self._baseline:
                        self._baseline &= self._gate.hard_rules(tiles)
                log.append(
                    AppliedEdit(
                        op=edit.op,
                        dim=dim,
                        cell_count=len(edit.cells),
                        accepted=ok,
                        reject_reason=reason,
                    )
                )
                if ok and len(edits) > 1:
                    current = self._measure(tiles)
                    if (
                        abs(float(target[_DIM[dim]] - current[_DIM[dim]]))
                        <= self._tolerance
                    ):
                        return accepted
            if not progressed:
                return accepted
        return accepted

    def _plan(
        self,
        tiles: np.ndarray,
        target: np.ndarray,
        measured: np.ndarray,
        dim: str,
        delta: float,
    ) -> list[Edit]:
        """Dispatch one dimension/direction to its planner."""
        ctx = self._ctx
        full = range(0, ctx.width)
        ranges = self._config.metrics.target_density_ranges
        if dim == "enemy_density":
            desired = target_count(target[_DIM[dim]], ranges.enemy.max, ctx.area)
            if delta < 0:
                return plan_remove_tiles(
                    tiles, ctx, ctx.enemy_ids, keep=desired, columns=full,
                    op="remove_enemies",
                )
            current = self._count(tiles, ctx.enemy_ids)
            return plan_add_tiles_on_surface(
                tiles, ctx, ctx.enemy_ids[0], add=desired - current, columns=full,
                op="add_enemies", spacing=3,
            )
        if dim == "coin_density":
            desired = target_count(target[_DIM[dim]], ranges.coin.max, ctx.area)
            if delta < 0:
                return plan_remove_tiles(
                    tiles, ctx, (ctx.coin_id,), keep=desired, columns=full,
                    op="remove_coins",
                )
            current = self._count(tiles, (ctx.coin_id,))
            return plan_add_tiles_on_surface(
                tiles, ctx, ctx.coin_id, add=desired - current, columns=full,
                op="add_coins", height_above=2, spacing=2,
            )
        if dim == "interactive_block_density":
            desired = target_count(target[_DIM[dim]], ranges.block.max, ctx.area)
            if delta < 0:
                return plan_remove_blocks(tiles, ctx, keep=desired, columns=full)
            current = self._count(tiles, ctx.block_ids)
            platform_headroom = (
                float(target[_DIM["platform_usage"]] - measured[_DIM["platform_usage"]])
                > self._tolerance
            )
            return plan_add_blocks(
                tiles, ctx, add=desired - current, columns=full,
                mountable=platform_headroom,
                max_standing_height=self._standing_cap(tiles, target),
            )
        if dim == "pipe_density":
            desired = target_count(target[_DIM[dim]], ranges.pipe.max, ctx.area)
            if delta < 0:
                return plan_remove_pipes(tiles, ctx, keep_tiles=desired, columns=full)
            spacing = 3 if float(target[_DIM[dim]]) >= 0.8 else None
            # Shorter pipes inject fewer envelope steps when the target also
            # asks for a smooth path (each pipe top is two unsmoothable edges).
            pipe_height = (
                2 if float(target[_DIM["path_smoothness"]]) >= 0.8 else 3
            )
            return plan_add_pipe(
                tiles, ctx, pipe_height=pipe_height, columns=full, spacing=spacing
            )
        if dim == "gap_difficulty":
            raw = target_gap_score(target[_DIM[dim]], ctx.width)
            if delta < 0:
                return plan_fill_gaps(tiles, ctx, target_score=raw, columns=full)
            for gap_width in (4, 3, 5):
                edits = plan_dig_gap(tiles, ctx, gap_width=gap_width, columns=full)
                if edits:
                    return edits
            return []
        if dim == "verticality":
            amplitude = target_amplitude(target[_DIM[dim]], self._config)
            base = self._envelope_min(tiles)
            if delta < 0:
                edits = plan_trim_pipes(tiles, ctx, max_height=3, columns=full)
                # The band maps any amplitude <= band_min to 0, but a user
                # asking for the extreme together with a near-max smoothness
                # means literally flat: deliver the literal reading (it never
                # hurts the metric, and roughen cannot fire to conflict).
                literal_flat = (
                    float(target[_DIM[dim]]) <= 0.05
                    and float(target[_DIM["path_smoothness"]]) >= 0.9
                )
                if amplitude <= 1 or literal_flat:
                    edits += plan_flatten(tiles, ctx, base_height=2, columns=full)
                else:
                    edits += plan_cap_heights(
                        tiles, ctx, cap=base + amplitude, columns=full
                    )
                edits += plan_remove_pipes_above(
                    tiles, ctx, cap=base + amplitude, columns=full
                )
                edits += plan_remove_platforms(
                    tiles, ctx, columns=full, above_height=base + amplitude
                )
                return edits
            smooth_target = float(target[_DIM["path_smoothness"]])
            peak = min(base + amplitude, self._ctx.height - 2)
            return plan_raise_hill(
                tiles, ctx, peak_height=peak,
                ramp_step=1 if smooth_target >= 0.5 else 2, columns=full,
            )
        if dim == "path_smoothness":
            if delta > 0:
                edits = plan_smooth(tiles, ctx, max_step=1, columns=full)
                platform_satisfied = (
                    float(
                        target[_DIM["platform_usage"]]
                        - measured[_DIM["platform_usage"]]
                    )
                    <= self._tolerance
                )
                if platform_satisfied:
                    # Mountable block tops are envelope steps no ground op can
                    # smooth; lifting them out of jump reach removes the spikes
                    # without touching the block count.
                    edits += plan_raise_blocks(
                        tiles, ctx, min_height_above=6, columns=full
                    )
                return edits
            needed = target_mean_delta(target[_DIM[dim]], self._config)
            present = target_mean_delta(float(measured[_DIM[dim]]), self._config)
            amplitude = 1 if float(target[_DIM["verticality"]]) <= 0.3 else 2
            boundaries = max(1, math.ceil((needed - present) * ctx.width / amplitude))
            return plan_roughen(
                tiles, ctx, amplitude=amplitude, boundaries=boundaries, columns=full,
            )
        if dim == "platform_usage":
            if delta < 0:
                return plan_remove_platforms(
                    tiles, ctx, columns=full
                ) + plan_raise_blocks(
                    tiles, ctx, min_height_above=6, columns=full
                )
            smooth_target = float(target[_DIM["path_smoothness"]])
            height_above = 2 if smooth_target >= 0.8 else 3
            cap = self._standing_cap(tiles, target)
            # A platform moves the elevated fraction by ~width/path-columns;
            # near the target a narrow platform avoids overshooting past the
            # tolerance and oscillating against the removal op, while a wide
            # one under a smooth-path target amortises its two edge steps.
            if smooth_target >= 0.8 and delta > 0.15:
                preferred = 8
            elif delta > 0.15:
                preferred = 6
            else:
                preferred = 3
            for width in dict.fromkeys((preferred, 4, 3)):
                edits = plan_add_platform(
                    tiles, ctx, width=width, height_above=height_above,
                    columns=full, max_standing_height=cap,
                )
                if edits:
                    return edits
            return []
        return []

    def _count(self, tiles: np.ndarray, ids: tuple[int, ...]) -> int:
        """Total occurrences of the given tile ids."""
        return int(np.isin(tiles, list(ids)).sum())
