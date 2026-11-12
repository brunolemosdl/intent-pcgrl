from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from pcgrl.core.domain.level import LevelGrid
from pcgrl.core.settings.schema import RuntimeConfig
from pcgrl.core.tiles.catalog import TileCatalog
from pcgrl.runtime.metrics.intent import (
    COUNT_KEYS,
    INTENT_KEYS,
    IntentMeasurement,
    effective_target,
    measure_intent,
    target_value,
)
from pcgrl.runtime.refine.context import build_context, target_amplitude
from pcgrl.runtime.refine.corrections import (
    LOWER,
    RAISE,
    correction_needed,
    is_literal_flat,
    plan_correction,
)
from pcgrl.runtime.refine.gates import RefineGate, grid_from_tiles
from pcgrl.runtime.refine.ops import (
    Edit,
    apply_edit,
    plan_flatten,
    plan_remove_pipes_above,
    plan_remove_platforms,
    plan_repair_pipes,
    revert_edit,
)

_DIM = {key: index for index, key in enumerate(INTENT_KEYS)}
_GATELESS_OPS = frozenset({"remove_coins", "add_coins"})
# Edits made of independent placements or removals: any prefix of their cells
# is still a sound edit, so they can be cut to the tiles the budget has left.
_DIVISIBLE_OPS = frozenset(
    {
        "add_enemies",
        "remove_enemies",
        "add_coins",
        "remove_coins",
        "add_blocks",
        "remove_blocks",
    }
)
# Planned edits tried per dimension and step, and the most steps one
# refinement takes.
_CANDIDATES_PER_DIM = 3
_MAX_STEPS = 64
# Slack when comparing overall errors and float32 targets (0.05 is stored as
# 0.0500000007).
_EPS = 1e-6


def intent_objective(measured: np.ndarray, target: np.ndarray) -> float:
    """Overall intent error the refiner minimises.

    Mean absolute error plus half the worst single dimension, so a change
    that improves the average by sacrificing one dimension is not free.
    """
    error = np.abs(np.asarray(measured, dtype=np.float64) - np.asarray(target, dtype=np.float64))
    return float(error.mean() + 0.5 * error.max())


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
        measured_before: Intent vector before refinement.
        measured_after: Intent vector after refinement.
        edits: Per-edit acceptance log.
        tiles_changed: Cells that differ between the input and the result.
    """

    level: LevelGrid
    applied: bool
    converged: bool
    measured_before: tuple[float, ...]
    measured_after: tuple[float, ...]
    edits: tuple[AppliedEdit, ...]
    tiles_changed: int = 0


@dataclass(frozen=True)
class _Step:
    """The edit a refinement step settled on."""

    edit: Edit
    dim: str
    objective: float


class IntentRefiner:
    """Deterministic touch-up of a level the policy has already edited.

    Description:
        The policy composes and edits toward the request; what is left is a
        small residual. The refiner closes it with the same planners the
        edit actions use (``plan_correction``), sized over the whole level
        instead of one segment: every step plans the correction of each
        dimension that is still outside the tolerance, tries the planned
        edits under the structural validators and the spawn-to-flag flood,
        and keeps the single edit that removes the most overall error per
        tile written. It stops when no edit helps or the tile budget
        (``max_tiles``) is spent, so it can correct a level but not redraw
        it.

        Edits are judged by the overall intent error (``intent_objective``:
        mean error plus half the worst dimension), not only by the dimension
        they serve, because dimensions interact: lifting a block for
        smoothness lowers platform usage, and a platform adds envelope
        steps. Every kept edit lowers that error, so the result is never
        farther from the target than the input.

        Targets are read through ``effective_target``: a count target in
        the "none" band means zero objects.
    """

    def __init__(
        self,
        config: RuntimeConfig,
        catalog: TileCatalog,
        *,
        tolerance: float = 0.05,
        max_tiles: int | None = None,
    ) -> None:
        """Bind context, gate, and stopping parameters.

        Args:
            config: Runtime configuration.
            catalog: Tile catalog.
            tolerance: Per-dimension absolute error target.
            max_tiles: Most tile writes one refinement may keep (``None``
                lifts the bound); pipe repairs are not counted.
        """
        self._config = config
        self._catalog = catalog
        self._ctx = build_context(config, catalog)
        self._gate = RefineGate(config, catalog)
        self._tolerance = tolerance
        self._max_tiles = max_tiles
        self._spent = 0
        self._reach = -1
        self._baseline: frozenset[str] = frozenset()
        self._objective = math.inf

    def refine(self, level: LevelGrid, target: np.ndarray) -> RefineResult:
        """Refine one level toward the target intent vector.

        Args:
            level: Stitched level to edit.
            target: Intent vector in ``[0, 1]^8``.

        Returns:
            ``RefineResult`` with the edited level and the acceptance log.
        """
        target = self.effective_target(target)
        tiles = level.tiles.astype(np.int16).copy()
        measured_before = self._measure(tiles)
        log: list[AppliedEdit] = []
        self._baseline = self._gate.hard_rules(tiles)
        # Edits may not shorten how far the flood gets (the flag, normally).
        self._reach = self._gate.reach(tiles)
        self._repair_pipes(tiles, log)
        self._objective = intent_objective(self._measure(tiles), target)
        self._spent = 0
        refused: set[tuple[str, Edit]] = set()
        for _ in range(_MAX_STEPS):
            step = self._best_step(tiles, target, log, refused)
            if step is None:
                break
            apply_edit(tiles, step.edit)
            self._accept(tiles, len(step.edit.cells), step.objective)
            log.append(AppliedEdit(step.edit.op, step.dim, len(step.edit.cells), True, None))
        self._literal_flat_cleanup(tiles, target, log)
        measured_after = self._measure(tiles)
        applied = any(entry.accepted for entry in log) and not np.array_equal(
            tiles, level.tiles.astype(np.int16)
        )
        refined = level.copy_with_tiles(tiles) if applied else level
        return RefineResult(
            level=refined,
            applied=applied,
            converged=self._within_tolerance(measured_after, target),
            measured_before=tuple(float(v) for v in measured_before),
            measured_after=tuple(float(v) for v in measured_after),
            edits=tuple(log),
            tiles_changed=int((tiles != level.tiles).sum()),
        )

    def effective_target(self, target: np.ndarray) -> np.ndarray:
        """Target the refiner actually drives toward (see ``effective_target``)."""
        return effective_target(target)

    def _state(self, tiles: np.ndarray) -> IntentMeasurement:
        """Measurement (intent vector and path geometry) of the working array."""
        return measure_intent(grid_from_tiles(tiles, self._config), self._catalog, self._config)

    def _measure(self, tiles: np.ndarray) -> np.ndarray:
        """Intent vector of the working array."""
        return self._state(tiles).vector

    def _within_tolerance(self, measured: np.ndarray, target: np.ndarray) -> bool:
        """Whether every dimension error is at most the tolerance."""
        return bool(np.max(np.abs(measured - target)) <= self._tolerance + _EPS)

    def _dim_tolerance(self, dim: str, target: np.ndarray) -> float:
        """Error below which a dimension is left alone.

        A count dimension targeted at zero must reach zero: one goomba
        already measures inside the generic tolerance.
        """
        if dim in COUNT_KEYS and target_value(target, dim) == 0.0:
            return 0.0
        return self._tolerance + _EPS

    def _fit(self, edit: Edit) -> Edit | None:
        """The edit cut to the remaining tile budget, or ``None`` when it cannot fit."""
        if self._max_tiles is None:
            return edit
        room = self._max_tiles - self._spent
        if len(edit.cells) <= room:
            return edit
        if edit.op in _DIVISIBLE_OPS and room > 0:
            return Edit(op=edit.op, cells=edit.cells[:room])
        return None

    def _gate_check(self, tiles: np.ndarray, edit: Edit) -> tuple[bool, str | None]:
        """Structural and reachability verdict on the tiles after ``edit``."""
        if edit.op in _GATELESS_OPS:
            return True, None
        return self._gate.check(tiles, baseline=self._baseline, min_reach=self._reach)

    def _accept(self, tiles: np.ndarray, cells: int, objective: float) -> None:
        """Book an edit that stays: budget, error, gate baseline and reach."""
        self._spent += cells
        self._objective = objective
        self._reach = max(self._reach, self._gate.reach(tiles))
        if self._baseline:
            self._baseline &= self._gate.hard_rules(tiles)

    def _repair_pipes(self, tiles: np.ndarray, log: list[AppliedEdit]) -> None:
        """Remove seam-clipped pipes; kept whatever they do to the intent error.

        One the path climbs on cannot be removed and becomes plain ground.
        """
        for solid in (False, True):
            for edit in plan_repair_pipes(tiles, self._ctx, solid=solid):
                snapshot = apply_edit(tiles, edit)
                ok, reason = self._gate.check(tiles, baseline=self._baseline, min_reach=self._reach)
                if ok:
                    self._baseline = self._gate.hard_rules(tiles)
                else:
                    revert_edit(tiles, snapshot)
                log.append(AppliedEdit(edit.op, "repair", len(edit.cells), ok, reason))

    def _best_step(
        self,
        tiles: np.ndarray,
        target: np.ndarray,
        log: list[AppliedEdit],
        refused: set[tuple[str, Edit]],
    ) -> _Step | None:
        """The planned edit that removes the most overall error per tile.

        Every dimension that still needs a correction contributes its first
        planned edits; each is tried on the tiles, judged and undone. Edits
        the gates or the error refuse are logged once.
        """
        state = self._state(tiles)
        full = range(0, self._ctx.width)
        best: _Step | None = None
        best_rate = 0.0
        for dim in INTENT_KEYS:
            delta = float(target[_DIM[dim]] - state.vector[_DIM[dim]])
            if abs(delta) <= self._dim_tolerance(dim, target):
                continue
            direction = RAISE if delta > 0 else LOWER
            if not correction_needed(
                tiles, self._ctx, target, state, dim, direction, tolerance=self._tolerance
            ):
                continue
            planned = plan_correction(
                tiles,
                self._ctx,
                target,
                state,
                dim,
                direction,
                columns=full,
                tolerance=self._tolerance,
            )
            tried = 0
            for edit in planned:
                if not edit.cells or tried >= _CANDIDATES_PER_DIM:
                    continue
                fitted = self._fit(edit)
                reason: str | None = "over_budget"
                if fitted is not None:
                    tried += 1
                    snapshot = apply_edit(tiles, fitted)
                    ok, reason = self._gate_check(tiles, fitted)
                    objective = intent_objective(self._measure(tiles), target) if ok else math.inf
                    revert_edit(tiles, snapshot)
                    gain = self._objective - objective
                    if ok and gain > _EPS:
                        rate = gain / len(fitted.cells)
                        if rate > best_rate:
                            best, best_rate = _Step(fitted, dim, objective), rate
                        continue
                    if ok:
                        reason = "raises_error"
                key = (dim, fitted or edit)
                if key not in refused:
                    refused.add(key)
                    log.append(AppliedEdit(edit.op, dim, len(edit.cells), False, reason))
        return best

    def _literal_flat_cleanup(
        self, tiles: np.ndarray, target: np.ndarray, log: list[AppliedEdit]
    ) -> None:
        """Deliver the literal reading of a flat-and-smooth request.

        The achievable bands map any envelope amplitude below their 5th
        percentile to a measured zero, so sub-band mounds, tall pipes, and
        floating platforms are metrically invisible while remaining visible
        to a player; no step of the error-driven loop gains anything from
        removing them. This pass runs when the request itself is literal
        flat (verticality at most 0.05 with smoothness at least 0.9), after
        the loop so gaps are already filled: floating platforms over open
        pits are load-bearing for the flood gate and their removal would be
        rejected before that. It uses literal caps, not the banded
        amplitude; pipes and platforms survive only when their own targets
        explicitly ask for them. Each stage replans on the tiles the
        previous stage mutated and is judged as a whole: removing one of
        two stacked platforms can briefly raise the measured relief that
        removing both brings back to zero. It spends the same tile budget.
        """
        if not is_literal_flat(target):
            return
        ctx = self._ctx
        full = range(0, ctx.width)
        base = self._state(tiles).envelope_min
        amplitude = target_amplitude(target_value(target, "verticality"), self._config)
        pipe_cap = base if target_value(target, "pipe_density") <= 0.05 else base + 2
        platform_cap = base if target_value(target, "platform_usage") <= 0.05 else base + amplitude
        stages: tuple[Callable[[], list[Edit]], ...] = (
            lambda: plan_flatten(tiles, ctx, base_height=ctx.base_height, columns=full),
            lambda: plan_remove_pipes_above(tiles, ctx, cap=pipe_cap, columns=full),
            lambda: plan_remove_platforms(tiles, ctx, columns=full, above_height=platform_cap),
        )
        for stage in stages:
            before = (tiles.copy(), self._baseline, self._spent, self._reach)
            stage_log: list[AppliedEdit] = []
            for edit in stage():
                if not edit.cells:
                    continue
                if self._fit(edit) is not edit:
                    stage_log.append(
                        AppliedEdit(edit.op, "literal_flat", len(edit.cells), False, "over_budget")
                    )
                    continue
                snapshot = apply_edit(tiles, edit)
                ok, reason = self._gate.check(tiles, baseline=self._baseline, min_reach=self._reach)
                if ok:
                    self._accept(tiles, len(edit.cells), self._objective)
                else:
                    revert_edit(tiles, snapshot)
                stage_log.append(AppliedEdit(edit.op, "literal_flat", len(edit.cells), ok, reason))
            objective = intent_objective(self._measure(tiles), target)
            if objective > self._objective + _EPS:
                tiles[...] = before[0]
                _, self._baseline, self._spent, self._reach = before
                stage_log = [
                    AppliedEdit(e.op, e.dim, e.cell_count, False, e.reject_reason or "raises_error")
                    for e in stage_log
                ]
            else:
                self._objective = objective
            log.extend(stage_log)
