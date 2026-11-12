from __future__ import annotations

import numpy as np

from pcgrl.core.domain.contract import ValidationResult, Violation
from pcgrl.core.domain.level import LevelGrid
from pcgrl.core.settings.schema import RuntimeConfig
from pcgrl.core.tiles.catalog import TileCatalog


class EnemyValidator:
    """Validator for ground-enemy placement using SMB1 spawn semantics.

    Description:
        Corpus tiles mark SPAWN positions: the game drops a ground enemy
        onto the first solid top below its column, so the corpus (and the
        chunks extracted from it) legitimately contain enemies with empty
        cells directly underneath — 47 of 466 ground-enemy placements in the
        library land lower in the column. Requiring direct support would
        reject corpus-faithful compositions at random. An enemy is
        therefore valid when ANY solid top exists below it; one with no
        ground at all (falls out of the level) is a SOFT violation — also
        present in the raw corpus, penalised but never invalidating.
    """

    def __init__(self, config: RuntimeConfig, catalog: TileCatalog) -> None:
        """Bind config and catalog.

        Args:
            config: Runtime configuration.
            catalog: Tile catalog.
        """
        self._config = config
        self._catalog = catalog
        self._enemy_names = set(config.categories.get("enemies", []))

    def validate(self, level: LevelGrid) -> ValidationResult:
        """Check every non-flying enemy can land on solid ground below.

        Description:
            Fully vectorised — per-id LUTs replace per-cell catalog lookups
            and a bottom-up running maximum answers "any solid top below"
            for every cell at once; this validator runs once per gated edit
            in the environment's edit phase.

        Args:
            level: Level to validate.

        Returns:
            Validation result with per-cell soft violations for enemies
            whose column has no solid top anywhere below the spawn.
        """
        if self._config.validation.allow_floating_enemies:
            return ValidationResult(is_valid=True, violations=(), penalty=0.0)
        ground_ids = [
            self._catalog.id_for_name(name)
            for name in self._enemy_names
            if name in self._catalog.name_to_id and not self._catalog.is_flying_enemy(name)
        ]
        enemy_cells = np.argwhere(np.isin(level.tiles, ground_ids))
        if enemy_cells.size == 0:
            return ValidationResult(is_valid=True, violations=(), penalty=0.0)
        support = np.zeros(self._catalog.size, dtype=bool)
        for tile_id in range(self._catalog.size):
            support[tile_id] = self._catalog.tile_for_id(tile_id).is_solid_top
        solid = support[level.tiles]
        landable_below = np.zeros_like(solid)
        landable_below[:-1, :] = np.maximum.accumulate(solid[::-1, :], axis=0)[::-1][1:, :]
        violations = [
            self._violation("enemy.orphaned", y, x)
            for y, x in enemy_cells.tolist()
            if not landable_below[y, x]
        ]
        penalty = min(1.0, 0.25 * len(violations))
        return ValidationResult(is_valid=True, violations=tuple(violations), penalty=penalty)

    @staticmethod
    def _violation(rule: str, y: int, x: int) -> Violation:
        """Build a soft violation at ``(y, x)``."""
        return Violation(
            rule=rule,
            severity="soft",
            message=f"Enemy with no ground below at ({y}, {x})",
            cells=((y, x),),
        )
