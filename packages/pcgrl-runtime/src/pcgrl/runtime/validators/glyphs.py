from __future__ import annotations

import numpy as np

from pcgrl.core.domain.contract import ValidationResult, Violation
from pcgrl.core.domain.level import LevelGrid
from pcgrl.core.tiles.catalog import TileCatalog


class GlyphsValidator:
    """Hard validator ensuring every tile id is within the catalog range."""

    def __init__(self, catalog: TileCatalog) -> None:
        """Bind the tile catalog.

        Args:
            catalog: Tile catalog defining valid id range.
        """
        self._catalog = catalog

    def validate(self, level: LevelGrid) -> ValidationResult:
        """Scan the grid for out-of-range tile ids.

        Args:
            level: Level to validate.

        Returns:
            Validation result with per-cell hard violations.
        """
        max_id = self._catalog.size - 1
        unknown = np.argwhere((level.tiles < 0) | (level.tiles > max_id))
        violations = [
            Violation(
                rule="glyphs.unknown_id",
                severity="hard",
                message=f"Unknown tile id {int(level.tiles[y, x])}",
                cells=((int(y), int(x)),),
            )
            for y, x in unknown
        ]
        penalty = 1.0 if violations else 0.0
        return ValidationResult(
            is_valid=not violations, violations=tuple(violations), penalty=penalty
        )
