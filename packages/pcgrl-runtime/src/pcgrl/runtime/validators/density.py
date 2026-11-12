from __future__ import annotations

from typing import Literal

import numpy as np

from pcgrl.core.domain.contract import ValidationResult, Violation
from pcgrl.core.domain.level import LevelGrid
from pcgrl.core.settings.schema import RuntimeConfig
from pcgrl.core.tiles.catalog import TileCatalog


class DensityValidator:
    """Soft/hard validator for overall solid-tile density bounds."""

    def __init__(self, config: RuntimeConfig, catalog: TileCatalog) -> None:
        """Bind config and catalog.

        Args:
            config: Runtime configuration with min/max solid ratios.
            catalog: Tile catalog.
        """
        self._config = config
        self._catalog = catalog
        self._solid_ids = [
            catalog.id_for_name(name)
            for name in config.categories.get("solid_like", [])
            if name in catalog.name_to_id
        ]

    def validate(self, level: LevelGrid) -> ValidationResult:
        """Measure solid ratio and emit violations when outside bounds.

        Args:
            level: Level to validate.

        Returns:
            Validation result; over-max is hard, under-min is soft.
        """
        total = level.width_tiles * level.height_tiles
        solid_count = int(np.isin(level.tiles, self._solid_ids).sum())
        ratio = solid_count / max(1, total)
        min_ratio = self._config.validation.min_solid_ratio
        max_ratio = self._config.validation.max_solid_ratio
        if min_ratio <= ratio <= max_ratio:
            return ValidationResult(is_valid=True, violations=(), penalty=0.0)
        distance = min(abs(ratio - min_ratio), abs(ratio - max_ratio))
        severity: Literal["hard", "soft"] = "hard" if ratio > max_ratio else "soft"
        violation = Violation(
            rule="density.solid_ratio",
            severity=severity,
            message=f"Solid ratio {ratio:.3f} outside [{min_ratio}, {max_ratio}]",
            cells=(),
        )
        return ValidationResult(
            is_valid=True, violations=(violation,), penalty=min(1.0, distance * 2.0)
        )
