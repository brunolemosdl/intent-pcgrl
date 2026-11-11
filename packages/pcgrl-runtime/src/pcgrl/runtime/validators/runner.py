from __future__ import annotations

from pcgrl.core.domain.contract import ValidationResult, Violation
from pcgrl.core.domain.level import LevelGrid
from pcgrl.core.settings.schema import RuntimeConfig
from pcgrl.core.tiles.catalog import TileCatalog
from pcgrl.runtime.validators.density import DensityValidator
from pcgrl.runtime.validators.endpoint import EndpointValidator
from pcgrl.runtime.validators.enemy import EnemyValidator
from pcgrl.runtime.validators.geometry import GeometryValidator
from pcgrl.runtime.validators.glyphs import GlyphsValidator
from pcgrl.runtime.validators.pipe import PipeValidator


class ValidatorRunner:
    """Run the ordered chain of structural validators on a level."""

    def __init__(self, config: RuntimeConfig, catalog: TileCatalog) -> None:
        """Construct the default validator pipeline.

        Args:
            config: Runtime configuration.
            catalog: Tile catalog.
        """
        self._validators: list[
            GeometryValidator
            | GlyphsValidator
            | EndpointValidator
            | PipeValidator
            | EnemyValidator
            | DensityValidator
        ] = [
            GeometryValidator(config),
            GlyphsValidator(catalog),
            EndpointValidator(config, catalog),
            PipeValidator(config, catalog),
            EnemyValidator(config, catalog),
            DensityValidator(config, catalog),
        ]

    def validate(self, level: LevelGrid, *, short_circuit_hard: bool = False) -> ValidationResult:
        """Run all validators and aggregate violations and penalties.

        Args:
            level: Level to validate.
            short_circuit_hard: Stop after the first hard violation when True.

        Returns:
            Combined validation result; ``is_valid`` is False when any hard
            violation is present.
        """
        violations: list[Violation] = []
        penalty = 0.0
        hard_found = False
        for validator in self._validators:
            result = validator.validate(level)
            violations.extend(result.violations)
            penalty += result.penalty
            if any(v.severity == "hard" for v in result.violations):
                hard_found = True
                if short_circuit_hard:
                    break
        return ValidationResult(
            is_valid=not hard_found, violations=tuple(violations), penalty=penalty
        )
