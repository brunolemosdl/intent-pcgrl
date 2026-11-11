from __future__ import annotations

from pcgrl.core.domain.contract import ValidationResult, Violation
from pcgrl.core.domain.level import LevelGrid
from pcgrl.core.settings.schema import RuntimeConfig


class GeometryValidator:
    """Hard validator for level grid shape vs configured dimensions."""

    def __init__(self, config: RuntimeConfig) -> None:
        """Store expected height and width from config.

        Args:
            config: Runtime configuration.
        """
        self._config = config

    def validate(self, level: LevelGrid) -> ValidationResult:
        """Require ``level.tiles`` shape to match config game dimensions.

        Args:
            level: Level to validate.

        Returns:
            Validation result with a hard violation on shape mismatch.
        """
        expected = (self._config.game.height_tiles, self._config.game.width_tiles)
        if level.tiles.shape != expected:
            violation = Violation(
                rule="geometry.shape",
                severity="hard",
                message=f"Expected shape {expected}, got {level.tiles.shape}",
                cells=(),
            )
            return ValidationResult(is_valid=False, violations=(violation,), penalty=1.0)
        return ValidationResult(is_valid=True, violations=(), penalty=0.0)
