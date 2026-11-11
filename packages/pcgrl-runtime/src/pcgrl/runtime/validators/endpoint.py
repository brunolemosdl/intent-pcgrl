from __future__ import annotations

import numpy as np

from pcgrl.core.domain.contract import ValidationResult, Violation
from pcgrl.core.domain.level import LevelGrid
from pcgrl.core.settings.schema import RuntimeConfig
from pcgrl.core.tiles.catalog import TileCatalog
from pcgrl.runtime.layout import resolve_layout


class EndpointValidator:
    """Hard validator for exactly one player and one flagpole at layout coords."""

    def __init__(self, config: RuntimeConfig, catalog: TileCatalog) -> None:
        """Bind config and catalog.

        Args:
            config: Runtime configuration with layout spec.
            catalog: Tile catalog.
        """
        self._config = config
        self._catalog = catalog

    def validate(self, level: LevelGrid) -> ValidationResult:
        """Require player and flagpole at resolved layout positions.

        Args:
            level: Level to validate.

        Returns:
            Validation result with hard violations on mismatch.
        """
        violations: list[Violation] = []
        player_id = self._catalog.id_for_name("player")
        flagpole_id = self._catalog.id_for_name("flagpole")
        ys, xs = np.where(level.tiles == player_id)
        player_cells = list(zip(ys.tolist(), xs.tolist(), strict=False))
        flag_ys, flag_xs = np.where(level.tiles == flagpole_id)
        flagpole_cells = list(zip(flag_ys.tolist(), flag_xs.tolist(), strict=False))
        layout = resolve_layout(
            self._config.layout,
            height=level.height_tiles,
            width=level.width_tiles,
        )
        expected_player = layout.player
        expected_flag = layout.flagpole
        if len(player_cells) != 1 or player_cells[0] != expected_player:
            violations.append(
                Violation(
                    rule="endpoint.player",
                    severity="hard",
                    message=f"Expected one player at {expected_player}",
                    cells=tuple(player_cells),
                )
            )
        if len(flagpole_cells) != 1 or flagpole_cells[0] != expected_flag:
            violations.append(
                Violation(
                    rule="endpoint.flagpole",
                    severity="hard",
                    message=f"Expected one flagpole at {expected_flag}",
                    cells=tuple(flagpole_cells),
                )
            )
        penalty = float(len(violations))
        return ValidationResult(
            is_valid=not violations, violations=tuple(violations), penalty=penalty
        )
