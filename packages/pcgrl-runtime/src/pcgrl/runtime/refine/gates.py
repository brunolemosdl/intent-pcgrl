from __future__ import annotations

import numpy as np

from pcgrl.core.domain.level import AreaType, LevelGrid
from pcgrl.core.settings.schema import RuntimeConfig
from pcgrl.core.tiles.catalog import TileCatalog
from pcgrl.runtime.validators.reachability import reachable_cells
from pcgrl.runtime.validators.runner import ValidatorRunner


def grid_from_tiles(tiles: np.ndarray, config: RuntimeConfig) -> LevelGrid:
    """Wrap a working tile array in a ``LevelGrid`` for validators/metrics.

    Args:
        tiles: Tile-id array of shape ``(height, width)``.
        config: Runtime configuration for the grid dimensions.

    Returns:
        Frozen ``LevelGrid`` sharing no state with the working array.
    """
    height, width = tiles.shape
    return LevelGrid(
        tiles=tiles.copy(),
        width_tiles=width,
        height_tiles=height,
        tile_size_px=config.game.tile_size_px,
        area_type=AreaType.OVERWORLD,
    )


def flood_reaches_flag(
    tiles: np.ndarray,
    catalog: TileCatalog,
    config: RuntimeConfig,
    *,
    anchors: tuple[tuple[int, int], int] | None = None,
) -> bool:
    """Check the reward's playability win condition on a tile array.

    Description:
        Mirrors ``ComposerTerminalReward._playability``: the flood from the
        player spawn must reach a standable column at or beyond the flagpole
        column. Edits that sever this path must be reverted.

    Args:
        tiles: Tile-id array of shape ``(height, width)``.
        catalog: Tile catalog.
        config: Runtime configuration (jump physics, layout anchors).
        anchors: Optional precomputed ``(player, flag_column)`` pair so
            per-edit callers skip re-resolving the layout.

    Returns:
        ``True`` when the flag column stays reachable.
    """
    if anchors is None:
        from pcgrl.runtime.layout import resolve_layout

        height, width = tiles.shape
        resolved = resolve_layout(config.layout, height=height, width=width)
        anchors = (resolved.player, resolved.flagpole[1])
    start, flag_column = anchors
    reach = reachable_cells(
        tiles,
        catalog,
        jump_height=config.validation.jump_height,
        jump_reach=config.validation.jump_reach,
        start=start,
    )
    return any(column >= flag_column for _, column in reach.standable)


class RefineGate:
    """Structural + playability gate applied after every atomic edit.

    Description:
        Runs the shared ``ValidatorRunner`` chain (short-circuiting on the
        first hard violation) followed by the spawn-to-flag flood. An edit
        that fails either check is reverted by the caller, so a refined level
        can never regress below the gates the unedited level already passed.
    """

    def __init__(self, config: RuntimeConfig, catalog: TileCatalog) -> None:
        """Bind the validator chain, physics, and resolved layout anchors.

        Args:
            config: Runtime configuration.
            catalog: Tile catalog.
        """
        from pcgrl.runtime.layout import resolve_layout

        self._config = config
        self._catalog = catalog
        self._runner = ValidatorRunner(config, catalog)
        resolved = resolve_layout(
            config.layout,
            height=config.game.height_tiles,
            width=config.game.width_tiles,
        )
        self._anchors = (resolved.player, resolved.flagpole[1])

    def hard_rules(self, tiles: np.ndarray) -> frozenset[str]:
        """Names of the hard rules a tile array currently violates."""
        level = grid_from_tiles(tiles, self._config)
        result = self._runner.validate(level)
        return frozenset(
            violation.rule for violation in result.violations if violation.severity == "hard"
        )

    def check(
        self, tiles: np.ndarray, *, baseline: frozenset[str] = frozenset()
    ) -> tuple[bool, str | None]:
        """Validate one working tile array against a violation baseline.

        Description:
            Composed levels can arrive with pre-existing hard violations
            (e.g. seam-merged pipes); an edit is acceptable when it
            introduces no NEW hard rule beyond the baseline and keeps the
            flag reachable, so refinement never makes a level worse and is
            never paralysed by damage it did not cause.

        Args:
            tiles: Tile-id array after applying a candidate edit.
            baseline: Hard rules already violated before the edit.

        Returns:
            ``(ok, reason)`` where ``reason`` names the first failure
            (``"structural:<rule>"`` or ``"flood_unreachable"``).
        """
        level = grid_from_tiles(tiles, self._config)
        result = self._runner.validate(level)
        for violation in result.violations:
            if violation.severity == "hard" and violation.rule not in baseline:
                return False, f"structural:{violation.rule}"
        if not flood_reaches_flag(tiles, self._catalog, self._config, anchors=self._anchors):
            return False, "flood_unreachable"
        return True, None
