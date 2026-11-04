from __future__ import annotations

from collections import deque

import numpy as np

from pcgrl.core.domain.contract import ValidationResult, Violation
from pcgrl.core.domain.level import LevelGrid
from pcgrl.core.settings.schema import RuntimeConfig
from pcgrl.core.tiles.catalog import TileCatalog


class PipeValidator:
    """Hard validator for pipe cluster width, height, and grounding."""

    def __init__(self, config: RuntimeConfig, catalog: TileCatalog) -> None:
        """Bind config and catalog.

        Args:
            config: Runtime configuration with pipe geometry rules.
            catalog: Tile catalog.
        """
        self._config = config
        self._catalog = catalog
        self._pipe_names = set(config.categories.get("pipes", []))

    def validate(self, level: LevelGrid) -> ValidationResult:
        """Validate every connected pipe cluster in the level.

        Args:
            level: Level to validate.

        Returns:
            Validation result with per-cluster hard violations.
        """
        violations: list[Violation] = []
        pipe_ids = {
            self._catalog.id_for_name(name)
            for name in self._pipe_names
            if name in self._catalog.name_to_id
        }
        # One vectorised scan finds the (few) pipe cells; the Python BFS then
        # touches only those instead of walking the whole grid per call —
        # this validator runs once per gated edit in the edit phase.
        pipe_cells = np.argwhere(np.isin(level.tiles, list(pipe_ids)))
        if pipe_cells.size == 0:
            return ValidationResult(is_valid=True, violations=(), penalty=0.0)
        visited = np.zeros_like(level.tiles, dtype=bool)
        for y, x in pipe_cells.tolist():
            if visited[y, x]:
                continue
            cluster = self._collect_cluster(level, x, y, pipe_ids, visited)
            xs = [cell[1] for cell in cluster]
            ys = [cell[0] for cell in cluster]
            width = max(xs) - min(xs) + 1
            height = max(ys) - min(ys) + 1
            if self._config.validation.require_pipe_width_two and width != 2:
                violations.append(
                    Violation(
                        rule="pipe.width",
                        severity="hard",
                        message=f"Pipe cluster width {width} != 2",
                        cells=tuple(cluster),
                    )
                )
            if height < self._config.validation.min_pipe_height:
                violations.append(
                    Violation(
                        rule="pipe.height",
                        severity="hard",
                        message=f"Pipe height {height} below minimum",
                        cells=tuple(cluster),
                    )
                )
            if self._config.validation.require_grounded_pipes:
                bottom_y = max(ys)
                grounded = all(
                    self._is_ground(
                        level,
                        cx,
                        bottom_y + 1 if bottom_y + 1 < level.height_tiles else bottom_y,
                    )
                    for cx in range(min(xs), max(xs) + 1)
                )
                if not grounded:
                    violations.append(
                        Violation(
                            rule="pipe.grounded",
                            severity="hard",
                            message="Pipe cluster is not grounded",
                            cells=tuple(cluster),
                        )
                    )
        penalty = min(1.0, 0.5 * len(violations))
        return ValidationResult(
            is_valid=not violations, violations=tuple(violations), penalty=penalty
        )

    def _collect_cluster(
        self, level: LevelGrid, start_x: int, start_y: int, pipe_ids: set[int], visited: np.ndarray
    ) -> list[tuple[int, int]]:
        """BFS-collect orthogonally adjacent pipe tiles from a seed cell."""
        cluster: list[tuple[int, int]] = []
        queue: deque[tuple[int, int]] = deque([(start_y, start_x)])
        while queue:
            y, x = queue.popleft()
            if visited[y, x] or int(level.tiles[y, x]) not in pipe_ids:
                continue
            visited[y, x] = True
            cluster.append((y, x))
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                ny, nx = (y + dy, x + dx)
                if 0 <= ny < level.height_tiles and 0 <= nx < level.width_tiles:
                    queue.append((ny, nx))
        return cluster

    def _is_ground(self, level: LevelGrid, x: int, y: int) -> bool:
        """Return True when ``(y, x)`` is solid ground or below the grid."""
        if y >= level.height_tiles:
            return True
        tile = self._catalog.tile_for_id(int(level.tiles[y, x]))
        return tile.is_solid_top or tile.name == "solid"
