from __future__ import annotations

from dataclasses import dataclass

from pcgrl.core.settings.schema import FixedRegionConfig, LayoutConfig


@dataclass(frozen=True)
class ResolvedLayout:
    """Absolute grid coordinates after layout resolution.

    Attributes:
        player: Resolved player spawn ``(row, col)``.
        flagpole: Resolved flagpole ``(row, col)``.
        ground_row: Resolved ground fill row index.
        reserved: Cells occupied by fixed regions.
    """

    player: tuple[int, int]
    flagpole: tuple[int, int]
    ground_row: int
    reserved: frozenset[tuple[int, int]]


def resolve_row(row: int, *, height: int) -> int:
    """Resolve a row index, supporting negative bottom-relative values.

    Args:
        row: Configured row index.
        height: Grid height in tiles.

    Returns:
        Absolute row index in ``[0, height)``.
    """
    return height + row if row < 0 else row


def resolve_col(col: int, *, width: int) -> int:
    """Resolve a column index, supporting negative right-relative values.

    Args:
        col: Configured column index.
        width: Grid width in tiles.

    Returns:
        Absolute column index in ``[0, width)``.
    """
    return width + col if col < 0 else col


def resolve_coord(coord: tuple[int, int], *, height: int, width: int) -> tuple[int, int]:
    """Resolve a ``(row, col)`` pair to absolute grid coordinates.

    Args:
        coord: Configured coordinate pair.
        height: Grid height in tiles.
        width: Grid width in tiles.

    Returns:
        Absolute ``(row, col)`` indices.
    """
    row, col = coord
    return (resolve_row(row, height=height), resolve_col(col, width=width))


def region_cells(
    region: FixedRegionConfig,
    *,
    height: int,
    width: int,
) -> set[tuple[int, int]]:
    """Return all grid cells covered by a fixed region.

    Args:
        region: Fixed region origin and size from layout config.
        height: Grid height in tiles.
        width: Grid width in tiles.

    Returns:
        Set of absolute ``(row, col)`` cell coordinates.
    """
    row0, col0 = resolve_coord(region.origin, height=height, width=width)
    region_h, region_w = region.size
    return {(row0 + dy, col0 + dx) for dy in range(region_h) for dx in range(region_w)}


def resolve_layout(layout: LayoutConfig, *, height: int, width: int) -> ResolvedLayout:
    """Resolve all layout anchors and reserved cells for a grid size.

    Args:
        layout: Layout configuration with relative coordinates.
        height: Grid height in tiles.
        width: Grid width in tiles.

    Returns:
        Absolute spawn, flagpole, ground row, and reserved cells.
    """
    reserved: set[tuple[int, int]] = set()
    for region in layout.fixed_regions:
        reserved.update(region_cells(region, height=height, width=width))
    return ResolvedLayout(
        player=resolve_coord(layout.player, height=height, width=width),
        flagpole=resolve_coord(layout.flagpole, height=height, width=width),
        ground_row=resolve_row(layout.ground_row, height=height),
        reserved=frozenset(reserved),
    )
