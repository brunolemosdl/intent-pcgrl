from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class SpriteType(IntEnum):
    """Enemy sprite identifiers parsed from engine layout characters."""

    NONE = 0
    GOOMBA = 2
    GOOMBA_WINGED = 3
    RED_KOOPA = 4
    RED_KOOPA_WINGED = 5
    GREEN_KOOPA = 6
    GREEN_KOOPA_WINGED = 7
    SPIKY = 8
    SPIKY_WINGED = 9
    ENEMY_FLOWER = 11


ENEMY_SHEET_INDEX: dict[SpriteType, int] = {
    SpriteType.GOOMBA: 16,
    SpriteType.GOOMBA_WINGED: 16,
    SpriteType.RED_KOOPA: 0,
    SpriteType.RED_KOOPA_WINGED: 0,
    SpriteType.GREEN_KOOPA: 8,
    SpriteType.GREEN_KOOPA_WINGED: 8,
    SpriteType.SPIKY: 24,
    SpriteType.SPIKY_WINGED: 24,
    SpriteType.ENEMY_FLOWER: 48,
}


@dataclass(frozen=True)
class ParsedLayout:
    """Structured output of ``parse_layout`` for rendering and metrics.

    Attributes:
        level_tiles: ``[x][y]`` terrain tile indices (engine encoding).
        sprite_types: ``[x][y]`` enemy sprite enum per cell.
        mario_tile_x: Mario spawn column in tile coordinates.
        mario_tile_y: Mario spawn row in tile coordinates.
        exit_tile_x: Flagpole column in tile coordinates.
        exit_tile_y: Flagpole base row in tile coordinates.
        mario_present: Whether an explicit ``M`` spawn was found in the text.
    """

    level_tiles: list[list[int]]
    sprite_types: list[list[SpriteType]]
    mario_tile_x: int
    mario_tile_y: int
    exit_tile_x: int
    exit_tile_y: int
    mario_present: bool


def _is_solid(char: str) -> bool:
    """Return whether an engine character counts as solid ground."""
    return char in "X#@!BCQ<>[]?SUD%tT"


def _find_first_floor(lines: list[str], x: int) -> int:
    """Scan column ``x`` upward for the first air cell above solid terrain."""
    skip_lines = True
    for row in range(len(lines) - 1, -1, -1):
        char = lines[row][x]
        if _is_solid(char):
            skip_lines = False
            continue
        if not skip_lines and (not _is_solid(char)):
            return row
    return -1


def parse_layout(level: str) -> ParsedLayout:
    """Parse Mario engine ASCII into tile indices and sprite placements.

    Args:
        level: Newline-separated engine layout string (Mario-AI format).

    Returns:
        Parsed terrain grid, enemy sprites, and inferred spawn/exit coords.
    """
    lines = [line.rstrip("\r") for line in level.splitlines() if line]
    if not lines:
        return ParsedLayout(
            level_tiles=[[]],
            sprite_types=[[]],
            mario_tile_x=0,
            mario_tile_y=0,
            exit_tile_x=0,
            exit_tile_y=0,
            mario_present=False,
        )
    width = len(lines[0])
    height = len(lines)
    level_tiles = [[0 for _ in range(height)] for _ in range(width)]
    sprite_types = [[SpriteType.NONE for _ in range(height)] for _ in range(width)]
    mario_tile_x = 0
    mario_tile_y = 0
    exit_tile_x = width - 2
    exit_tile_y = 0
    mario_loc_init = False
    exit_loc_init = False
    for y, line in enumerate(lines):
        for x, char in enumerate(line):
            temp_index = 0
            single_pipe = False
            match char:
                case "M":
                    mario_tile_x = x
                    mario_tile_y = y
                    mario_loc_init = True
                case "F":
                    exit_tile_x = x
                    exit_tile_y = y
                    exit_loc_init = True
                case "y":
                    sprite_types[x][y] = SpriteType.SPIKY
                case "Y":
                    sprite_types[x][y] = SpriteType.SPIKY_WINGED
                case "E" | "g":
                    sprite_types[x][y] = SpriteType.GOOMBA
                case "G":
                    sprite_types[x][y] = SpriteType.GOOMBA_WINGED
                case "k":
                    sprite_types[x][y] = SpriteType.GREEN_KOOPA
                case "K":
                    sprite_types[x][y] = SpriteType.GREEN_KOOPA_WINGED
                case "r":
                    sprite_types[x][y] = SpriteType.RED_KOOPA
                case "R":
                    sprite_types[x][y] = SpriteType.RED_KOOPA_WINGED
                case "X":
                    level_tiles[x][y] = 1
                case "#":
                    level_tiles[x][y] = 2
                case "%":
                    temp_index = 0
                    if x > 0 and line[x - 1] == "%":
                        temp_index += 2
                    if x < width - 1 and line[x + 1] == "%":
                        temp_index += 1
                    level_tiles[x][y] = 43 + temp_index
                case "|":
                    level_tiles[x][y] = 47
                case "*":
                    temp_index = 0
                    if y > 0 and lines[y - 1][x] == "*":
                        temp_index += 1
                    if y > 1 and lines[y - 2][x] == "*":
                        temp_index += 1
                    level_tiles[x][y] = 3 + temp_index
                case "B":
                    level_tiles[x][y] = 3
                case "b":
                    temp_index = 0
                    if y > 1 and lines[y - 2][x] == "B":
                        temp_index += 1
                    level_tiles[x][y] = 4 + temp_index
                case "?" | "@":
                    level_tiles[x][y] = 8
                case "Q" | "!":
                    level_tiles[x][y] = 11
                case "1":
                    level_tiles[x][y] = 48
                case "2":
                    level_tiles[x][y] = 49
                case "D":
                    level_tiles[x][y] = 14
                case "S":
                    level_tiles[x][y] = 6
                case "C":
                    level_tiles[x][y] = 7
                case "U":
                    level_tiles[x][y] = 50
                case "L":
                    level_tiles[x][y] = 51
                case "o":
                    level_tiles[x][y] = 15
                case "t":
                    single_pipe = (x >= width - 1 or lines[y][x + 1].lower() != "t") and (
                        x == 0 or lines[y][x - 1].lower() != "t"
                    )
                    temp_index = 0
                    if x > 0 and level_tiles[x - 1][y] in (18, 20):
                        temp_index += 1
                    if y > 0 and lines[y - 1][x].lower() == "t":
                        temp_index += 2 if not single_pipe else 1
                    level_tiles[x][y] = 52 + temp_index if single_pipe else 18 + temp_index
                case "T":
                    single_pipe = (x >= width - 1 or lines[y][x + 1].lower() != "t") and (
                        x == 0 or lines[y][x - 1].lower() != "t"
                    )
                    temp_index = 0
                    if x > 0 and level_tiles[x - 1][y] in (18, 20):
                        temp_index += 1
                    if y > 0 and lines[y - 1][x].lower() == "t":
                        temp_index += 2 if not single_pipe else 1
                    if not single_pipe and temp_index == 0:
                        sprite_types[x][y] = SpriteType.ENEMY_FLOWER
                    level_tiles[x][y] = 52 + temp_index if single_pipe else 18 + temp_index
                case "<":
                    level_tiles[x][y] = 18
                case ">":
                    level_tiles[x][y] = 19
                case "[":
                    level_tiles[x][y] = 20
                case "]":
                    level_tiles[x][y] = 21
    if not mario_loc_init:
        mario_tile_x = 0
        mario_tile_y = _find_first_floor(lines, mario_tile_x)
    if not exit_loc_init:
        exit_tile_x = width - 2
        exit_tile_y = _find_first_floor(lines, exit_tile_x)
    if exit_tile_y >= 0:
        pole_top = max(0, exit_tile_y - 11)
        for y in range(exit_tile_y, pole_top, -1):
            if 0 <= y < height:
                level_tiles[exit_tile_x][y] = 40
        if 0 <= pole_top < height:
            level_tiles[exit_tile_x][pole_top] = 39
    return ParsedLayout(
        level_tiles=level_tiles,
        sprite_types=sprite_types,
        mario_tile_x=mario_tile_x,
        mario_tile_y=mario_tile_y,
        exit_tile_x=exit_tile_x,
        exit_tile_y=exit_tile_y,
        mario_present=mario_loc_init,
    )
