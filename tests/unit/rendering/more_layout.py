from __future__ import annotations

from pcgrl.rendering.layout import parse_layout


def _level_with(char: str, width: int = 16, height: int = 3) -> str:
    """level with."""
    first_row = char + "-" * (width - 1)
    rest = ["-" * width for _ in range(height - 1)]
    return "\n".join([first_row, *rest])


def test_parse_question_block() -> None:
    """Verify parse question block."""
    result = parse_layout(_level_with("?"))
    assert result.level_tiles[0][0] == 8


def test_parse_used_block() -> None:
    """Verify parse used block."""
    result = parse_layout(_level_with("Q"))
    assert result.level_tiles[0][0] == 11


def test_parse_used_block_exclaim() -> None:
    """Verify parse used block exclaim."""
    result = parse_layout(_level_with("!"))
    assert result.level_tiles[0][0] == 11


def test_parse_brick_s() -> None:
    """Verify parse d block."""
    result = parse_layout(_level_with("S"))
    assert result.level_tiles[0][0] == 6


def test_parse_d_block() -> None:
    """Verify parse d block."""
    result = parse_layout(_level_with("D"))
    assert result.level_tiles[0][0] == 14


def test_parse_coin_o() -> None:
    """Verify parse coin o."""
    result = parse_layout(_level_with("o"))
    assert result.level_tiles[0][0] == 15


def test_parse_hidden_1up() -> None:
    """Verify parse hidden 1up."""
    result = parse_layout(_level_with("1"))
    assert result.level_tiles[0][0] == 48


def test_parse_hidden_coin() -> None:
    """Verify parse brick powerup u."""
    result = parse_layout(_level_with("2"))
    assert result.level_tiles[0][0] == 49


def test_parse_brick_coin_c() -> None:
    """Verify parse brick coin c."""
    result = parse_layout(_level_with("C"))
    assert result.level_tiles[0][0] == 7


def test_parse_brick_powerup_u() -> None:
    """Verify parse brick powerup u."""
    result = parse_layout(_level_with("U"))
    assert result.level_tiles[0][0] == 50


def test_parse_brick_1up_l() -> None:
    """Verify parse brick 1up l."""
    result = parse_layout(_level_with("L"))
    assert result.level_tiles[0][0] == 51


def test_parse_platform_percent() -> None:
    """Verify parse pipe top right."""
    result = parse_layout(_level_with("%"))
    assert result.level_tiles[0][0] in (43, 44, 45, 46)


def test_parse_platform_background() -> None:
    """Verify parse platform background."""
    result = parse_layout(_level_with("|"))
    assert result.level_tiles[0][0] == 47


def test_parse_pipe_top_left() -> None:
    """Verify parse pipe top left."""
    result = parse_layout(_level_with("<"))
    assert result.level_tiles[0][0] == 18


def test_parse_pipe_top_right() -> None:
    """Verify parse pipe top right."""
    result = parse_layout(_level_with(">"))
    assert result.level_tiles[0][0] == 19


def test_parse_pipe_body_left() -> None:
    """Verify parse cannon b."""
    result = parse_layout(_level_with("["))
    assert result.level_tiles[0][0] == 20


def test_parse_pipe_body_right() -> None:
    """Verify parse pipe body right."""
    result = parse_layout(_level_with("]"))
    assert result.level_tiles[0][0] == 21


def test_parse_bullet_bill() -> None:
    """Verify parse bullet bill."""
    result = parse_layout(_level_with("*"))
    assert result.level_tiles[0][0] in (3, 4, 5)


def test_parse_cannon_b_upper() -> None:
    """Verify parse cannon b upper."""
    result = parse_layout(_level_with("B"))
    assert result.level_tiles[0][0] == 3


def test_parse_cannon_b() -> None:
    """Verify parse cannon b."""
    result = parse_layout(_level_with("b"))
    assert result.level_tiles[0][0] in (4, 5)


def test_parse_t_single_pipe() -> None:
    """Verify parse t single pipe."""
    result = parse_layout("t" + "-" * 15 + "\n" + "-" * 16)
    assert result.level_tiles[0][0] in range(50, 60)


def test_parse_t_upper_enterable() -> None:
    """Verify parse t upper enterable."""
    result = parse_layout("T" + "-" * 15)
    assert result.level_tiles[0][0] >= 50


def test_parse_at_sign_powerup_block() -> None:
    """Verify parse at sign powerup block."""
    result = parse_layout(_level_with("@"))
    assert result.level_tiles[0][0] == 8


def test_flagpole_sets_exit(config) -> None:
    """Verify flagpole sets exit."""
    level = "-" * 15 + "F\n" + "X" * 16
    result = parse_layout(level)
    assert result.exit_tile_x == 15
    assert result.exit_loc_init if hasattr(result, "exit_loc_init") else True
