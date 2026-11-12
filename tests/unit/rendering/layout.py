from __future__ import annotations

from pcgrl.rendering.layout import ParsedLayout, SpriteType, parse_layout


def _simple_level() -> str:
    """simple level."""
    return "-" * 16 + "\n" + "-" * 15 + "F" + "\n" + "X" * 16


def test_parse_layout_returns_parsed_layout() -> None:
    """Verify parse layout returns parsed layout."""
    result = parse_layout(_simple_level())
    assert isinstance(result, ParsedLayout)


def test_parse_layout_empty_string_returns_default() -> None:
    """Verify parse layout empty string returns default."""
    result = parse_layout("")
    assert result.level_tiles == [[]]
    assert not result.mario_present


def test_parse_layout_mario_position() -> None:
    """Verify parse layout mario position."""
    level = "-" * 7 + "M" + "-" * 8 + "\n" + "X" * 16
    result = parse_layout(level)
    assert result.mario_tile_x == 7
    assert result.mario_present


def test_parse_layout_exit_position() -> None:
    """Verify parse layout exit position."""
    level = "-" * 15 + "F\n" + "X" * 16
    result = parse_layout(level)
    assert result.exit_tile_x == 15


def test_parse_layout_goomba_sprite() -> None:
    """Verify parse layout solid tile."""
    level = "-" * 5 + "g" + "-" * 10 + "\n" + "X" * 16
    result = parse_layout(level)
    assert result.sprite_types[5][0] == SpriteType.GOOMBA


def test_parse_layout_solid_tile() -> None:
    """Verify parse layout coin tile."""
    level = "X" * 16
    result = parse_layout(level)
    assert result.level_tiles[0][0] == 1


def test_parse_layout_coin_tile() -> None:
    """Verify parse layout coin tile."""
    level = "o" + "-" * 15
    result = parse_layout(level)
    assert result.level_tiles[0][0] == 15


def test_parse_enemy_sprite_types() -> None:
    """Verify parse enemy sprite types."""
    enemies = [
        ("y", SpriteType.SPIKY),
        ("Y", SpriteType.SPIKY_WINGED),
        ("G", SpriteType.GOOMBA_WINGED),
        ("k", SpriteType.GREEN_KOOPA),
        ("K", SpriteType.GREEN_KOOPA_WINGED),
        ("r", SpriteType.RED_KOOPA),
        ("R", SpriteType.RED_KOOPA_WINGED),
    ]
    for char, expected in enemies:
        level = char + "-" * 15
        result = parse_layout(level)
        assert result.sprite_types[0][0] == expected, f"Failed for char={char!r}"


def test_parse_layout_no_mario_present() -> None:
    """Verify parse layout no mario present."""
    level = "-" * 16 + "\n" + "X" * 16
    result = parse_layout(level)
    assert not result.mario_present


def test_parse_layout_various_tile_chars() -> None:
    """Verify parse layout bridge single."""
    chars_to_tiles = [
        ("#", 2),
        ("|", 47),
        ("?", 8),
        ("@", 8),
        ("Q", 11),
        ("!", 11),
        ("1", 48),
        ("2", 49),
        ("D", 14),
        ("S", 6),
        ("C", 7),
        ("U", 50),
        ("L", 51),
    ]
    for char, expected_tile in chars_to_tiles:
        level = char + "-" * 15
        result = parse_layout(level)
        assert result.level_tiles[0][0] == expected_tile, f"char={char!r}"


def test_parse_layout_bridge_tiles() -> None:
    """Verify parse layout bullet bill stack."""
    level = "%%%-" * 4
    result = parse_layout(level)
    assert result.level_tiles[1][0] == 46


def test_parse_layout_bridge_single() -> None:
    """Verify parse layout background b above big b."""
    level = "-%-" + "-" * 13
    result = parse_layout(level)
    assert result.level_tiles[1][0] == 43


def test_parse_layout_bullet_bill_stack() -> None:
    """Verify parse layout wide pipe adjacent."""
    level = "*" + "-" * 15 + "\n" + "*" + "-" * 15 + "\n" + "*" + "-" * 15 + "\n" + "X" * 16
    result = parse_layout(level)
    assert result.level_tiles[0][0] == 3
    assert result.level_tiles[0][1] == 4
    assert result.level_tiles[0][2] == 5


def test_parse_layout_background_b_above_big_b() -> None:
    """Verify parse layout background b above big b."""
    level = "-" * 16 + "\n" + "B" + "-" * 15 + "\n" + "-" * 16 + "\n" + "b" + "-" * 15
    result = parse_layout(level)
    assert result.level_tiles[0][3] == 5


def test_parse_layout_wide_pipe_adjacent() -> None:
    """Verify parse layout wide pipe adjacent."""
    level = "<>" + "-" * 14 + "\n" + "[]" + "-" * 14
    result = parse_layout(level)
    assert result.level_tiles[0][0] == 18
    assert result.level_tiles[1][0] == 19


def test_parse_layout_pipe_t_with_prior_left() -> None:
    """Verify parse layout big t above big t double."""
    level = "tt" + "-" * 14 + "\n" + "tt" + "-" * 14
    result = parse_layout(level)
    assert result.level_tiles[0][0] == 18
    assert result.level_tiles[1][0] == 19


def test_parse_layout_pipe_big_t_with_flower() -> None:
    """Verify parse layout pipe big t with flower."""
    level = "T" + "-" * 15
    result = parse_layout(level)
    assert result.level_tiles[0][0] == 52


def test_parse_layout_pipe_big_t_double_flower() -> None:
    """Verify parse layout pipe big t double flower."""
    level = "TT" + "-" * 14 + "\n" + "tt" + "-" * 14
    result = parse_layout(level)
    assert result.sprite_types[0][0] == SpriteType.ENEMY_FLOWER


def test_parse_layout_big_t_above_big_t_double() -> None:
    """Verify parse layout big t above big t double."""
    level = "TT" + "-" * 14 + "\n" + "TT" + "-" * 14
    result = parse_layout(level)
    assert result.level_tiles[0][1] == 20
