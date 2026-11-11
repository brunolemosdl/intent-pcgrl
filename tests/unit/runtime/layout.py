from __future__ import annotations

from pcgrl.core.layout import resolve_coord, resolve_layout, resolve_row
from pcgrl.core.settings.schema import LayoutConfig


def test_resolve_layout_default_positions() -> None:
    """Verify resolve layout default positions."""
    layout = LayoutConfig()
    resolved = resolve_layout(layout, height=16, width=152)
    assert resolved.player == (14, 1)
    assert resolved.flagpole == (14, 150)
    assert resolved.ground_row == 15
    assert (14, 0) in resolved.reserved
    assert (14, 2) in resolved.reserved
    assert (15, 0) in resolved.reserved
    assert (14, 149) in resolved.reserved
    assert (14, 151) in resolved.reserved
    assert (14, 10) not in resolved.reserved


def test_negative_coord_resolution() -> None:
    """Verify negative coord resolution."""
    assert resolve_row(-1, height=16) == 15
    assert resolve_coord((-2, -2), height=16, width=152) == (14, 150)


def test_layout_scales_with_width() -> None:
    """Verify layout scales with width."""
    layout = LayoutConfig()
    narrow = resolve_layout(layout, height=16, width=80)
    wide = resolve_layout(layout, height=16, width=200)
    assert narrow.player == wide.player == (14, 1)
    assert narrow.flagpole == (14, 78)
    assert wide.flagpole == (14, 198)
    assert (14, 0) in narrow.reserved and (15, 2) in narrow.reserved
    assert (14, 77) in narrow.reserved and (15, 79) in narrow.reserved
    assert (14, 0) in wide.reserved and (15, 2) in wide.reserved
    assert (14, 197) in wide.reserved and (15, 199) in wide.reserved


def test_layout_yaml_pairs(config) -> None:
    """Verify layout yaml pairs."""
    assert config.layout.player == (-2, 1)
    assert config.layout.flagpole == (-2, -2)
    assert config.layout.fixed_regions[0].origin == (-2, 0)
    assert config.layout.fixed_regions[1].origin == (-2, -3)
    resolved = resolve_layout(
        config.layout, height=config.game.height_tiles, width=config.game.width_tiles
    )
    assert resolved.player == (config.game.height_tiles - 2, 1)
    assert resolved.flagpole == (config.game.height_tiles - 2, config.game.width_tiles - 2)
