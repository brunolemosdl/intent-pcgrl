from __future__ import annotations

import numpy as np
from pcgrl.core.domain.level import AreaType, LevelGrid
from pcgrl.core.tiles.catalog import TileCatalog
from pcgrl.runtime.validators.endpoint import EndpointValidator


def _make_level(h: int, w: int, tiles: np.ndarray) -> LevelGrid:
    """ make level."""
    return LevelGrid(
        tiles=tiles, width_tiles=w, height_tiles=h, tile_size_px=16, area_type=AreaType.OVERWORLD
    )


def test_fails_without_player(config, catalog: TileCatalog, blank_level) -> None:
    """Verify fails without player."""
    result = EndpointValidator(config, catalog).validate(blank_level)
    assert not result.is_valid


def test_fails_without_flagpole(config, catalog: TileCatalog, blank_level) -> None:
    """Verify fails without flagpole."""
    h, w = (config.game.height_tiles, config.game.width_tiles)
    tiles = blank_level.tiles.copy()
    tiles[h - 2, 1] = catalog.id_for_name("player")
    level = _make_level(h, w, tiles)
    result = EndpointValidator(config, catalog).validate(level)
    assert not result.is_valid
    assert any(v.rule == "endpoint.flagpole" for v in result.violations)


def test_passes_with_player_and_flagpole(config, catalog: TileCatalog, blank_level) -> None:
    """Verify passes with player and flagpole."""
    h, w = (config.game.height_tiles, config.game.width_tiles)
    tiles = blank_level.tiles.copy()
    tiles[h - 2, 1] = catalog.id_for_name("player")
    tiles[h - 2, w - 2] = catalog.id_for_name("flagpole")
    level = _make_level(h, w, tiles)
    result = EndpointValidator(config, catalog).validate(level)
    assert result.is_valid


def test_player_wrong_position_fails(config, catalog: TileCatalog, blank_level) -> None:
    """Verify player wrong position fails."""
    h, w = (config.game.height_tiles, config.game.width_tiles)
    tiles = blank_level.tiles.copy()
    tiles[0, 0] = catalog.id_for_name("player")
    tiles[h - 2, w - 2] = catalog.id_for_name("flagpole")
    level = _make_level(h, w, tiles)
    result = EndpointValidator(config, catalog).validate(level)
    assert not result.is_valid
    assert any(v.rule == "endpoint.player" for v in result.violations)


def test_penalty_reflects_violation_count(config, catalog: TileCatalog, blank_level) -> None:
    """Verify penalty reflects violation count."""
    result = EndpointValidator(config, catalog).validate(blank_level)
    assert result.penalty == float(len(result.violations))
