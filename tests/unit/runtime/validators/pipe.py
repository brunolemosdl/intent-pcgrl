from __future__ import annotations

import numpy as np
from pcgrl.core.domain.level import AreaType, LevelGrid
from pcgrl.core.tiles.catalog import TileCatalog
from pcgrl.runtime.validators.pipe import PipeValidator


def _stamp_pipe(tiles: np.ndarray, y: int, x: int, catalog: TileCatalog) -> None:
    """ stamp pipe."""
    tiles[y, x] = catalog.id_for_name("pipe_top_left")
    tiles[y, x + 1] = catalog.id_for_name("pipe_top_right")
    tiles[y + 1, x] = catalog.id_for_name("pipe_body_left")
    tiles[y + 1, x + 1] = catalog.id_for_name("pipe_body_right")


def test_no_pipes_passes(config, catalog, blank_level) -> None:
    """Verify no pipes passes."""
    result = PipeValidator(config, catalog).validate(blank_level)
    assert result.is_valid
    assert result.penalty == 0.0


def test_valid_grounded_pipe_passes(config, catalog) -> None:
    """Verify valid grounded pipe passes."""
    h, w = (config.game.height_tiles, config.game.width_tiles)
    solid_id = catalog.id_for_name("solid")
    empty_id = catalog.id_for_name("empty")
    tiles = np.full((h, w), empty_id, dtype=np.int16)
    tiles[-1, :] = solid_id
    _stamp_pipe(tiles, h - 3, 4, catalog)
    level = LevelGrid(
        tiles=tiles, width_tiles=w, height_tiles=h, tile_size_px=16, area_type=AreaType.OVERWORLD
    )
    result = PipeValidator(config, catalog).validate(level)
    assert result.is_valid


def test_pipe_too_short_fails(config, catalog) -> None:
    """Verify pipe too short fails."""
    h, w = (config.game.height_tiles, config.game.width_tiles)
    solid_id = catalog.id_for_name("solid")
    pipe_id = catalog.id_for_name("pipe_top_left")
    empty_id = catalog.id_for_name("empty")
    tiles = np.full((h, w), empty_id, dtype=np.int16)
    tiles[-1, :] = solid_id
    tiles[-2, 4] = pipe_id
    tiles[-2, 5] = catalog.id_for_name("pipe_top_right")
    level = LevelGrid(
        tiles=tiles, width_tiles=w, height_tiles=h, tile_size_px=16, area_type=AreaType.OVERWORLD
    )
    result = PipeValidator(config, catalog).validate(level)
    assert not result.is_valid
    assert any(v.rule == "pipe.height" for v in result.violations)


def test_pipe_wrong_width_fails(config, catalog) -> None:
    """Verify pipe wrong width fails."""
    h, w = (config.game.height_tiles, config.game.width_tiles)
    solid_id = catalog.id_for_name("solid")
    pipe_id = catalog.id_for_name("pipe_top_left")
    empty_id = catalog.id_for_name("empty")
    tiles = np.full((h, w), empty_id, dtype=np.int16)
    tiles[-1, :] = solid_id
    for row in range(-4, -1):
        tiles[row, 4] = pipe_id
    level = LevelGrid(
        tiles=tiles, width_tiles=w, height_tiles=h, tile_size_px=16, area_type=AreaType.OVERWORLD
    )
    result = PipeValidator(config, catalog).validate(level)
    assert any(v.rule == "pipe.width" for v in result.violations)


def test_floating_pipe_fails(config, catalog) -> None:
    """Verify floating pipe fails."""
    h, w = (config.game.height_tiles, config.game.width_tiles)
    empty_id = catalog.id_for_name("empty")
    tiles = np.full((h, w), empty_id, dtype=np.int16)
    _stamp_pipe(tiles, 2, 4, catalog)
    level = LevelGrid(
        tiles=tiles, width_tiles=w, height_tiles=h, tile_size_px=16, area_type=AreaType.OVERWORLD
    )
    result = PipeValidator(config, catalog).validate(level)
    assert any(v.rule == "pipe.grounded" for v in result.violations)


def test_pipe_at_bottom_row_is_grounded(config, catalog) -> None:
    """Verify pipe at bottom row is grounded."""
    h, w = (config.game.height_tiles, config.game.width_tiles)
    empty_id = catalog.id_for_name("empty")
    tiles = np.full((h, w), empty_id, dtype=np.int16)
    _stamp_pipe(tiles, h - 2, 4, catalog)
    level = LevelGrid(
        tiles=tiles, width_tiles=w, height_tiles=h, tile_size_px=16, area_type=AreaType.OVERWORLD
    )
    result = PipeValidator(config, catalog).validate(level)
    assert not any(v.rule == "pipe.grounded" for v in result.violations)
