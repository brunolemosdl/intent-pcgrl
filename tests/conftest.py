from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pcgrl.core.domain.level import AreaType, LevelGrid
from pcgrl.core.settings.loader import load_config
from pcgrl.core.settings.schema import RuntimeConfig
from pcgrl.core.tiles.catalog import TileCatalog, load_catalog


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def config() -> RuntimeConfig:
    """Default project config (composer environment)."""
    return load_config(training="initial", apply_profile=False)


@pytest.fixture(scope="session")
def catalog(config: RuntimeConfig) -> TileCatalog:
    """Tile catalog loaded from the default test config."""
    return load_catalog(config)


@pytest.fixture
def blank_level(config: RuntimeConfig, catalog: TileCatalog) -> LevelGrid:
    """Empty level grid matching default game dimensions."""
    h = config.game.height_tiles
    w = config.game.width_tiles
    tiles = np.full((h, w), catalog.id_for_name("empty"), dtype=np.int16)
    return LevelGrid(
        tiles=tiles,
        width_tiles=w,
        height_tiles=h,
        tile_size_px=config.game.tile_size_px,
        area_type=AreaType.OVERWORLD,
        metadata={},
    )
