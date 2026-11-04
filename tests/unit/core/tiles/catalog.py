from __future__ import annotations

import pytest
from pcgrl.core.domain.tile import TileCategory
from pcgrl.core.settings.loader import load_config
from pcgrl.core.tiles.catalog import TileCatalog


def test_catalog_has_corpus_tiles(catalog: TileCatalog) -> None:
    """Verify catalog has corpus tiles."""
    assert catalog.size >= 10


def test_id_for_name_goomba(catalog: TileCatalog) -> None:
    """Verify id for name goomba."""
    goomba_id = catalog.id_for_name("goomba")
    assert goomba_id >= 0


def test_glyph_for_id_solid(catalog: TileCatalog) -> None:
    """Verify glyph for id solid."""
    solid_id = catalog.id_for_name("solid")
    assert catalog.glyph_for_id(solid_id) == "X"


def test_glyph_for_id_empty(catalog: TileCatalog) -> None:
    """Verify glyph for id empty."""
    empty_id = catalog.id_for_name("empty")
    assert catalog.glyph_for_id(empty_id) == "-"


def test_alias_ground_resolves_to_solid(catalog: TileCatalog) -> None:
    """Verify alias ground resolves to solid."""
    assert catalog.id_for_name("ground") == catalog.id_for_name("solid")


def test_alias_tube_resolves_to_pipe_top_left(catalog: TileCatalog) -> None:
    """Verify alias tube resolves to pipe top left."""
    assert catalog.id_for_name("tube") == catalog.id_for_name("pipe_top_left")


def test_unknown_name_raises(catalog: TileCatalog) -> None:
    """Verify unknown name raises."""
    with pytest.raises(KeyError):
        catalog.id_for_name("nonexistent_tile_xyz")


def test_unknown_glyph_raises(catalog: TileCatalog) -> None:
    """Verify unknown glyph raises."""
    with pytest.raises(KeyError):
        catalog.id_for_glyph("~")


def test_id_for_glyph_unique_roundtrip(catalog: TileCatalog) -> None:
    """Verify id for glyph unique roundtrip."""
    used_block_id = catalog.id_for_name("used_block")
    glyph = catalog.glyph_for_id(used_block_id)
    registered_id = catalog.id_for_glyph(glyph)
    assert catalog.glyph_for_id(registered_id) == glyph


def test_name_for_id_roundtrip(catalog: TileCatalog) -> None:
    """Verify names in category returns tuple."""
    solid_id = catalog.id_for_name("solid")
    assert catalog.name_for_id(solid_id) == "solid"


def test_names_in_category_enemy_nonempty(catalog: TileCatalog) -> None:
    """Verify names in category enemy nonempty."""
    enemies = catalog.names_in_category(TileCategory.ENEMY)
    assert len(enemies) > 0
    assert "goomba" in enemies


def test_names_in_category_returns_tuple(catalog: TileCatalog) -> None:
    """Verify catalog tiles are tuple."""
    result = catalog.names_in_category(TileCategory.SOLID)
    assert isinstance(result, tuple)


def test_tile_for_id_returns_tile(catalog: TileCatalog) -> None:
    """Verify tile for id returns tile."""
    tile = catalog.tile_for_id(0)
    assert tile.tile_id == 0


def test_is_flying_enemy_goomba_false(catalog: TileCatalog) -> None:
    """Verify is flying enemy goomba false."""
    assert not catalog.is_flying_enemy("goomba")


def test_catalog_tiles_are_tuple(catalog: TileCatalog) -> None:
    """Verify catalog tiles are tuple."""
    assert isinstance(catalog.tiles, tuple)


def test_empty_tiles_raises() -> None:
    """Verify empty tiles raises."""
    cfg = load_config(training="initial")
    from pydantic import ValidationError

    with pytest.raises((ValueError, ValidationError)):
        from pcgrl.core.settings.schema import RuntimeConfig

        RuntimeConfig.model_validate({**cfg.model_dump(mode="python"), "tiles": {}})


def test_infer_category_question_block() -> None:
    """Verify infer category question block."""
    from pcgrl.core.tiles.catalog import _infer_category

    cat = _infer_category("question_block", {})
    from pcgrl.core.domain.tile import TileCategory

    assert cat == TileCategory.BLOCK


def test_infer_category_brick() -> None:
    """Verify infer category brick."""
    from pcgrl.core.domain.tile import TileCategory
    from pcgrl.core.tiles.catalog import _infer_category

    cat = _infer_category("brick_solid", {})
    assert cat == TileCategory.BLOCK


def test_is_passable_named_coin() -> None:
    """Verify load catalog duplicate name raises."""
    from pcgrl.core.domain.tile import TileCategory
    from pcgrl.core.tiles.catalog import _is_passable

    assert _is_passable("coin", TileCategory.TERRAIN)


def test_is_passable_returns_false_for_other() -> None:
    """Verify is passable returns false for other."""
    from pcgrl.core.domain.tile import TileCategory
    from pcgrl.core.tiles.catalog import _is_passable

    assert not _is_passable("solid", TileCategory.TERRAIN)


def test_is_solid_top_named_solid() -> None:
    """Verify is solid top named solid."""
    from pcgrl.core.domain.tile import TileCategory
    from pcgrl.core.tiles.catalog import _is_solid_top

    assert _is_solid_top("solid", TileCategory.TERRAIN)


def test_load_catalog_duplicate_name_raises(config) -> None:
    """Verify load catalog duplicate name raises."""
    from pcgrl.core.settings.schema import RuntimeConfig

    raw = config.model_dump(mode="python")
    tiles = dict(raw["tiles"])
    first_name, _first_glyph = next(iter(tiles.items()))
    tiles["_extra_key_z"] = "Z"
    tiles[first_name] = "Z"
    import pytest

    modified = {**raw, "tiles": {**raw["tiles"], "new_tile": "AB"}}
    with pytest.raises(ValueError, match="single character"):
        from pcgrl.core.tiles.catalog import load_catalog as lc

        modified_config = RuntimeConfig.model_validate(modified)
        lc(modified_config)


def test_load_catalog_bad_alias_raises(config) -> None:
    """Verify load catalog bad alias raises."""
    from pcgrl.core.settings.schema import RuntimeConfig

    raw = config.model_dump(mode="python")
    modified = {**raw, "aliases": {**raw.get("aliases", {}), "bad_alias": "nonexistent_tile_xyz"}}
    with pytest.raises(ValueError, match="unknown tile"):
        from pcgrl.core.tiles.catalog import load_catalog as lc

        modified_config = RuntimeConfig.model_validate(modified)
        lc(modified_config)
