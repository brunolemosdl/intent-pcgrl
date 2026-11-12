from __future__ import annotations

from dataclasses import dataclass

from pcgrl.core.settings.schema import RuntimeConfig
from pcgrl.core.tiles.catalog import TileCatalog
from pcgrl.runtime.layout import resolve_layout
from pcgrl.runtime.metrics.intent import GAP_BUDGET_PER_COLUMN
from pcgrl.runtime.metrics.usefulness import block_names, solid_like_ids


@dataclass(frozen=True)
class RefineContext:
    """Precomputed ids, layout anchors, and bands shared by every edit op.

    Description:
        Built once per (config, catalog) pair; planners are pure functions over
        a tile array plus this context, so the environment edit phase and the
        deterministic polish controller share identical edit semantics.

    Attributes:
        config: Runtime configuration (bands, physics, layout).
        catalog: Tile catalog for id lookups.
        width: Level width in tiles.
        height: Level height in tiles.
        area: ``width * height`` used by the density formulas.
        empty_id: Id written when material is removed.
        solid_id: Ground tile id used for fills, hills, and platforms.
        coin_id: Coin tile id.
        enemy_ids: Ground (non-flying) enemy tile ids.
        block_ids: Interactive-block tile ids (the block-density metric set).
        pipe_ids: All pipe tile ids.
        pipe_top_left: Pipe top-left id.
        pipe_top_right: Pipe top-right id.
        pipe_body_left: Pipe body-left id.
        pipe_body_right: Pipe body-right id.
        solid_like: Tile ids the ground-height profile treats as solid.
        player: Resolved player spawn ``(row, col)``.
        flagpole: Resolved flagpole ``(row, col)``.
        protected: Cells edits must never write (endpoints + reserved regions).
        protected_columns: Columns containing protected cells; terrain ops skip
            them entirely so the endpoint platforms stay intact.
        base_height: Rows of plain ground the layout stands on (the corpus
            floor), where a literally flat request settles.
    """

    config: RuntimeConfig
    catalog: TileCatalog
    width: int
    height: int
    area: int
    empty_id: int
    solid_id: int
    coin_id: int
    enemy_ids: tuple[int, ...]
    block_ids: tuple[int, ...]
    pipe_ids: frozenset[int]
    pipe_top_left: int
    pipe_top_right: int
    pipe_body_left: int
    pipe_body_right: int
    solid_like: frozenset[int]
    player: tuple[int, int]
    flagpole: tuple[int, int]
    protected: frozenset[tuple[int, int]]
    protected_columns: frozenset[int]
    base_height: int


def build_context(config: RuntimeConfig, catalog: TileCatalog) -> RefineContext:
    """Resolve every id and layout anchor the edit planners need.

    Args:
        config: Runtime configuration.
        catalog: Tile catalog.

    Returns:
        Frozen ``RefineContext``.
    """
    height = config.game.height_tiles
    width = config.game.width_tiles
    resolved = resolve_layout(config.layout, height=height, width=width)
    protected = frozenset(resolved.reserved) | {resolved.player, resolved.flagpole}
    enemy_ids = tuple(
        catalog.id_for_name(name)
        for name in config.categories.get("enemies", [])
        if name in catalog.name_to_id and not catalog.is_flying_enemy(name)
    )
    # Config order, not set order: the planners place blocks in this sequence.
    blocks = block_names(config)
    block_ids = tuple(
        catalog.id_for_name(name)
        for name in config.categories.get("solid_like", [])
        if name in blocks and name in catalog.name_to_id
    )
    pipe_names = tuple(config.categories.get("pipes", []))
    return RefineContext(
        config=config,
        catalog=catalog,
        width=width,
        height=height,
        area=width * height,
        empty_id=catalog.id_for_name("empty"),
        solid_id=catalog.id_for_name(config.layout.ground_tile),
        coin_id=catalog.id_for_name("coin"),
        enemy_ids=enemy_ids,
        block_ids=block_ids,
        pipe_ids=frozenset(catalog.id_for_name(name) for name in pipe_names),
        pipe_top_left=catalog.id_for_name("pipe_top_left"),
        pipe_top_right=catalog.id_for_name("pipe_top_right"),
        pipe_body_left=catalog.id_for_name("pipe_body_left"),
        pipe_body_right=catalog.id_for_name("pipe_body_right"),
        solid_like=frozenset(solid_like_ids(catalog, config)),
        player=resolved.player,
        flagpole=resolved.flagpole,
        protected=protected,
        protected_columns=frozenset(column for _, column in protected),
        base_height=height - resolved.ground_row,
    )


def target_count(value: float, band_max: float, area: int) -> int:
    """Tile count whose density measures ``value`` on a ``[0, band_max]`` band.

    Args:
        value: Target band value in ``[0, 1]``.
        band_max: Raw density mapped to 1 (band minimum is 0 for densities).
        area: Level area in tiles.

    Returns:
        Rounded tile count; the residual is at most half a band step.
    """
    return round(float(value) * band_max * area)


def target_amplitude(value: float, config: RuntimeConfig) -> int:
    """Envelope amplitude (rows) measuring ``value`` on the verticality band.

    Args:
        value: Target verticality in ``[0, 1]``.
        config: Runtime configuration with the frozen band.
    """
    band = config.metrics.verticality_band
    raw = band.min + float(value) * (band.max - band.min)
    return round(raw * max(1, config.game.height_tiles - 2))


def target_mean_delta(value: float, config: RuntimeConfig) -> float:
    """Mean envelope step measuring ``value`` on the smoothness band.

    Args:
        value: Target path smoothness in ``[0, 1]``.
        config: Runtime configuration with the frozen band.
    """
    band = config.metrics.path_smoothness_band
    raw = band.min + float(value) * (band.max - band.min)
    return (1.0 - raw) * max(1, config.validation.jump_height)


def target_gap_score(value: float, width_tiles: int) -> float:
    """Raw gap score measuring ``value`` for a level of the given width.

    Args:
        value: Target gap difficulty in ``[0, 1]``.
        width_tiles: Level width in tiles.
    """
    return float(value) * max(1.0, width_tiles * GAP_BUDGET_PER_COLUMN)


def target_elevated_fraction(value: float, config: RuntimeConfig) -> float:
    """Elevated-column fraction measuring ``value`` on the platform band.

    Args:
        value: Target platform usage in ``[0, 1]``.
        config: Runtime configuration with the frozen band.
    """
    band = config.metrics.platform_usage_band
    return band.min + float(value) * (band.max - band.min)
