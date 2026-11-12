from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from pcgrl.core.domain.level import LevelGrid
from pcgrl.core.settings.schema import RuntimeConfig
from pcgrl.core.tiles.catalog import TileCatalog
from pcgrl.runtime.metrics.usefulness import block_names, solid_like_ids
from pcgrl.runtime.validators.reachability import reachable_cells

INTENT_KEYS = (
    "enemy_density",
    "coin_density",
    "interactive_block_density",
    "pipe_density",
    "gap_difficulty",
    "verticality",
    "path_smoothness",
    "platform_usage",
)

GAP_BUDGET_PER_COLUMN = 1.0 / 24.0

# Dimensions that count objects: "none" for them means zero objects.
COUNT_KEYS = (
    "enemy_density",
    "coin_density",
    "interactive_block_density",
    "pipe_density",
    "gap_difficulty",
)
# The language bridge writes "no"/"none" as ~0.05 (see its intensity mapping),
# and 0.05 of a density cap is about one object, so count targets in this band
# are read as zero.
NONE_BAND = 0.05


def target_value(target: np.ndarray, key: str) -> float:
    """One target coordinate as the decimal it was written as.

    Targets travel as float32, where 0.05 is stored as 0.0500000007; rounding
    keeps threshold comparisons such as ``<= 0.05`` exact.
    """
    return round(float(target[INTENT_KEYS.index(key)]), 6)


def effective_target(target: np.ndarray) -> np.ndarray:
    """Intent vector the generator drives toward.

    Count dimensions requested at or below ``NONE_BAND`` become zero; every
    other coordinate is the request unchanged. The environment, the terminal
    reward, the polish stage and the generation report all use this reading,
    so "no enemies" means the same thing everywhere.

    Args:
        target: Requested intent vector in ``[0, 1]^8``.

    Returns:
        Float32 copy with the "none" band snapped to zero.
    """
    effective = np.asarray(target, dtype=np.float32).copy()
    for key in COUNT_KEYS:
        if target_value(effective, key) <= NONE_BAND:
            effective[INTENT_KEYS.index(key)] = 0.0
    return effective


@lru_cache(maxsize=64)
def _id_table(ids: frozenset[int]) -> np.ndarray:
    """Boolean lookup table over tile ids (ids are small catalog indices)."""
    table = np.zeros(max(ids) + 1, dtype=bool)
    table[list(ids)] = True
    return table


def _count_names(tiles: np.ndarray, catalog: TileCatalog, names: set[str]) -> int:
    """Total tiles matching any of the given catalog names."""
    ids = frozenset(catalog.id_for_name(name) for name in names if name in catalog.name_to_id)
    if not ids:
        return 0
    table = _id_table(ids)
    inside = tiles < len(table)
    return int((table[np.where(inside, tiles, 0)] & inside).sum())


def _normalize_band(value: float, minimum: float, maximum: float) -> float:
    """Clip-normalise a raw density into a corpus band, yielding [0, 1]."""
    if maximum <= minimum:
        return 0.0
    return float(np.clip((value - minimum) / (maximum - minimum), 0.0, 1.0))


def ground_height_profile(
    level: LevelGrid, catalog: TileCatalog, config: RuntimeConfig
) -> list[int]:
    """Per-column height of the bottom-contiguous solid stack (0 = floor gap)."""
    solid = np.isin(level.tiles, list(solid_like_ids(catalog, config)))
    heights: list[int] = np.cumprod(solid[::-1], axis=0).sum(axis=0).tolist()
    return heights


def _default_start(level: LevelGrid, ground_heights: list[int]) -> tuple[int, int] | None:
    """Virtual spawn on the surface of the leftmost grounded column.

    Used for corpus levels, which carry no player tile."""
    for column, height in enumerate(ground_heights):
        if height > 0:
            return (level.height_tiles - 1 - height, column)
    return None


def traversable_path_envelope(
    level: LevelGrid,
    catalog: TileCatalog,
    config: RuntimeConfig,
    *,
    start: tuple[int, int] | None = None,
) -> list[int | None]:
    """Per-column max standable height reachable from the start (None = no path).

    Description:
        Height counts rows above the grid bottom: standing on a 2-tall floor is
        height 2; a platform 4 rows above it is height 6. The envelope includes
        every standable structure the flood can reach — floating platforms
        included, not just the ground stack — which is what a player perceives
        as verticality. Flood physics come from
        ``config.validation.jump_height/jump_reach``.

    Args:
        level: See signature.
        catalog: See signature.
        config: See signature.
        start: See signature.

    Returns:
        list[int | None]: Result of the operation.
    """
    ground = ground_height_profile(level, catalog, config)
    spawn: tuple[int, int] | None = None
    if start is None:
        spawn = _default_start(level, ground)
        if spawn is None:
            return [None] * level.width_tiles
        # A generated level is flooded from its player tile, the cell the
        # gates and the reward start from, so one flood serves all three;
        # the virtual spawn next to it still counts as path.
        hits = np.argwhere(level.tiles == catalog.id_for_name("player"))
        start = (int(hits[0][0]), int(hits[0][1])) if hits.size else spawn
    reach = reachable_cells(
        level,
        catalog,
        jump_height=config.validation.jump_height,
        jump_reach=config.validation.jump_reach,
        start=start,
    )
    envelope: list[int | None] = [None] * level.width_tiles
    cells = reach.standable if spawn is None else (*reach.standable, spawn)
    for row, column in cells:
        height = level.height_tiles - 1 - row
        current = envelope[column]
        if current is None or height > current:
            envelope[column] = height
    return envelope


def gap_widths(ground_heights: list[int]) -> list[int]:
    """Widths of maximal runs of floor-gap columns (ground height 0)."""
    widths: list[int] = []
    run = 0
    for height in ground_heights:
        if height == 0:
            run += 1
        elif run:
            widths.append(run)
            run = 0
    if run:
        widths.append(run)
    return widths


def compute_gap_difficulty(
    ground_heights: list[int], *, jump_reach: int, width_tiles: int
) -> float:
    """Sum of per-gap difficulty over a width-proportional gap budget.

    Description:
        A gap of jump_reach+1 columns is just-impossible and scores 1.0; wider
        gaps do not score more (they are caught by playability, not difficulty).
        The budget is GAP_BUDGET_PER_COLUMN gaps per level column, an E2
        recalibration candidate.

    Args:
        ground_heights: See signature.
        jump_reach: See signature.
        width_tiles: See signature.

    Returns:
        float: Result of the operation.
    """
    widths = gap_widths(ground_heights)
    if not widths:
        return 0.0
    cap = jump_reach + 1
    budget = max(1.0, width_tiles * GAP_BUDGET_PER_COLUMN)
    score = sum(min(width, cap) / cap for width in widths)
    return float(np.clip(score / budget, 0.0, 1.0))


def compute_verticality(
    envelope: list[int | None],
    height_tiles: int,
    *,
    band_min: float = 0.0,
    band_max: float = 1.0,
) -> float:
    """Amplitude of the path envelope, band-renormalised.

    Description:
        The raw amplitude ratio saturates: one tall pipe or staircase pushes a
        level toward the ceiling, compressing the corpus into ~[0.67, 1.0].
        The band (``metrics.verticality_band``, frozen from the viability
        map's p05/p95) stretches the ACHIEVABLE range over [0, 1], so 0 means
        "as flat as the composer can build" and 1 "as vertical as it gets".

    Args:
        envelope: Per-column traversable path heights (None = gap).
        height_tiles: Level height for the raw amplitude ratio.
        band_min: Raw ratio mapped to 0.
        band_max: Raw ratio mapped to 1.
    """
    heights = [h for h in envelope if h is not None]
    if not heights:
        return 0.0
    amplitude = max(heights) - min(heights)
    raw = amplitude / max(1, height_tiles - 2)
    return _normalize_band(raw, band_min, band_max)


def compute_path_smoothness(
    envelope: list[int | None],
    *,
    jump_height: int,
    band_min: float = 0.0,
    band_max: float = 1.0,
) -> float:
    """1 - mean |delta| between consecutive path columns, band-renormalised.

    Description:
        The raw score (in jump-height units) compresses achievable levels
        into ~[0.64, 0.93] — low targets were impossible (E5 sweeps). The
        band (``metrics.path_smoothness_band``, frozen from the achievable
        p05/p95) stretches that range over [0, 1]: 0 means "as bumpy as the
        composer can build", 1 "as flat as it gets". Gap columns are skipped
        (scored by the gap dims instead).

    Args:
        envelope: See signature.
        jump_height: See signature.
        band_min: Raw score mapped to 0.
        band_max: Raw score mapped to 1.

    Returns:
        float: Result of the operation.
    """
    heights = [h for h in envelope if h is not None]
    if len(heights) < 2:
        return 1.0
    deltas = [abs(b - a) for a, b in itertools.pairwise(heights)]
    mean_delta = sum(deltas) / len(deltas)
    raw = float(np.clip(1.0 - mean_delta / max(1, jump_height), 0.0, 1.0))
    return _normalize_band(raw, band_min, band_max)


def compute_platform_usage(
    envelope: list[int | None],
    ground_heights: list[int],
    *,
    band_min: float = 0.0,
    band_max: float = 1.0,
) -> float:
    """Fraction of elevated path columns, band-renormalised.

    Description:
        A column is elevated when its best standing point sits ABOVE the
        ground-connected stack — traversal over platforms, pipes or blocks.
        The raw fraction compresses achievable levels into ~[0.11, 0.34] —
        high targets were impossible (E5 sweeps). The band
        (``metrics.platform_usage_band``, frozen from the achievable
        p05/p95) stretches that range over [0, 1].

    Args:
        envelope: See signature.
        ground_heights: See signature.
        band_min: Raw fraction mapped to 0.
        band_max: Raw fraction mapped to 1.

    Returns:
        float: Result of the operation.
    """
    total = 0
    elevated = 0
    for path_height, ground_height in zip(envelope, ground_heights, strict=True):
        if path_height is None:
            continue
        total += 1
        if path_height > ground_height:
            elevated += 1
    if total == 0:
        return 0.0
    return _normalize_band(elevated / total, band_min, band_max)


@dataclass(frozen=True)
class IntentMeasurement:
    """Measured intent vector plus the path geometry it was derived from.

    Attributes:
        vector: Float32 intent vector ordered as ``INTENT_KEYS``.
        envelope: Per-column max standable height reachable from the spawn
            (``None`` where no path exists).
        ground: Per-column height of the bottom-contiguous solid stack.
    """

    vector: np.ndarray
    envelope: tuple[int | None, ...]
    ground: tuple[int, ...]

    @property
    def envelope_min(self) -> int:
        """Lowest reachable standing height (the verticality baseline).

        Amplitude targets are anchored on the measured envelope minimum, so
        they hold for any floor the composition ends up with.
        """
        heights = [h for h in self.envelope if h is not None]
        if heights:
            return min(heights)
        return min((h for h in self.ground if h > 0), default=1)

    @property
    def amplitude(self) -> int:
        """Rows between the lowest and highest reachable standing heights."""
        heights = [h for h in self.envelope if h is not None]
        return max(heights) - min(heights) if heights else 0

    @property
    def path_columns(self) -> int:
        """Number of columns the path envelope covers."""
        return sum(1 for h in self.envelope if h is not None)

    @property
    def mean_step(self) -> float:
        """Mean height step between consecutive path columns, in rows."""
        heights = [h for h in self.envelope if h is not None]
        if len(heights) < 2:
            return 0.0
        return sum(abs(b - a) for a, b in itertools.pairwise(heights)) / (len(heights) - 1)

    @property
    def elevated_columns(self) -> int:
        """Path columns whose standing point sits above the ground stack."""
        return sum(
            1
            for path, ground in zip(self.envelope, self.ground, strict=True)
            if path is not None and path > ground
        )


def measure_intent(
    level: LevelGrid,
    catalog: TileCatalog,
    config: RuntimeConfig,
    *,
    start: tuple[int, int] | None = None,
) -> IntentMeasurement:
    """Measure a level: intent vector, path envelope and ground profile.

    Args:
        level: Stitched or partial level grid.
        catalog: Tile catalog.
        config: Runtime configuration with density bands and jump physics.
        start: Optional flood start override for corpus levels.

    Returns:
        ``IntentMeasurement`` whose vector is ordered as ``INTENT_KEYS``,
        each coordinate in ``[0, 1]``.
    """
    area = max(1, level.width_tiles * level.height_tiles)
    ranges = config.metrics.target_density_ranges
    enemies = _count_names(level.tiles, catalog, set(config.categories.get("enemies", [])))
    coins = _count_names(level.tiles, catalog, {"coin"})
    blocks = _count_names(level.tiles, catalog, block_names(config))
    pipes = _count_names(level.tiles, catalog, set(config.categories.get("pipes", [])))
    ground = ground_height_profile(level, catalog, config)
    envelope = traversable_path_envelope(level, catalog, config, start=start)
    vector = np.array(
        [
            _normalize_band(enemies / area, ranges.enemy.min, ranges.enemy.max),
            _normalize_band(coins / area, ranges.coin.min, ranges.coin.max),
            _normalize_band(blocks / area, ranges.block.min, ranges.block.max),
            _normalize_band(pipes / area, ranges.pipe.min, ranges.pipe.max),
            compute_gap_difficulty(
                ground,
                jump_reach=config.validation.jump_reach,
                width_tiles=level.width_tiles,
            ),
            compute_verticality(
                envelope,
                level.height_tiles,
                band_min=config.metrics.verticality_band.min,
                band_max=config.metrics.verticality_band.max,
            ),
            compute_path_smoothness(
                envelope,
                jump_height=config.validation.jump_height,
                band_min=config.metrics.path_smoothness_band.min,
                band_max=config.metrics.path_smoothness_band.max,
            ),
            compute_platform_usage(
                envelope,
                ground,
                band_min=config.metrics.platform_usage_band.min,
                band_max=config.metrics.platform_usage_band.max,
            ),
        ],
        dtype=np.float32,
    )
    return IntentMeasurement(vector=vector, envelope=tuple(envelope), ground=tuple(ground))


def compute_intent_vector(
    level: LevelGrid,
    catalog: TileCatalog,
    config: RuntimeConfig,
    *,
    start: tuple[int, int] | None = None,
) -> np.ndarray:
    """Compute the measured intent vector.

    Args:
        level: Stitched or partial level grid.
        catalog: Tile catalog.
        config: Runtime configuration with density bands and jump physics.
        start: Optional flood start override for corpus levels.

    Returns:
        Float32 vector ordered as ``INTENT_KEYS``, each in ``[0, 1]``.
    """
    return measure_intent(level, catalog, config, start=start).vector
