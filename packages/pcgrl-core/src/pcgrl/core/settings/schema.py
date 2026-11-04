from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    """Frozen pydantic base that rejects unknown fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ExperimentConfig(StrictModel):
    """Top-level experiment metadata.

    Attributes:
        name: Human-readable run name used in artifact paths.
        seed: Global random seed.
        artifacts_root: Root directory for run outputs.
        game: Game identifier (e.g. ``mario``).
    """

    name: str = "mario_ppo"
    seed: int = 42
    artifacts_root: str = "artifacts/runs"
    game: str = "mario"


class LogConfig(StrictModel):
    """Logging configuration.

    Attributes:
        level: Logger level name (e.g. ``info``, ``debug``).
    """

    level: str = "info"


class GameConfig(StrictModel):
    """Playfield dimensions and rendering scale.

    Attributes:
        width_tiles: Level width in tiles.
        height_tiles: Level height in tiles.
        tile_size_px: Pixel size of each tile.
        area_type: Playfield theme identifier.
    """

    width_tiles: int = 152
    height_tiles: int = 16
    tile_size_px: int = 16
    area_type: Literal["overworld"] = "overworld"

    @property
    def width_px(self) -> int:
        """Return the level width in pixels."""
        return self.width_tiles * self.tile_size_px

    @property
    def height_px(self) -> int:
        """Return the level height in pixels."""
        return self.height_tiles * self.tile_size_px

    @model_validator(mode="after")
    def validate_dimensions(self) -> GameConfig:
        """Ensure tile dimensions are positive."""
        if self.width_tiles < 1:
            raise ValueError("game.width_tiles must be >= 1")
        if self.height_tiles < 1:
            raise ValueError("game.height_tiles must be >= 1")
        if self.tile_size_px < 1:
            raise ValueError("game.tile_size_px must be >= 1")
        return self


class ValidationConfig(StrictModel):
    """Mario jump physics and structural placement constraints.

    Description:
        Single source of truth for jump reach used by the composer stitching
        mask and reachability flood; calibrated by the ``calibrate`` command.

    Attributes:
        jump_height: Maximum jump height in tiles.
        jump_reach: Maximum horizontal jump reach in tiles.
        min_solid_ratio: Minimum allowed solid-tile fraction.
        max_solid_ratio: Maximum allowed solid-tile fraction.
        min_pipe_height: Minimum pipe body height in tiles.
        require_grounded_pipes: Whether pipes must sit on solid ground.
        require_pipe_width_two: Whether pipes must be two tiles wide.
        allow_floating_enemies: Whether aerial enemies are permitted.
        pipe_min_gap: Minimum horizontal spacing between distinct pipes.
        max_gap_width: Maximum width of a contiguous floor hole.
        max_gap_count: Maximum number of distinct floor holes (0 = no cap).
        min_floating_platform_height: Minimum height for floating platforms.
    """

    # Single source of truth for Mario jump physics. Used by the composer
    # stitching mask and the reachability flood; calibrated against the bridge
    # A* by the `calibrate` command (E0) instead of hand-picked.
    jump_height: int = 4
    jump_reach: int = 5
    min_solid_ratio: float = 0.063
    max_solid_ratio: float = 0.65
    min_pipe_height: int = 2
    require_grounded_pipes: bool = True
    require_pipe_width_two: bool = True
    allow_floating_enemies: bool = False
    # Minimum horizontal spacing (columns) between distinct pipes, enforced by
    # the action mask so pipes spread across the level instead of clustering.
    pipe_min_gap: int = 6
    # Maximum width (columns) of a contiguous floor hole. The action mask blocks
    # digging a gap that would exceed this, so the floor stays corpus-like
    # (overworld holes are mostly width 1-4) instead of forming huge chasms.
    max_gap_width: int = 6
    # Maximum number of distinct floor holes (0 = no cap). Derived from the corpus
    # profile (per-width-normalised gap count) so the floor is not over-perforated
    # (corpus levels carry ~9 holes; an unconstrained policy tends toward ~37).
    # The mask blocks digging a new hole once this many already exist.
    max_gap_count: int = 0
    # Minimum height (rows above the local surface) for a floating platform/block;
    # the mask blocks placing one below this, killing the "glued to the ground"
    # degenerate. 0 = no floor. Derived from the corpus height histogram.
    min_floating_platform_height: int = 0


class FixedRegionConfig(StrictModel):
    """Rectangular reserved region in layout coordinates.

    Attributes:
        origin: Top-left ``(row, col)`` anchor (may be negative-relative).
        size: Region extent as ``(height, width)`` in tiles.
    """

    origin: tuple[int, int]
    size: tuple[int, int]


class LayoutConfig(StrictModel):
    """Spawn, flagpole, and fixed-region layout anchors.

    Description:
        Grid coordinates are ``[row, col]``. Non-negative values are absolute
        positions from the top-left corner. Negative row values count from the
        bottom; negative col values count from the right.

    Attributes:
        player: Player spawn coordinate.
        flagpole: Flagpole coordinate.
        ground_row: Row index filled with ground tiles.
        fixed_regions: Reserved rectangles (spawn and ending zones).
        fill_full_ground: Whether to pre-fill the ground row.
        ground_tile: Tile name used for ground fill.
    """

    player: tuple[int, int] = (-2, 1)
    flagpole: tuple[int, int] = (-2, -2)
    ground_row: int = -1
    fixed_regions: list[FixedRegionConfig] = Field(
        default_factory=lambda: [
            FixedRegionConfig(origin=(-2, 0), size=(2, 3)),
            FixedRegionConfig(origin=(-2, -3), size=(2, 3)),
        ]
    )
    fill_full_ground: bool = True
    ground_tile: str = "solid"

    @model_validator(mode="after")
    def validate_layout(self) -> LayoutConfig:
        """Ensure fixed regions have positive size."""
        for region in self.fixed_regions:
            if region.size[0] < 1 or region.size[1] < 1:
                raise ValueError("layout.fixed_regions.size must be at least [1, 1]")
        return self


class GameYamlConfig(StrictModel):
    """Game-definition YAML block (tiles, layout, validation).

    Attributes:
        game: Playfield dimensions.
        tiles: Canonical tile name to glyph mapping.
        aliases: Alternate tile names accepted in ASCII input.
        categories: Tile groupings used for metrics and masking.
        validation: Jump and placement constraints.
        layout: Spawn and reserved-region anchors.
        area_tiles: Allowed tile names per area type.
    """

    game: GameConfig
    tiles: dict[str, str]
    aliases: dict[str, str] = Field(default_factory=dict)
    categories: dict[str, list[str]] = Field(default_factory=dict)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    layout: LayoutConfig = Field(default_factory=LayoutConfig)
    area_tiles: dict[str, list[str]] = Field(default_factory=dict)


class DatasetConfig(StrictModel):
    """Corpus ingestion settings.

    Attributes:
        corpus_root: Root directory of raw VGLC levels.
        processed_root: Output directory for processed corpus artifacts.
    """

    corpus_root: str
    processed_root: str


class LibraryConfig(StrictModel):
    """Chunk-library extraction stage settings.

    Description:
        Chunks are cut from raw corpus text levels (bottom-anchored, with
        provenance), never from the processed segment npz.

    Attributes:
        level_sets: Corpus subdirectories to scan (e.g. ``mario-1``).
        output_root: Directory for ``chunk_library.npz`` and sidecar JSON.
        chunk_widths: Candidate chunk widths in tiles.
        stride_tiles: Horizontal stride when sliding windows.
        max_chunks: Maximum chunks retained after deduplication.
        deduplicate: Whether to collapse identical tile patterns.
        heldout_level_fraction: Fraction of levels reserved for evaluation.
        ending_columns: Right-edge columns classified as the ending family.
        viability_episodes: Random rollouts for the intent viability map.
    """

    # Level sets read from <dataset.corpus_root>/<set>; target style is SMB1,
    # so only mario-1 enters by default.
    level_sets: list[str] = Field(default_factory=lambda: ["mario-1"])
    output_root: str = "data/processed/mario/library"
    chunk_widths: list[int] = Field(default_factory=lambda: [16])
    stride_tiles: int = 8
    max_chunks: int = 800
    deduplicate: bool = True
    # Fraction of LEVELS (not windows) kept out of the library and the reward's
    # pattern reference, reserved for the anti-Goodhart style evaluation.
    heldout_level_fraction: float = 0.2
    # Columns at the right edge of each level treated as the ENDING family
    # (SMB1 staircase + flag region).
    ending_columns: int = 16
    # Random-legal rollouts used to build the intent viability map (E2).
    viability_episodes: int = 300

    @model_validator(mode="after")
    def validate_library(self) -> LibraryConfig:
        """Validate library extraction numeric constraints."""
        if not self.level_sets:
            raise ValueError("library.level_sets must not be empty")
        if not self.chunk_widths or any(width < 1 for width in self.chunk_widths):
            raise ValueError("library.chunk_widths must contain positive widths")
        if self.stride_tiles < 1:
            raise ValueError("library.stride_tiles must be >= 1")
        if self.max_chunks < 1:
            raise ValueError("library.max_chunks must be >= 1")
        if not 0.0 <= self.heldout_level_fraction < 1.0:
            raise ValueError("library.heldout_level_fraction must be in [0, 1)")
        if self.ending_columns < 1:
            raise ValueError("library.ending_columns must be >= 1")
        if self.viability_episodes < 2:
            raise ValueError("library.viability_episodes must be >= 2")
        return self


class ModelConfig(StrictModel):
    """Policy and value-network architecture settings.

    Attributes:
        actor: Actor-head hidden dimensions.
        critic: Critic-head hidden dimensions.
    """

    actor: dict[str, int] = Field(default_factory=lambda: {"hidden_dim": 256})
    critic: dict[str, int] = Field(
        default_factory=lambda: {"value_hidden_dim": 256}
    )


class TrainingConfig(StrictModel):
    """PPO training schedule and optimisation hyperparameters.

    Attributes:
        total_steps: Total environment steps across all workers.
        num_environments: Parallel rollout workers.
        rollout_steps: Steps collected per worker per update.
        batch_size: Minibatch size for policy updates.
        num_epochs: Epochs per rollout batch.
        learning_rate: Adam learning rate.
        gamma: Discount factor.
        gae_lambda: GAE lambda for advantage estimation.
        clip_range: PPO clip range.
        entropy_coefficient: Entropy bonus weight.
        value_function_coefficient: Value-loss weight.
        max_grad_norm: Gradient clipping threshold.
        log_interval: Updates between log lines.
        save_interval_steps: Steps between checkpoint saves.
    """

    total_steps: int = 6000000
    num_environments: int = 8
    rollout_steps: int = 2048
    batch_size: int = 1024
    num_epochs: int = 4
    learning_rate: float = 0.0003
    # The composer episode is ~10-19 terminal-rewarded decisions, so the
    # default is undiscounted; gamma stays configurable for ablations.
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coefficient: float = 0.01
    value_function_coefficient: float = 0.5
    max_grad_norm: float = 0.5
    log_interval: int = 10
    save_interval_steps: int = 100000


class RewardGameplayConfig(StrictModel):
    """Bridge gameplay evaluation toggles for reward computation.

    Attributes:
        enabled: Whether to run bridge evaluation for rewards.
        cache: Whether to cache bridge results per level hash.
    """

    enabled: bool = True
    cache: bool = True


class RewardComposerConfig(StrictModel):
    """Terminal-only composer reward weights.

    Description:
        Every term is normalised to ``[0, 1]``. The total reward combines
        intent alignment, TPKL style, playability flood, and duplicate-chunk
        penalties.

    Attributes:
        weight_alignment: Weight on intent-vector alignment.
        alignment_mean_weight: Mean-error weight inside alignment.
        alignment_max_weight: Max-error weight inside alignment.
        weight_style: Weight on TPKL style term.
        style_pattern_size: Pattern size ``k`` for training-time TPKL.
        weight_playability: Weight on start-to-flag flood playability.
        weight_duplicate: Weight on recent duplicate-chunk penalty.
    """

    weight_alignment: float = 4.0
    alignment_mean_weight: float = 0.6
    alignment_max_weight: float = 0.4
    weight_style: float = 2.0
    # Pattern size k used by the TPKL style term during TRAINING (evaluation
    # uses evaluation.tpkl_sizes against the heldout levels instead).
    style_pattern_size: int = 3
    weight_playability: float = 3.0
    weight_duplicate: float = 1.0

    @model_validator(mode="after")
    def validate_composer_reward(self) -> RewardComposerConfig:
        """Ensure alignment weights sum to 1.0 and pattern size is positive."""
        total = self.alignment_mean_weight + self.alignment_max_weight
        if abs(total - 1.0) > 1e-06:
            raise ValueError(
                "reward.composer alignment mean/max weights must sum to 1.0"
            )
        if self.style_pattern_size < 1:
            raise ValueError("reward.composer.style_pattern_size must be >= 1")
        return self


class RewardConfig(StrictModel):
    """Reward model configuration.

    Attributes:
        composer: Terminal composer reward weights.
        gameplay: Bridge evaluation cache toggles.
    """

    composer: RewardComposerConfig = Field(default_factory=RewardComposerConfig)
    gameplay: RewardGameplayConfig = Field(default_factory=RewardGameplayConfig)


class TargetDensityConfig(StrictModel):
    """Per-category density targets for masking and metrics.

    Attributes:
        enemy: Target enemy density.
        coin: Target coin density.
        pipe: Target pipe density.
        platform: Target platform density.
        block: Target block density.
    """

    enemy: float = 0.04
    coin: float = 0.06
    pipe: float = 0.05
    platform: float = 0.08
    block: float = 0.15


class DensityRangeConfig(StrictModel):
    """Inclusive density bounds for a single category.

    Attributes:
        min: Minimum allowed density.
        max: Maximum allowed density.
    """

    min: float
    max: float

    @model_validator(mode="after")
    def validate_range(self) -> DensityRangeConfig:
        """Ensure ``max`` exceeds ``min`` and both are non-negative."""
        if self.min < 0.0:
            raise ValueError("metrics.target_density_ranges.*.min must be >= 0")
        if self.max <= self.min:
            raise ValueError("metrics.target_density_ranges.*.max must be > min")
        return self


class TargetDensityRangesConfig(StrictModel):
    """Per-category density ranges used by generation gates.

    Attributes:
        enemy: Enemy density bounds.
        coin: Coin density bounds.
        pipe: Pipe density bounds.
        platform: Platform density bounds.
        block: Block density bounds.
    """

    enemy: DensityRangeConfig = Field(
        default_factory=lambda: DensityRangeConfig(min=0.0, max=0.035)
    )
    coin: DensityRangeConfig = Field(
        default_factory=lambda: DensityRangeConfig(min=0.0, max=0.08)
    )
    pipe: DensityRangeConfig = Field(
        default_factory=lambda: DensityRangeConfig(min=0.0, max=0.05)
    )
    platform: DensityRangeConfig = Field(
        default_factory=lambda: DensityRangeConfig(min=0.0, max=0.08)
    )
    block: DensityRangeConfig = Field(
        default_factory=lambda: DensityRangeConfig(min=0.0, max=0.18)
    )


class SpatialMetricConfig(StrictModel):
    """Spatial-distribution metric parameters.

    Attributes:
        segment_count: Number of horizontal segments for concentration checks.
        concentration_threshold: Maximum allowed per-segment concentration.
    """

    segment_count: int = 8
    concentration_threshold: float = 0.4

    @model_validator(mode="after")
    def validate_spatial(self) -> SpatialMetricConfig:
        """Validate segment count and concentration threshold."""
        if self.segment_count < 2:
            raise ValueError("metrics.spatial.segment_count must be >= 2")
        if not 0.0 <= self.concentration_threshold <= 1.0:
            raise ValueError(
                "metrics.spatial.concentration_threshold must be in [0, 1]"
            )
        return self


class SemanticConfig(StrictModel):
    """Corpus-derived placement bands consumed by the usefulness metrics.

    Description:
        ``apply_corpus_profile`` overrides the height bands from
        ``corpus_profile.json`` percentiles at load time.

    Attributes:
        coin_max_height_above_surface: Maximum coin height above surface.
        block_max_height_above_surface: Maximum block height above surface.
        min_platform_width: Minimum platform width in tiles.
    """

    coin_max_height_above_surface: int = 5
    block_max_height_above_surface: int = 5
    min_platform_width: int = 3

    @model_validator(mode="after")
    def validate_semantic(self) -> SemanticConfig:
        """Validate semantic threshold ranges."""
        if self.min_platform_width < 1:
            raise ValueError("metrics.semantic.min_platform_width must be >= 1")
        return self


class CorpusProfileConfig(StrictModel):
    """Policy for deriving generation rules from the corpus profile.

    Description:
        Selects which percentiles to read from ``corpus_profile.json``. When
        enabled and the profile is present, ``apply_corpus_profile`` overrides
        densities, height bands, and gap rules before the config is frozen.

    Attributes:
        enabled: Whether corpus-profile overrides are applied at load time.
        density_pct: Percentile used as per-category density cap.
        coin_height_max_pct: Percentile for coin max height above surface.
        block_height_max_pct: Percentile for block max height above surface.
        platform_min_curve_threshold: Histogram mass threshold for platform floor.
        gap_width_pct: Percentile for maximum floor-gap width.
        gap_count_pct: Percentile for maximum floor-gap count.
        pipe_gap_pct: Percentile for minimum pipe spacing.
    """

    enabled: bool = True
    # Percentile used as the per-category density cap (corpus rarely exceeds it).
    density_pct: int = 95
    # Percentile for the coin/block max-height-above-surface band.
    coin_height_max_pct: int = 90
    block_height_max_pct: int = 90
    # Floating-platform floor: smallest height whose corpus histogram mass is
    # >= this threshold becomes the minimum (the mask blocks floating platforms
    # below it). 0.10 -> ~3 on the current corpus (mode is 4), so the too-low
    # floating platforms the policy favoured are blocked while 3+ stays allowed
    # and the height_match reward pulls the bulk toward the 4/8 modes.
    platform_min_curve_threshold: float = 0.10
    # Percentiles for the floor-gap caps.
    gap_width_pct: int = 90
    gap_count_pct: int = 90
    # Percentile for the minimum pipe spacing (lower = allow corpus-close pipes).
    pipe_gap_pct: int = 25


class MetricsConfig(StrictModel):
    """Intent, masking, and evaluation metric configuration.

    Attributes:
        target_density: Per-category density targets.
        target_density_ranges: Per-category density bounds.
        verticality_band: Raw verticality ratios mapped to ``[0, 1]``.
        path_smoothness_band: Raw path smoothness mapped to ``[0, 1]``.
        platform_usage_band: Raw platform usage mapped to ``[0, 1]``.
        spatial: Spatial-distribution parameters.
        semantic: Corpus-derived placement bands.
        corpus_profile: Corpus-profile override policy.
    """

    target_density: TargetDensityConfig = Field(default_factory=TargetDensityConfig)
    target_density_ranges: TargetDensityRangesConfig = Field(
        default_factory=TargetDensityRangesConfig
    )
    # Raw verticality ratios mapped to [0, 1]. Frozen from the p05/p95 of the
    # viability map (E2): the raw amplitude metric compresses the corpus into
    # ~[0.67, 1.0]; banding by the ACHIEVABLE distribution makes
    # the dimension controllable. Frozen (not derived at load time) so the
    # metric cannot drift with every library rebuild.
    verticality_band: DensityRangeConfig = Field(
        default_factory=lambda: DensityRangeConfig(min=0.41, max=0.92)
    )
    # Raw path-smoothness mapped to [0, 1]. Frozen from the p05/p95 over 500
    # random legal compositions (E5): the raw metric compresses achievable
    # levels into ~[0.64, 0.93], making low targets impossible without banding.
    path_smoothness_band: DensityRangeConfig = Field(
        default_factory=lambda: DensityRangeConfig(min=0.64, max=0.93)
    )
    # Raw platform-usage mapped to [0, 1]. Same E5 freeze: achievable levels
    # sat in ~[0.11, 0.34], making high targets impossible.
    platform_usage_band: DensityRangeConfig = Field(
        default_factory=lambda: DensityRangeConfig(min=0.11, max=0.34)
    )
    spatial: SpatialMetricConfig = Field(default_factory=SpatialMetricConfig)
    semantic: SemanticConfig = Field(default_factory=SemanticConfig)
    corpus_profile: CorpusProfileConfig = Field(default_factory=CorpusProfileConfig)


class PolishConfig(StrictModel):
    """Deterministic post-selection refinement toward the intent target.

    Description:
        The generation pipeline's hard-tolerance safety net: after best-of-N
        selection the ``IntentRefiner`` closes any residual measured-vs-target
        gap the learned edit phase left, re-gated by the structural chain,
        flood, and bridge with fallback to the unpolished level. Evaluation
        harnesses disable it explicitly to measure the raw policy.

    Attributes:
        enabled: Whether generation polishes the selected candidate.
        tolerance: Per-dimension absolute error target.
        max_passes: Refiner passes before accepting the residual.
    """

    enabled: bool = True
    tolerance: float = 0.05
    max_passes: int = 3

    @model_validator(mode="after")
    def validate_polish(self) -> PolishConfig:
        """Validate tolerance and pass bounds."""
        if not 0.0 < self.tolerance < 1.0:
            raise ValueError("generation.polish.tolerance must be in (0, 1)")
        if self.max_passes < 1:
            raise ValueError("generation.polish.max_passes must be >= 1")
        return self


class TargetMixConfig(StrictModel):
    """Training-target sampling mixture for the composer environment.

    Description:
        The edit phase makes (nearly) the whole intent cube reachable, so
        training targets blend the corpus-anchored viability cloud with
        uniform exploration and explicit corner emphasis (each dimension
        drawn from {0, 1, U(0, 1)}), teaching the policy to use edits at the
        extremes the chunk material cannot reach.

    Attributes:
        viability: Share of targets sampled from the viability cloud.
        uniform: Share of targets sampled uniformly over the cube.
        corners: Share of corner-emphasis targets.
    """

    viability: float = 0.4
    uniform: float = 0.4
    corners: float = 0.2

    @model_validator(mode="after")
    def validate_mix(self) -> TargetMixConfig:
        """Shares must be non-negative and sum to 1."""
        total = self.viability + self.uniform + self.corners
        if min(self.viability, self.uniform, self.corners) < 0:
            raise ValueError("target_mix shares must be >= 0")
        if abs(total - 1.0) > 1e-6:
            raise ValueError("target_mix shares must sum to 1")
        return self


class ComposerEnvConfig(StrictModel):
    """Two-phase composer environment settings.

    Description:
        An episode composes chunks until an ENDING closes the level, then
        enters the edit phase: the same policy applies structured macro-edit
        actions (the refine-op vocabulary) to the stitched level until STOP
        or the edit budget runs out. The terminal-only reward is computed on
        the edited level.

    Attributes:
        duplicate_window: Recent decisions checked for duplicate-chunk penalty.
        require_start_family: Whether the first chunk must be a start family.
        require_ending_family: Whether the last chunk must be an ending family.
        intent_budget_mask: Whether intent budgets join the legal-chunk mask
            (factor C of the training factorial).
        width_slack_tiles: Allowed width mismatch during stitching.
        edit_budget: Maximum edit-phase actions per episode (0 skips the
            phase — an experiment-only ablation cell, not a product mode).
        target_mix: Training-target sampling shares.
    """

    # Sliding window checked by the duplicate mask/penalty: re-picking a chunk
    # that appeared in the last N decisions is discouraged.
    duplicate_window: int = 3
    require_start_family: bool = True
    require_ending_family: bool = True
    intent_budget_mask: bool = True
    # Slack (in tiles) allowed between the composed width and game.width_tiles;
    # the final column count is padded/clamped during stitching.
    width_slack_tiles: int = 8
    edit_budget: int = 24
    target_mix: TargetMixConfig = Field(default_factory=lambda: TargetMixConfig())

    @model_validator(mode="after")
    def validate_composer(self) -> ComposerEnvConfig:
        """Validate composer environment numeric constraints."""
        if self.duplicate_window < 0:
            raise ValueError("environment.composer.duplicate_window must be >= 0")
        if self.width_slack_tiles < 0:
            raise ValueError("environment.composer.width_slack_tiles must be >= 0")
        if self.edit_budget < 0:
            raise ValueError("environment.composer.edit_budget must be >= 0")
        return self


class EnvironmentConfig(StrictModel):
    """RL environment behaviour flags.

    Attributes:
        composer: Composer-specific settings.
        parallel_env_seed_stride: Seed stride between parallel workers.
    """

    composer: ComposerEnvConfig = Field(default_factory=ComposerEnvConfig)
    parallel_env_seed_stride: int = 997


class GenerationConfig(StrictModel):
    """Composer level-generation retry and gate settings.

    Description:
        Candidate ranking uses the composer terminal reward itself, so the only
        knobs left are the attempt budget and the bridge win/completion gate.

    Attributes:
        max_attempts: Maximum generation attempts per target.
        require_gameplay_bridge: Whether bridge evaluation is required.
        require_bridge_win: Whether at least one bridge run must win.
        bridge_num_runs: Stochastic bridge runs per candidate.
        minimum_completion_ratio: Minimum mean completion ratio to accept.
        bridge_validation_start_fraction: Fraction of attempts before bridge gating.
        polish: Deterministic post-selection refinement settings.
    """

    max_attempts: int = 5
    require_gameplay_bridge: bool = True
    require_bridge_win: bool = True
    bridge_num_runs: int = 3
    minimum_completion_ratio: float = 0.95
    bridge_validation_start_fraction: float = 0.0
    polish: PolishConfig = Field(default_factory=PolishConfig)

    @model_validator(mode="after")
    def validate_generation(self) -> GenerationConfig:
        """Validate generation gate numeric ranges."""
        if not 0.0 <= self.bridge_validation_start_fraction <= 1.0:
            raise ValueError(
                "generation.bridge_validation_start_fraction must be in [0, 1]"
            )
        if not 0.0 <= self.minimum_completion_ratio <= 1.0:
            raise ValueError("generation.minimum_completion_ratio must be in [0, 1]")
        if self.bridge_num_runs < 1:
            raise ValueError("generation.bridge_num_runs must be >= 1")
        if self.max_attempts < 1:
            raise ValueError("generation.max_attempts must be >= 1")
        return self


class EvaluationTargetCase(StrictModel):
    """Named intent-target override for evaluation suites.

    Attributes:
        name: Case identifier used in reports.
        overrides: Partial ``IntentTargetConfig`` field overrides.
    """

    name: str
    overrides: dict[str, float]

    @model_validator(mode="after")
    def validate_overrides(self) -> EvaluationTargetCase:
        """Ensure override keys are valid intent dimensions."""
        valid = set(IntentTargetConfig.field_names())
        unknown = set(self.overrides) - valid
        if unknown:
            raise ValueError(
                f"evaluation.target_suite overrides contain unknown fields: {sorted(unknown)}"
            )
        return self


def _default_target_suite() -> list[EvaluationTargetCase]:
    """Return the default evaluation target sweep cases."""
    return [
        EvaluationTargetCase(name="low_enemy", overrides={"enemy_density": 0.1}),
        EvaluationTargetCase(name="high_enemy", overrides={"enemy_density": 0.9}),
        EvaluationTargetCase(name="low_coin", overrides={"coin_density": 0.1}),
        EvaluationTargetCase(name="high_coin", overrides={"coin_density": 0.9}),
        EvaluationTargetCase(name="low_verticality", overrides={"verticality": 0.1}),
        EvaluationTargetCase(name="high_verticality", overrides={"verticality": 0.9}),
        EvaluationTargetCase(
            name="low_platform_usage", overrides={"platform_usage": 0.1}
        ),
        EvaluationTargetCase(
            name="high_platform_usage", overrides={"platform_usage": 0.9}
        ),
        EvaluationTargetCase(
            name="easy_level",
            overrides={"gap_difficulty": 0.1, "enemy_density": 0.2, "path_smoothness": 0.9},
        ),
        EvaluationTargetCase(
            name="hard_level",
            overrides={"gap_difficulty": 0.8, "enemy_density": 0.8, "path_smoothness": 0.4},
        ),
        EvaluationTargetCase(
            name="balanced_level",
            overrides={
                "enemy_density": 0.5,
                "coin_density": 0.5,
                "interactive_block_density": 0.5,
                "pipe_density": 0.5,
                "gap_difficulty": 0.5,
                "verticality": 0.5,
                "path_smoothness": 0.5,
                "platform_usage": 0.5,
            },
        ),
        EvaluationTargetCase(
            name="sparse_level",
            overrides={
                "enemy_density": 0.1,
                "coin_density": 0.1,
                "interactive_block_density": 0.15,
                "pipe_density": 0.1,
                "platform_usage": 0.15,
            },
        ),
    ]


class ControllabilityConfig(StrictModel):
    """Per-dimension controllability sweep protocol.

    Description:
        Computes MAE and Spearman rho over ``sweep_values`` x
        ``levels_per_value`` generated levels per training seed.

    Attributes:
        sweep_values: Target values swept per intent dimension.
        levels_per_value: Generated levels per sweep value.
        max_mae: Maximum acceptable mean absolute error.
        min_spearman: Minimum acceptable Spearman correlation.
    """

    # The 0.0/1.0 endpoints are the acceptance criterion of the edit-phase
    # architecture: every slider extreme must be deliverable, not just the
    # interior of the corpus manifold.
    sweep_values: list[float] = Field(
        default_factory=lambda: [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    )
    levels_per_value: int = 10
    max_mae: float = 0.10
    min_spearman: float = 0.8

    @model_validator(mode="after")
    def validate_controllability(self) -> ControllabilityConfig:
        """Validate sweep values and per-value level count."""
        if not self.sweep_values:
            raise ValueError("evaluation.controllability.sweep_values must not be empty")
        if any(not 0.0 <= value <= 1.0 for value in self.sweep_values):
            raise ValueError(
                "evaluation.controllability.sweep_values must be in [0, 1]"
            )
        if self.levels_per_value < 1:
            raise ValueError(
                "evaluation.controllability.levels_per_value must be >= 1"
            )
        return self


class CalibrationConfig(StrictModel):
    """Flood-vs-bridge physics calibration settings.

    Description:
        Builds a confusion matrix over corpus, perturbed, and generated levels
        to freeze ``validation.jump_height`` and ``validation.jump_reach``.

    Attributes:
        num_levels: Number of levels sampled for calibration.
        max_false_playable: Maximum allowed false-playable rate.
        jump_height_candidates: Candidate jump heights to search.
        jump_reach_candidates: Candidate jump reaches to search.
    """

    num_levels: int = 120
    max_false_playable: float = 0.05
    jump_height_candidates: list[int] = Field(default_factory=lambda: [3, 4, 5])
    jump_reach_candidates: list[int] = Field(default_factory=lambda: [4, 5, 6, 7])

    @model_validator(mode="after")
    def validate_calibration(self) -> CalibrationConfig:
        """Validate calibration sample size and false-playable bound."""
        if self.num_levels < 1:
            raise ValueError("evaluation.calibration.num_levels must be >= 1")
        if not 0.0 <= self.max_false_playable <= 1.0:
            raise ValueError(
                "evaluation.calibration.max_false_playable must be in [0, 1]"
            )
        return self


class EvaluationConfig(StrictModel):
    """Bridge evaluation, benchmarking, and metric-suite settings.

    Attributes:
        agents: Mario agent names used for bridge evaluation.
        require_gameplay_bridge: Whether bridge runs are mandatory.
        timer_seconds: Bridge episode timer in seconds.
        mario_state: Initial Mario power-up state for bridge runs.
        num_runs: Stochastic bridge runs per level.
        benchmark_seeds: Seeds used for benchmark generation.
        benchmark_levels_per_seed: Levels generated per benchmark seed.
        target_suite: Named intent-target evaluation cases.
        eval_episodes: Episodes per inline training evaluation.
        benchmark_episodes: Episodes for standalone benchmark runs.
        eval_every_steps: Training steps between inline evaluations.
        minimum_agent_win_rate: Minimum acceptable agent win rate.
        minimum_average_completion: Minimum acceptable completion ratio.
        controllability: Controllability sweep protocol.
        calibration: Jump-physics calibration protocol.
        tpkl_sizes: TPKL pattern sizes evaluated against heldout levels.
    """

    agents: list[str]
    require_gameplay_bridge: bool = True
    timer_seconds: int = 30
    mario_state: int = 0
    num_runs: int = 5
    benchmark_seeds: list[int] = Field(default_factory=lambda: [11, 22, 33])
    benchmark_levels_per_seed: int = 10
    target_suite: list[EvaluationTargetCase] = Field(
        default_factory=_default_target_suite
    )
    eval_episodes: int = 3
    benchmark_episodes: int = 50
    eval_every_steps: int = 100000
    minimum_agent_win_rate: float = 0.0
    minimum_average_completion: float = 0.0
    controllability: ControllabilityConfig = Field(
        default_factory=ControllabilityConfig
    )
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    # Pattern sizes evaluated by the TPKL style metric against the heldout
    # levels (the reward only ever sees reward.composer.style_pattern_size
    # against corpus_train — anti-Goodhart split).
    tpkl_sizes: list[int] = Field(default_factory=lambda: [2, 3, 4])


class IntentTargetConfig(StrictModel):
    """Intent target vector defaults and overrides.

    Description:
        Formulas live in ``pcgrl.runtime.metrics.intent``. Defaults sit near
        corpus medians so unconditioned generation requests a typical SMB1 level.

    Attributes:
        enemy_density: Target enemy density in ``[0, 1]``.
        coin_density: Target coin density in ``[0, 1]``.
        interactive_block_density: Target interactive-block density.
        pipe_density: Target pipe density.
        gap_difficulty: Target gap difficulty.
        verticality: Target verticality.
        path_smoothness: Target path smoothness.
        platform_usage: Target platform usage.
    """

    enemy_density: float = 0.5
    coin_density: float = 0.4
    interactive_block_density: float = 0.5
    pipe_density: float = 0.3
    gap_difficulty: float = 0.4
    verticality: float = 0.65
    path_smoothness: float = 0.40
    platform_usage: float = 0.55

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        """Return intent dimension field names in vector order."""
        return tuple(cls.model_fields.keys())

    def to_vector(self) -> list[float]:
        """Serialise intent fields to a float vector in ``field_names`` order."""
        return [float(getattr(self, name)) for name in self.field_names()]

    def with_overrides(self, **overrides: float) -> IntentTargetConfig:
        """Return a copy with selected intent fields overridden.

        Args:
            **overrides: Field name to value mappings.

        Returns:
            New ``IntentTargetConfig`` instance.

        Raises:
            ValueError: When an unknown field name is provided.
        """
        data = self.model_dump()
        unknown = set(overrides) - set(data)
        if unknown:
            raise ValueError(f"Unknown target fields: {sorted(unknown)}")
        data.update(overrides)
        return self.model_validate(data)


class TranslatorConfig(StrictModel):
    """Natural-language description -> intent vector translator settings.

    Attributes:
        provider: Backend producing the vector (``openai_compatible`` or
            ``fake``).
        mode: ``live`` (no store), ``record`` (store misses call the backend),
            or ``replay`` (store only; misses fail loudly).
        model: Model id for the LLM backend.
        max_tokens: Response token cap for structured-output calls that use it.
        base_url: OpenAI-compatible endpoint override (paid or local).
        api_key_env: Environment variable holding the API key.
        store_path: JSON record/replay store location.
        grounding: Whether the prompt includes corpus bands and viability
            quantiles (``False`` is the zero-shot factorial arm).
    """

    provider: Literal["openai_compatible", "fake"] = "openai_compatible"
    mode: Literal["live", "record", "replay"] = "record"
    model: str = "gpt-5.2"
    max_tokens: int = 1024
    # OpenAI-compatible endpoint base URL; None means the official OpenAI API.
    # Covers Gemini (generativelanguage.googleapis.com/v1beta/openai/) and
    # local servers (Ollama http://localhost:11434/v1, LM Studio, vLLM).
    base_url: str | None = None
    # Environment variable holding the API key; None picks OPENAI_API_KEY.
    # Local servers that need
    # no key work when base_url is set and the variable is absent.
    api_key_env: str | None = None
    store_path: str = "data/descriptions/recorded_store.json"
    grounding: bool = True

    @model_validator(mode="after")
    def validate_translator(self) -> TranslatorConfig:
        """Validate translator numeric constraints."""
        if self.max_tokens < 64:
            raise ValueError("translator.max_tokens must be >= 64")
        return self


class RuntimeConfig(StrictModel):
    """Fully validated runtime configuration for training and generation.

    Description:
        Aggregates every YAML block into a single frozen model. Loaded by
        ``pcgrl.core.settings.loader.load_config`` and snapshotted per run.

    Attributes:
        experiment: Top-level experiment metadata.
        log: Logging configuration.
        game: Playfield dimensions.
        tiles: Canonical tile name to glyph mapping.
        aliases: Tile name aliases.
        categories: Tile category groupings.
        validation: Jump and placement constraints.
        layout: Spawn and reserved-region anchors.
        area_tiles: Allowed tiles per area type.
        dataset: Corpus ingestion settings.
        library: Chunk-library extraction settings.
        model: Policy architecture settings.
        training: PPO training hyperparameters.
        reward: Reward model configuration.
        metrics: Intent and masking metric settings.
        environment: RL environment behaviour flags.
        generation: Generation gate settings.
        evaluation: Evaluation and benchmarking settings.
        default_target: Default intent vector for unconditioned generation.
        translator: Description -> intent translator settings.
    """

    experiment: ExperimentConfig
    log: LogConfig = Field(default_factory=LogConfig)
    game: GameConfig
    tiles: dict[str, str]
    aliases: dict[str, str] = Field(default_factory=dict)
    categories: dict[str, list[str]] = Field(default_factory=dict)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    layout: LayoutConfig = Field(default_factory=LayoutConfig)
    area_tiles: dict[str, list[str]] = Field(default_factory=dict)
    dataset: DatasetConfig
    library: LibraryConfig = Field(default_factory=LibraryConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig
    default_target: IntentTargetConfig = Field(default_factory=IntentTargetConfig)
    translator: TranslatorConfig = Field(default_factory=TranslatorConfig)

    @field_validator("tiles")
    @classmethod
    def validate_tiles_non_empty(cls, value: dict[str, str]) -> dict[str, str]:
        """Ensure the tile catalog mapping is non-empty."""
        if not value:
            raise ValueError("tiles catalog must not be empty")
        return value

    @model_validator(mode="after")
    def validate_runtime(self) -> RuntimeConfig:
        """Validate batch size and layout bounds."""
        if (
            self.training.batch_size
            > self.training.num_environments * self.training.rollout_steps
        ):
            raise ValueError(
                "training.batch_size must be <= num_environments * rollout_steps"
            )
        self._validate_layout_bounds()
        return self

    @property
    def target_vector_dim(self) -> int:
        """Return the intent target vector dimensionality."""
        return len(IntentTargetConfig.field_names())

    def target_vector_field_names(self) -> tuple[str, ...]:
        """Return ordered intent dimension names for the target vector."""
        return IntentTargetConfig.field_names()

    def default_target_vector(self) -> list[float]:
        """Return the default intent target as a float vector."""
        return self.default_target.to_vector()

    def build_target_vector(self, **overrides: float) -> list[float]:
        """Build an intent target vector with optional field overrides.

        Args:
            **overrides: Partial intent field overrides.

        Returns:
            Float vector in ``field_names`` order.
        """
        return self.default_target.with_overrides(**overrides).to_vector()

    def _validate_layout_bounds(self) -> None:
        """Ensure layout anchors and fixed regions resolve inside the grid."""
        from pcgrl.core.layout import region_cells, resolve_layout

        height = self.game.height_tiles
        width = self.game.width_tiles
        layout = self.layout
        resolved = resolve_layout(layout, height=height, width=width)

        def in_bounds(row: int, col: int) -> bool:
            """Return whether ``(row, col)`` lies inside the configured grid."""
            return 0 <= row < height and 0 <= col < width

        if not in_bounds(*resolved.player):
            raise ValueError(f"layout.player {layout.player} resolves outside the grid")
        if not in_bounds(*resolved.flagpole):
            raise ValueError(
                f"layout.flagpole {layout.flagpole} resolves outside the grid"
            )
        if not 0 <= resolved.ground_row < height:
            raise ValueError(
                f"layout.ground_row {layout.ground_row} resolves outside the grid"
            )
        for index, region in enumerate(layout.fixed_regions):
            for row, col in region_cells(region, height=height, width=width):
                if not in_bounds(row, col):
                    raise ValueError(
                        f"layout.fixed_regions[{index}] cell ({row}, {col}) is outside the grid"
                    )

    def snapshot_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable configuration mapping for run snapshots."""
        return self.model_dump(mode="json")
