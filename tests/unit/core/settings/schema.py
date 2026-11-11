from __future__ import annotations

import pytest
from pcgrl.core.settings.schema import (
    GameConfig,
    RuntimeConfig,
    TrainingConfig,
)
from pydantic import ValidationError


def test_training_config_has_valid_optimization_budget() -> None:
    """Verify training config has valid optimization budget."""
    tc = TrainingConfig()
    rollout_budget = tc.num_environments * tc.rollout_steps
    assert rollout_budget > 0
    assert tc.batch_size <= rollout_budget
    assert 0.0 <= tc.entropy_coefficient <= 1.0
    assert 0.0 <= tc.value_function_coefficient <= 1.0


def test_game_config_tile_dimensions(config) -> None:
    """Verify game config invalid width tiles."""
    assert config.game.width_tiles > 0
    assert config.game.width_px == config.game.width_tiles * config.game.tile_size_px
    assert config.game.height_px == config.game.height_tiles * config.game.tile_size_px


def test_game_config_invalid_width_tiles(config) -> None:
    """Verify runtime config frozen."""
    raw = config.game.model_dump()
    raw["width_tiles"] = 0
    with pytest.raises(ValidationError, match="width_tiles"):
        GameConfig.model_validate(raw)


def test_runtime_config_frozen(config) -> None:
    """Verify runtime config frozen."""
    with pytest.raises(ValidationError):
        config.experiment.seed = 99


def test_runtime_config_batch_size_too_large(config) -> None:
    """Verify runtime config batch size too large."""
    raw = config.model_dump(mode="python")
    raw["training"]["batch_size"] = 999999
    with pytest.raises(ValidationError, match="batch_size"):
        RuntimeConfig.model_validate(raw)


def test_build_target_vector_uses_configured_defaults(config) -> None:
    """Verify build target vector rejects unknown fields."""
    default_vector = config.default_target_vector()
    assert config.build_target_vector() == default_vector
    assert len(default_vector) == config.target_vector_dim
    overridden = config.build_target_vector(enemy_density=0.9, verticality=0.1)
    assert overridden[0] == pytest.approx(0.9)
    assert overridden[5] == pytest.approx(0.1)
    assert overridden[1] == pytest.approx(config.default_target.coin_density)


def test_build_target_vector_rejects_unknown_fields(config) -> None:
    """Verify build target vector rejects unknown fields."""
    with pytest.raises(ValueError, match="Unknown target fields"):
        config.build_target_vector(not_a_metric=0.1)


def test_library_config_defaults(config) -> None:
    """Verify library config defaults."""
    library = config.library
    assert library.level_sets == ["mario-1"]
    assert library.chunk_widths == [16]
    assert library.stride_tiles == 8
    assert 0.0 <= library.heldout_level_fraction < 1.0


def test_library_config_rejects_bad_values() -> None:
    """Verify environment kind defaults to composer."""
    from pcgrl.core.settings.schema import LibraryConfig

    with pytest.raises(ValidationError, match="chunk_widths"):
        LibraryConfig(chunk_widths=[])
    with pytest.raises(ValidationError, match="heldout_level_fraction"):
        LibraryConfig(heldout_level_fraction=1.0)
    with pytest.raises(ValidationError, match="level_sets"):
        LibraryConfig(level_sets=[])


def test_environment_composer_settings(config) -> None:
    """Verify the composer environment block exposes its mask knobs."""
    assert config.environment.composer.duplicate_window >= 0
    assert config.environment.parallel_env_seed_stride >= 1


def test_validation_exposes_jump_physics(config) -> None:
    """Verify validation exposes jump physics."""
    assert config.validation.jump_height == 4
    assert config.validation.jump_reach == 5


def test_reward_composer_alignment_weights_must_sum_to_one() -> None:
    """Verify reward composer alignment weights must sum to one."""
    from pcgrl.core.settings.schema import RewardComposerConfig

    with pytest.raises(ValidationError, match="sum"):
        RewardComposerConfig(alignment_mean_weight=0.7, alignment_max_weight=0.4)
    composer = RewardComposerConfig()
    assert composer.weight_alignment == pytest.approx(4.0)
    assert composer.style_pattern_size == 3


def test_evaluation_composer_blocks(config) -> None:
    """Verify evaluation composer blocks."""
    assert config.evaluation.tpkl_sizes == [2, 3, 4]
    assert config.evaluation.controllability.sweep_values == [
        0.0,
        0.1,
        0.3,
        0.5,
        0.7,
        0.9,
        1.0,
    ]
    assert config.evaluation.calibration.num_levels >= 1


def test_training_defaults_are_terminal_only() -> None:
    """Verify the default training discount matches the terminal-only reward."""
    assert TrainingConfig().gamma == pytest.approx(1.0)
