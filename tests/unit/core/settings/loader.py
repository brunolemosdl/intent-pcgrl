from __future__ import annotations

import pytest
from pcgrl.core.settings.loader import (
    _deep_merge,
    _expand_templates,
    default_config_paths,
    load_config,
    load_yaml,
)
from pcgrl.core.settings.schema import RuntimeConfig


@pytest.mark.parametrize(
    ("tier", "expected_steps", "expected_eval_episodes"),
    (
        ("quick", 300_000, 1),
        ("initial", 750_000, 2),
        ("medium", 1_500_000, 3),
        ("advanced", 3_000_000, 3),
        ("final", 10_000_000, 5),
    ),
)
def test_training_tiers_define_progressive_budgets(
    tier: str,
    expected_steps: int,
    expected_eval_episodes: int,
) -> None:
    """Verify training tiers define progressive budgets."""
    config = load_config(training=tier)
    assert isinstance(config, RuntimeConfig)
    assert config.training.total_steps == expected_steps
    assert config.evaluation.eval_episodes == expected_eval_episodes
    assert config.training.batch_size <= (
        config.training.num_environments * config.training.rollout_steps
    )


def test_default_config_paths_use_profile_layout() -> None:
    """Verify default config paths use profile layout."""
    paths = default_config_paths(training="quick")
    assert paths[0].name == "mario.yaml"
    assert paths[1].as_posix().endswith("configs/experiments/quick.yaml")


def test_load_config_seed_override() -> None:
    """Verify load config seed override."""
    config = load_config(training="initial", seed=123)
    assert config.experiment.seed == 123


def test_load_config_tiles_nonempty(config) -> None:
    """Verify load config layout from game yaml."""
    assert len(config.tiles) >= 10


def test_load_config_layout_from_game_yaml(config) -> None:
    """Verify load config layout from game yaml."""
    assert config.layout.player == (-2, 1)
    assert config.layout.flagpole == (-2, -2)
    assert len(config.layout.fixed_regions) == 2


def test_load_config_environment_renamed_fields(config) -> None:
    """Verify environment behaviour flags survive the merge."""
    assert config.environment.parallel_env_seed_stride == 997
    assert config.environment.composer.duplicate_window >= 0


def test_load_config_spatial_metric_fields(config) -> None:
    """Verify load config spatial metric fields."""
    assert config.metrics.spatial.segment_count >= 2
    assert 0.0 <= config.metrics.spatial.concentration_threshold <= 1.0


def test_load_config_dataset_fields(config) -> None:
    """Verify dataset paths survive the merge."""
    assert config.dataset.corpus_root.endswith("corpus/mario")
    assert config.dataset.processed_root.endswith("processed/mario")


def test_load_config_generation_renamed_fields(config) -> None:
    """Verify load config generation renamed fields."""
    assert config.generation.max_attempts >= 1
    assert config.generation.require_gameplay_bridge is True
    assert config.generation.require_bridge_win is True
    assert 0.0 <= config.generation.minimum_completion_ratio <= 1.0
    assert config.generation.bridge_num_runs >= 1


def test_load_config_evaluation_has_agents(config) -> None:
    """Verify deep merge nested."""
    assert len(config.evaluation.agents) > 0
    assert config.evaluation.num_runs >= 1


def test_deep_merge_flat() -> None:
    """Verify deep merge flat."""
    base = {"a": 1, "b": 2}
    override = {"b": 99, "c": 3}
    result = _deep_merge(base, override)
    assert result == {"a": 1, "b": 99, "c": 3}


def test_deep_merge_nested() -> None:
    """Verify deep merge nested."""
    base = {"training": {"lr": 0.0001, "steps": 1000}}
    override = {"training": {"steps": 5000}}
    result = _deep_merge(base, override)
    assert result["training"]["lr"] == pytest.approx(0.0001)
    assert result["training"]["steps"] == 5000


def test_deep_merge_base_unchanged() -> None:
    """Verify deep merge base unchanged."""
    base = {"a": {"b": 1}}
    override = {"a": {"c": 2}}
    _deep_merge(base, override)
    assert "c" not in base["a"]


def test_expand_templates_name() -> None:
    """Verify expand templates name."""
    data = {"experiment": {"name": "myrun"}, "path": "artifacts/${experiment.name}/out"}
    result = _expand_templates(data)
    assert result["path"] == "artifacts/myrun/out"


def test_expand_templates_nested() -> None:
    """Verify expand templates nested."""
    data = {"experiment": {"name": "test"}, "nested": {"key": "${experiment.name}_value"}}
    result = _expand_templates(data)
    assert result["nested"]["key"] == "test_value"


def test_load_yaml_valid(tmp_path) -> None:
    """Verify load yaml valid."""
    p = tmp_path / "cfg.yaml"
    p.write_text("key: value\n")
    result = load_yaml(p)
    assert result == {"key": "value"}


def test_load_yaml_non_mapping_raises(tmp_path) -> None:
    """Verify load yaml non mapping raises."""
    p = tmp_path / "list.yaml"
    p.write_text("- a\n- b\n")
    with pytest.raises(ValueError, match="mapping"):
        load_yaml(p)


def test_load_config_override_path(tmp_path, config) -> None:
    """Verify load config override path."""
    override = tmp_path / "override.yaml"
    override.write_text("experiment:\n  seed: 777\n")
    cfg = load_config(training="initial", override_path=override)
    assert cfg.experiment.seed == 777


def test_load_config_passes_library_block(tmp_path) -> None:
    """Verify load config passes library block."""
    # Regression for the loader flat-dict: top-level keys not listed there are
    # silently dropped, which would make `library:` overrides no-ops.
    override = tmp_path / "override.yaml"
    override.write_text("library:\n  stride_tiles: 4\n  max_chunks: 100\n")
    cfg = load_config(training="initial", override_path=override, apply_profile=False)
    assert cfg.library.stride_tiles == 4
    assert cfg.library.max_chunks == 100


def test_load_config_with_game_param() -> None:
    """Verify load config with game param."""
    cfg = load_config(game="mario", training="initial")
    assert cfg.experiment.game == "mario"
    assert cfg.game.width_tiles > 0


def test_git_commit_hash_returns_string_or_none() -> None:
    """Verify git commit hash returns string or none."""
    from pcgrl.core.settings.loader import _git_commit_hash

    result = _git_commit_hash()
    assert result is None or isinstance(result, str)


def test_git_commit_hash_success() -> None:
    """Verify git commit hash success."""
    from unittest.mock import MagicMock, patch

    from pcgrl.core.settings.loader import _git_commit_hash

    mock_proc = MagicMock()
    mock_proc.stdout = "abc123def456\n"
    with patch("pcgrl.core.settings.loader.subprocess.run", return_value=mock_proc):
        result = _git_commit_hash()
    assert result == "abc123def456"


def test_file_hash_existing_file(tmp_path) -> None:
    """Verify file hash existing file."""
    from pcgrl.core.settings.loader import _file_hash

    f = tmp_path / "data.bin"
    f.write_bytes(b"hello world")
    h = _file_hash(f)
    assert h is not None
    assert len(h) == 32


def test_file_hash_missing_file(tmp_path) -> None:
    """Verify file hash missing file."""
    from pcgrl.core.settings.loader import _file_hash

    result = _file_hash(tmp_path / "nonexistent.bin")
    assert result is None


def test_resolve_and_snapshot_creates_files(tmp_path) -> None:
    """Verify resolve and snapshot creates files."""
    from pcgrl.core.settings.loader import resolve_and_snapshot

    cfg = load_config(training="initial")
    path = resolve_and_snapshot(cfg, run_dir=tmp_path)
    assert (tmp_path / "config.snapshot.yaml").is_file()
    assert (tmp_path / "seed.txt").is_file()
    assert path.name == "config.snapshot.yaml"
