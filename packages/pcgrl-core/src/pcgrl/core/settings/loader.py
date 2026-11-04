from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any, cast

import yaml

from pcgrl.core.corpus.profile import CorpusProfile
from pcgrl.core.helpers.path import PROJECT_ROOT, ensure_dir, resolve_path
from pcgrl.core.settings.schema import RuntimeConfig


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a recursive merge without mutating either input mapping.

    Args:
        base: Base mapping.
        override: Mapping whose values take precedence.

    Returns:
        New merged mapping.
    """
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _expand_templates(data: dict[str, Any]) -> dict[str, Any]:
    """Expand supported string templates in a raw config mapping.

    Args:
        data: Raw merged YAML mapping.

    Returns:
        Mapping with template placeholders substituted.
    """
    experiment = data.get("experiment", {})
    name = experiment.get("name", "run")

    def expand(value: Any) -> Any:
        """Recursively substitute ``${experiment.name}`` in strings."""
        if isinstance(value, str):
            return value.replace("${experiment.name}", str(name))
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        return value

    return cast(dict[str, Any], expand(data))


def _git_commit_hash() -> str | None:
    """Return the current commit hash when inside a Git worktree."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _file_hash(path: Path) -> str | None:
    """Return a stable content hash for a file, or None when absent."""
    if not path.is_file():
        return None
    digest = hashlib.blake2b(path.read_bytes(), digest_size=16)
    return digest.hexdigest()


def default_config_paths(game: str = "mario", training: str = "medium") -> list[Path]:
    """Return the ordered config files for a game and experiment tier.

    Args:
        game: Game config stem under ``configs/``.
        training: Experiment tier stem under ``configs/experiments/``.

    Returns:
        Primary game YAML followed by the experiment override YAML.
    """
    primary = PROJECT_ROOT / f"configs/{game}.yaml"
    experiment = PROJECT_ROOT / f"configs/experiments/{training}.yaml"
    return [primary, experiment]


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from disk.

    Args:
        path: Path to a YAML file.

    Returns:
        Parsed top-level mapping.

    Raises:
        ValueError: When the YAML root is not a mapping.
        OSError: When the file cannot be read.
    """
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML in {path} must be a mapping")
    return cast(dict[str, Any], data)


def apply_corpus_profile(config: RuntimeConfig, profile: CorpusProfile) -> RuntimeConfig:
    """Apply data-derived density and geometry rules to a validated config.

    Args:
        config: Validated runtime configuration.
        profile: Corpus profile loaded from the processed dataset.

    Returns:
        Updated configuration, or the original when profiling is disabled or
        unavailable.
    """
    policy = config.metrics.corpus_profile
    if not policy.enabled or not profile.available:
        return config

    def value_or_default(value: float | None, fallback: float) -> float:
        """Return ``value`` as float when set, otherwise ``fallback``."""
        return float(value) if value is not None else float(fallback)

    def int_or_default(value: float | None, fallback: int) -> int:
        """Return rounded ``value`` when set, otherwise ``fallback``."""
        return round(value) if value is not None else int(fallback)

    target_density = config.metrics.target_density
    new_target_density = target_density.model_copy(
        update={
            name: value_or_default(
                profile.density(name, policy.density_pct),
                getattr(target_density, name),
            )
            for name in ("enemy", "coin", "pipe", "platform", "block")
        }
    )
    ranges = config.metrics.target_density_ranges
    new_ranges = ranges.model_copy(
        update={
            name: getattr(ranges, name).model_copy(
                update={"max": getattr(new_target_density, name)}
            )
            for name in ("enemy", "coin", "pipe", "platform", "block")
        }
    )
    semantic = config.metrics.semantic
    new_semantic = semantic.model_copy(
        update={
            "coin_max_height_above_surface": int_or_default(
                profile.height_pct("coin", policy.coin_height_max_pct),
                semantic.coin_max_height_above_surface,
            ),
            "block_max_height_above_surface": int_or_default(
                profile.height_pct("block", policy.block_height_max_pct),
                semantic.block_max_height_above_surface,
            ),
        }
    )
    new_metrics = config.metrics.model_copy(
        update={
            "target_density": new_target_density,
            "target_density_ranges": new_ranges,
            "semantic": new_semantic,
        }
    )
    validation = config.validation
    new_validation = validation.model_copy(
        update={
            "max_gap_width": int_or_default(
                profile.gap_width(policy.gap_width_pct),
                validation.max_gap_width,
            ),
            "max_gap_count": int_or_default(
                profile.gap_count(policy.gap_count_pct),
                validation.max_gap_count,
            ),
            "pipe_min_gap": int_or_default(
                profile.pipe_spacing(policy.pipe_gap_pct),
                validation.pipe_min_gap,
            ),
            "min_floating_platform_height": int_or_default(
                profile.min_curve_height(
                    "floating_platform",
                    policy.platform_min_curve_threshold,
                ),
                validation.min_floating_platform_height,
            ),
        }
    )
    return config.model_copy(update={"metrics": new_metrics, "validation": new_validation})


def _is_primary_config(path: Path) -> bool:
    """Return whether a path is a built-in primary config file."""
    resolved = resolve_path(path)
    return resolved in {
        PROJECT_ROOT / "configs/base.yaml",
        PROJECT_ROOT / "configs/mario.yaml",
    }


def load_config(
    *,
    game: str | None = None,
    training: str = "medium",
    override_path: Path | None = None,
    seed: int | None = None,
    apply_profile: bool = True,
) -> RuntimeConfig:
    """Load, merge, validate, and optionally profile-calibrate runtime settings.

    Args:
        game: Game config stem; defaults to ``mario``.
        training: Experiment tier override stem.
        override_path: Optional extra YAML merged after the defaults.
        seed: Optional seed injected into ``experiment.seed``.
        apply_profile: Whether to apply corpus-profile overrides.

    Returns:
        Validated ``RuntimeConfig`` instance.

    Raises:
        ValueError: When merged YAML fails pydantic validation.
        OSError: When a config file cannot be read.
    """
    game_name = game or "mario"
    merged: dict[str, Any] = {}
    for path in default_config_paths(game=game_name, training=training):
        if path.is_file():
            merged = _deep_merge(merged, load_yaml(path))

    if override_path is not None and not _is_primary_config(override_path):
        resolved_override = resolve_path(override_path)
        if resolved_override.is_file():
            merged = _deep_merge(merged, load_yaml(resolved_override))

    merged = _expand_templates(merged)
    if seed is not None:
        merged.setdefault("experiment", {})
        merged["experiment"]["seed"] = seed
    if game is not None:
        merged.setdefault("experiment", {})
        merged["experiment"]["game"] = game

    flat = {
        "experiment": merged.get("experiment", {}),
        "log": merged.get("log", {}),
        "game": merged.get("game", {}),
        "tiles": merged.get("tiles", {}),
        "aliases": merged.get("aliases", {}),
        "categories": merged.get("categories", {}),
        "validation": merged.get("validation", {}),
        "layout": merged.get("layout", {}),
        "area_tiles": merged.get("area_tiles", {}),
        "dataset": merged.get("dataset", {}),
        "library": merged.get("library", {}),
        "model": merged.get("model", {}),
        "training": merged.get("training", {}),
        "reward": merged.get("reward", {}),
        "metrics": merged.get("metrics", {}),
        "environment": merged.get("environment", {}),
        "generation": merged.get("generation", {}),
        "evaluation": merged.get("evaluation", {}),
        "default_target": merged.get("default_target", {}),
        "translator": merged.get("translator", {}),
    }
    config = RuntimeConfig.model_validate(flat)
    if apply_profile and config.metrics.corpus_profile.enabled:
        profile = CorpusProfile.load(resolve_path(config.dataset.processed_root))
        config = apply_corpus_profile(config, profile)
    return config


def resolve_and_snapshot(config: RuntimeConfig, *, run_dir: Path) -> Path:
    """Persist an immutable config snapshot and provenance metadata for a run.

    Args:
        config: Validated runtime configuration to snapshot.
        run_dir: Run artifact directory.

    Returns:
        Path to the written ``config.snapshot.yaml`` file.

    Raises:
        OSError: When snapshot or seed files cannot be written.
        RuntimeError: When Git metadata cannot be collected and is required.
    """
    ensure_dir(run_dir)
    snapshot_path = run_dir / "config.snapshot.yaml"
    meta = {
        "git_commit": _git_commit_hash(),
        "dataset_meta_hash": _file_hash(resolve_path(config.dataset.processed_root) / "meta.json"),
        "bridge_jar_hash": _file_hash(resolve_path("vendor/mario/mario-bridge.jar")),
    }
    payload = {"provenance": meta, "config": config.snapshot_dict()}
    with snapshot_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    seed_path = run_dir / "seed.txt"
    seed_path.write_text(str(config.experiment.seed), encoding="utf-8")
    return snapshot_path
