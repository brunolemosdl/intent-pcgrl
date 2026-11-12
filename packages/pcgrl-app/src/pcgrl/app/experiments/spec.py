from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from pcgrl.core.settings.schema import RuntimeConfig


class FactorSpec(BaseModel):
    """One ±1-coded factor of a factorial experiment.

    Attributes:
        name: Short factor label (used in effect names).
        overrides_low: Dotted-path config overrides for the -1 level.
        overrides_high: Dotted-path config overrides for the +1 level.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    overrides_low: dict[str, Any] = Field(default_factory=dict)
    overrides_high: dict[str, Any] = Field(default_factory=dict)


class FactorialSpec(BaseModel):
    """Standalone factorial experiment specification.

    Description:
        Deliberately NOT part of ``RuntimeConfig`` — specs live in their own
        YAML files and apply dotted-path overrides onto a base config via
        ``model_copy``, so no loader flat-dict changes are needed.

    Attributes:
        name: Experiment identifier (output directory stem).
        kind: ``training`` (each cell retrains) or ``inference`` (no retraining).
        training: Training tier used by training-kind cells.
        replicates: Replicates per cell (never below 2 — no error term otherwise).
        episodes: Evaluation episodes per (cell, suite case).
        base_overrides: Constant overrides applied to every cell.
        factors: Two to six ±1-coded factors. Override keys are dotted config
            paths; keys under the ``param.`` prefix are runner-local cell
            parameters (e.g. ``param.description_style``), not config paths.
        responses: Response column names aggregated per cell.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    kind: Literal["training", "inference"] = "training"
    training: str = "quick"
    replicates: int = 2
    episodes: int = 3
    # Constant dotted-path overrides applied to EVERY cell (e.g. disabling the
    # bridge for factorial evaluations).
    base_overrides: dict[str, Any] = Field(default_factory=dict)
    factors: list[FactorSpec]
    responses: list[str] = Field(default_factory=lambda: ["intent_error_mean", "terminal_reward"])

    @model_validator(mode="after")
    def validate_spec(self) -> FactorialSpec:
        """Validate factor count and replicate floor."""
        if not 2 <= len(self.factors) <= 6:
            raise ValueError("factorial specs support 2..6 factors")
        if self.replicates < 2:
            raise ValueError("replicates must be >= 2 (no error term otherwise)")
        names = [factor.name for factor in self.factors]
        if len(set(names)) != len(names):
            raise ValueError("factor names must be unique")
        return self

    @classmethod
    def load(cls, path: Path) -> FactorialSpec:
        """Load and validate a spec YAML file."""
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(payload)


def split_cell_params(overrides: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split runner-local ``param.*`` keys from dotted config overrides.

    Args:
        overrides: Merged cell overrides.

    Returns:
        Tuple of (config overrides, runner params without the prefix).
    """
    config_overrides = {k: v for k, v in overrides.items() if not k.startswith("param.")}
    params = {k.removeprefix("param."): v for k, v in overrides.items() if k.startswith("param.")}
    return config_overrides, params


def apply_overrides(config: RuntimeConfig, overrides: dict[str, Any]) -> RuntimeConfig:
    """Apply dotted-path overrides onto a frozen config via ``model_copy``.

    Args:
        config: Base runtime configuration.
        overrides: Mapping like ``{"reward.composer.weight_style": 0.0}``.

    Returns:
        New config with every override applied (``model_copy`` does not
        re-run validators; spec values are trusted experiment levels).

    Raises:
        AttributeError: When a dotted path does not exist on the config.
    """
    updated = config
    for path, value in overrides.items():
        updated = _apply_one(updated, path.split("."), value)
    return updated


def _apply_one(node: Any, parts: list[str], value: Any) -> Any:
    """Recursively rebuild frozen pydantic models along one dotted path."""
    field = parts[0]
    if not hasattr(node, field):
        raise AttributeError(f"config has no field {field!r} along override path")
    if len(parts) == 1:
        return node.model_copy(update={field: value})
    child = _apply_one(getattr(node, field), parts[1:], value)
    return node.model_copy(update={field: child})
