from __future__ import annotations

import json

import numpy as np

from pcgrl.core.helpers.path import resolve_path
from pcgrl.core.settings.schema import RuntimeConfig

_INTENSITY_PHRASES = ("no {}", "few {}", "some {}", "many {}", "as many {} as possible")
_INTENSITY_CENTERS = (0.05, 0.20, 0.40, 0.70, 0.95)
_INTENSITY_THRESHOLDS = (0.125, 0.30, 0.55, 0.825)

_COUNT_NOUNS = {
    "enemy_density": "enemies",
    "coin_density": "coins",
    "interactive_block_density": "blocks",
    "pipe_density": "pipes",
    "gap_difficulty": "gaps",
    "platform_usage": "platforms",
}

_VERTICALITY_PHRASES = ("a flat path", "a hilly path", "a mountainous path")
_SMOOTHNESS_PHRASES = ("a rough path", "some smoothness", "a smooth path")
_SMOOTHNESS_THRESHOLDS = (0.30, 0.70)
_DEFAULT_VERTICALITY_THRESHOLDS = (0.40, 0.85)


def verticality_thresholds(config: RuntimeConfig) -> tuple[float, float]:
    """Three-band verticality thresholds anchored on the viability quantiles.

    Description:
        Corpus verticality is compressed into roughly ``[0.67, 1]``, so fixed
        thresholds would put almost every target in the top band. The p25/p75
        of the *achievable* verticality (viability map) split the bands where
        the composer can actually move — clamped so the fake translator's
        phrase values (0.05 / 0.70 / 0.95) still land inside their own band,
        which is what keeps the ``fake(verbalize(v))`` round-trip property true.

    Args:
        config: Runtime settings locating ``intent_viability.json``.

    Returns:
        Tuple of (low/mid threshold, mid/high threshold).
    """
    path = resolve_path(config.library.output_root) / "intent_viability.json"
    if not path.is_file():
        return _DEFAULT_VERTICALITY_THRESHOLDS
    payload = json.loads(path.read_text(encoding="utf-8"))
    stats = payload.get("quantiles", {}).get("verticality")
    if not isinstance(stats, dict):
        return _DEFAULT_VERTICALITY_THRESHOLDS
    low = float(np.clip(float(stats.get("p25", 0.4)), 0.10, 0.65))
    high = float(np.clip(float(stats.get("p75", 0.85)), 0.75, 0.94))
    return (low, high)


def band_index(dim: str, value: float, config: RuntimeConfig) -> int:
    """Return the verbalizer band index for one dimension value.

    Args:
        dim: Intent dimension name.
        value: Value in ``[0, 1]``.
        config: Runtime settings (for the verticality thresholds).

    Returns:
        Band index (5 bands for count dims, 3 for the path dims).
    """
    if dim == "verticality":
        thresholds: tuple[float, ...] = verticality_thresholds(config)
    elif dim == "path_smoothness":
        thresholds = _SMOOTHNESS_THRESHOLDS
    else:
        thresholds = _INTENSITY_THRESHOLDS
    return int(np.searchsorted(np.asarray(thresholds), value, side="right"))


def band_phrase(dim: str, band: int) -> str:
    """Return the canonical phrase for one dimension band.

    Args:
        dim: Intent dimension name.
        band: Band index from ``band_index``.

    Returns:
        Clause-level English phrase the fake translator can invert.

    Raises:
        KeyError: For unknown dimension names.
    """
    if dim == "verticality":
        return _VERTICALITY_PHRASES[band]
    if dim == "path_smoothness":
        return _SMOOTHNESS_PHRASES[band]
    return _INTENSITY_PHRASES[band].format(_COUNT_NOUNS[dim])


def verbalize(vector: np.ndarray | list[float], config: RuntimeConfig) -> str:
    """Render an intent vector as a canonical English description.

    Description:
        Uses the same banded phrase tables the fake translator parses, which
        makes the templated description suite self-consistent: the reference
        vector is known by construction and ``fake(verbalize(v))`` recovers
        each dimension's band.

    Args:
        vector: Intent vector in ``field_names`` order.
        config: Runtime settings for dimension names and thresholds.

    Returns:
        One-sentence level description covering all eight dimensions.
    """
    names = config.target_vector_field_names()
    phrases = [
        band_phrase(name, band_index(name, float(value), config))
        for name, value in zip(names, vector, strict=True)
    ]
    return "A level with " + ", ".join(phrases) + "."


def band_center(dim: str, band: int, config: RuntimeConfig) -> float:
    """Return a representative value inside one band.

    Args:
        dim: Intent dimension name.
        band: Band index.
        config: Runtime settings (for the verticality thresholds).

    Returns:
        The band's canonical center used as the reference target.
    """
    if dim == "verticality":
        low, high = verticality_thresholds(config)
        centers = (low / 2.0, (low + high) / 2.0, (high + 1.0) / 2.0)
        return centers[band]
    if dim == "path_smoothness":
        centers = (0.15, 0.50, 0.85)
        return centers[band]
    return _INTENSITY_CENTERS[band]
