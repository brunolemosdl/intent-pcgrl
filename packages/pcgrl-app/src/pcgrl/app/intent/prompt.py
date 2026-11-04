from __future__ import annotations

import json

from pcgrl.core.helpers.path import resolve_path
from pcgrl.core.settings.schema import RuntimeConfig

# Bump when the prompt text or semantics change so stale store entries are not
# replayed under the new prompt.
PROMPT_VERSION = "v1"

_GLOSSARY = """\
You translate a natural-language description of a Super Mario Bros level into \
an 8-dimensional intent vector. Every dimension is a float in [0, 1] where 0 \
means "as little as possible" and 1 means "as much as possible":

- enemy_density: how many enemies appear along the level.
- coin_density: how many collectible coins appear.
- interactive_block_density: how many brick / question blocks appear.
- pipe_density: how many pipes appear.
- gap_difficulty: how wide and frequent the floor gaps (pits) are.
- verticality: how much the traversal path moves up and down (hills, \
platforms, stairs); 0 is a flat path.
- path_smoothness: how even the traversal path is; 1 means few abrupt \
height changes.
- platform_usage: how much of the route relies on floating platforms \
instead of the ground.

Rules:
- Set ONLY the dimensions the description mentions (directly or by clear \
implication, e.g. "flat" implies low verticality and high path_smoothness).
- For every dimension the description does NOT mention, use exactly the \
corpus-typical default listed below.
- Intensity mapping: none/no -> ~0.05, few/rare -> ~0.2, some/moderate -> \
~0.4, many/frequent -> ~0.7, maximum/extreme -> ~0.95.\
"""


def _defaults_section(config: RuntimeConfig) -> str:
    """Render the unmentioned-dimension defaults from the config."""
    names = config.target_vector_field_names()
    defaults = config.default_target_vector()
    lines = [
        f"- {name}: {value:.2f}"
        for name, value in zip(names, defaults, strict=True)
    ]
    return "Corpus-typical defaults for unmentioned dimensions:\n" + "\n".join(lines)


def _grounding_sections(config: RuntimeConfig) -> list[str]:
    """Render corpus bands and viability quantiles when the artifacts exist.

    Description:
        Reads ``intent_viability.json`` (per-dimension achievable quantiles
        from random-legal rollouts) and ``corpus_profile.json`` (raw per-tile
        density percentiles). Both reads are deterministic — fixed key order
        and fixed float formatting — so the prompt is byte-stable and cacheable.
    """
    sections: list[str] = []
    viability_path = resolve_path(config.library.output_root) / "intent_viability.json"
    if viability_path.is_file():
        payload = json.loads(viability_path.read_text(encoding="utf-8"))
        quantiles = payload.get("quantiles", {})
        lines = []
        for name in config.target_vector_field_names():
            stats = quantiles.get(name)
            if not isinstance(stats, dict):
                continue
            lines.append(
                f"- {name}: p05={float(stats.get('p05', 0.0)):.2f} "
                f"p50={float(stats.get('p50', 0.0)):.2f} "
                f"p95={float(stats.get('p95', 1.0)):.2f}"
            )
        if lines:
            sections.append(
                "Achievable range per dimension (quantiles of what the level "
                "composer can actually produce; keep targets inside p05..p95):\n"
                + "\n".join(lines)
            )
    profile_path = resolve_path(config.dataset.processed_root) / "corpus_profile.json"
    if profile_path.is_file():
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
        density = payload.get("density", {})
        lines = []
        for category in sorted(density):
            stats = density[category]
            if not isinstance(stats, dict):
                continue
            lines.append(
                f"- {category}: p10={float(stats.get('p10', 0.0)):.4f} "
                f"p50={float(stats.get('p50', 0.0)):.4f} "
                f"p90={float(stats.get('p90', 0.0)):.4f}"
            )
        if lines:
            sections.append(
                "Raw corpus tile densities (fraction of level area, for "
                "calibration context only):\n" + "\n".join(lines)
            )
    return sections


def build_system_prompt(config: RuntimeConfig, *, grounding: bool = True) -> str:
    """Build the deterministic system prompt for the intent translator.

    Args:
        config: Runtime settings for defaults and artifact paths.
        grounding: When ``False``, return the zero-shot glossary-only prompt
            (the ``grounding: false`` factor level of the inference factorial).

    Returns:
        Byte-stable prompt string for the given config and artifacts.
    """
    sections = [_GLOSSARY, _defaults_section(config)]
    if grounding:
        sections.extend(_grounding_sections(config))
    sections.append(
        "Respond with the eight floats (and an optional short rationale) in "
        "the requested structured format."
    )
    return "\n\n".join(sections)
