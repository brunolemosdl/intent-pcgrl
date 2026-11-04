from __future__ import annotations

from pcgrl.core.helpers.path import resolve_path
from pcgrl.core.settings.schema import RuntimeConfig
from pcgrl.core.tiles.catalog import load_catalog
from pcgrl.runtime.environment.composer import MarioComposerEnv
from pcgrl.runtime.environment.context import EnvContext
from pcgrl.runtime.validators.runner import ValidatorRunner


def build_env_context(config: RuntimeConfig) -> EnvContext:
    """Assemble the construction context for the composer environment.

    Args:
        config: Runtime settings for library paths and reward weights.

    Returns:
        Fully wired ``EnvContext`` for ``make_env``.

    Raises:
        FileNotFoundError: When the chunk library or pattern reference is
            missing; run the ``library`` and ``prepare`` stages first.
    """
    from pcgrl.core.library.types import ChunkLibrary
    from pcgrl.runtime.environment.viability import IntentViabilityMap
    from pcgrl.runtime.metrics.tpkl import PatternReference
    from pcgrl.runtime.reward.composer_terminal import ComposerTerminalReward

    catalog = load_catalog(config)
    library_root = resolve_path(config.library.output_root)
    library = ChunkLibrary.load(library_root)
    reference = PatternReference.load(
        resolve_path(config.dataset.processed_root), split="train"
    )
    terminal = ComposerTerminalReward(config, catalog, reference)
    try:
        viability = IntentViabilityMap.load(library_root)
    except FileNotFoundError:
        viability = None
    return EnvContext(
        config=config,
        catalog=catalog,
        validator=ValidatorRunner(config, catalog),
        chunk_library=library,
        pattern_reference=reference,
        composer_terminal=terminal,
        viability_map=viability,
    )


def make_env(
    config: RuntimeConfig, context: EnvContext, seed: int | None = None
) -> MarioComposerEnv:
    """Instantiate the composer environment.

    Args:
        config: Runtime settings (kept for signature stability).
        context: Pre-built construction context from ``build_env_context``.
        seed: Optional RNG seed forwarded to the environment.

    Returns:
        Seeded ``MarioComposerEnv``.
    """
    return MarioComposerEnv(context, seed=seed)
