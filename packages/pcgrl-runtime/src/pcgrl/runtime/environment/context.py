from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pcgrl.core.settings.schema import RuntimeConfig
from pcgrl.core.tiles.catalog import TileCatalog
from pcgrl.runtime.validators.runner import ValidatorRunner

if TYPE_CHECKING:
    from pcgrl.core.library.types import ChunkLibrary
    from pcgrl.runtime.environment.viability import IntentViabilityMap
    from pcgrl.runtime.metrics.tpkl import PatternReference
    from pcgrl.runtime.reward.composer_terminal import ComposerTerminalReward


@dataclass
class EnvContext:
    """Shared construction context for the composer environment.

    Attributes:
        config: Runtime configuration.
        catalog: Tile catalog for the active area type.
        validator: Structural validation runner.
        chunk_library: Annotated chunk library.
        pattern_reference: Corpus tile-pattern distributions for the style term.
        composer_terminal: Terminal reward for chunk composition.
        viability_map: Empirical reachable-intent cloud (optional).
    """

    config: RuntimeConfig
    catalog: TileCatalog
    validator: ValidatorRunner
    chunk_library: ChunkLibrary | None = None
    pattern_reference: PatternReference | None = None
    composer_terminal: ComposerTerminalReward | None = None
    viability_map: IntentViabilityMap | None = None
