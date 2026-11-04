from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel

from pcgrl.core.settings.schema import IntentTargetConfig, StrictModel


class TranslationError(RuntimeError):
    """The translator failed to produce a usable intent vector."""


class TranslatorUnavailableError(TranslationError):
    """The translator backend cannot run (missing package, key, or store)."""


class TranslatedIntent(BaseModel):
    """LLM output schema: the eight named intent dimensions plus a rationale.

    Description:
        Field order MUST match ``IntentTargetConfig.field_names()`` — the
        schema carries the intent semantics into the structured-output
        request. Values are plain floats because structured outputs cannot
        enforce numeric ranges server-side; the caller clamps to ``[0, 1]``.
    """

    enemy_density: float
    coin_density: float
    interactive_block_density: float
    pipe_density: float
    gap_difficulty: float
    verticality: float
    path_smoothness: float
    platform_usage: float
    rationale: str | None = None

    def vector(self) -> tuple[float, ...]:
        """Return the dimensions as a tuple in ``field_names`` order."""
        return tuple(
            float(getattr(self, name)) for name in IntentTargetConfig.field_names()
        )


class IntentTranslation(StrictModel):
    """Frozen provenance record for one description -> vector translation.

    Attributes:
        description: The natural-language input.
        vector: Clamped 8-d intent vector.
        source: Backend that produced the vector.
        model: LLM model id, or ``None`` for rule-based backends.
        prompt_version: Version tag of the grounded prompt.
        clamped: Whether any raw value fell outside ``[0, 1]``.
        rationale: Optional model-provided reasoning.
        created_at: ISO-8601 creation timestamp.
    """

    description: str
    vector: tuple[float, ...]
    source: Literal["openai_compatible", "fake", "recorded"]
    model: str | None = None
    prompt_version: str
    clamped: bool = False
    rationale: str | None = None
    created_at: str


class IntentTranslator(Protocol):
    """Anything that turns a description into an ``IntentTranslation``."""

    def translate(self, description: str) -> IntentTranslation:
        """Translate one natural-language description into an intent vector."""
        ...


def clamp_vector(values: tuple[float, ...]) -> tuple[tuple[float, ...], bool]:
    """Clamp every value to ``[0, 1]`` and report whether anything moved.

    Args:
        values: Raw vector values.

    Returns:
        Tuple of (clamped vector, clamped flag).
    """
    clamped = tuple(min(1.0, max(0.0, float(v))) for v in values)
    return clamped, clamped != tuple(float(v) for v in values)
