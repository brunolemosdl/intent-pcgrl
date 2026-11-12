from __future__ import annotations

from pcgrl.app.intent.fake import FakeIntentTranslator
from pcgrl.app.intent.prompt import PROMPT_VERSION
from pcgrl.app.intent.recorded import RecordedTranslator
from pcgrl.app.intent.store import TranslationStore
from pcgrl.app.intent.types import IntentTranslator
from pcgrl.core.helpers.path import resolve_path
from pcgrl.core.settings.schema import RuntimeConfig


def build_translator(config: RuntimeConfig) -> IntentTranslator:
    """Build the configured description -> intent translator.

    Args:
        config: Runtime settings with the ``translator`` block.

    Returns:
        Translator honouring ``translator.provider`` and ``translator.mode``.

    Raises:
        TranslatorUnavailableError: When the configured provider cannot run.
        ValueError: On unknown provider names.
    """
    settings = config.translator
    inner: IntentTranslator
    model: str | None
    if settings.provider == "fake":
        inner = FakeIntentTranslator(config)
        model = None
    elif settings.provider == "openai_compatible":
        model = settings.model
        if settings.mode == "replay":
            inner = None  # type: ignore[assignment]
        else:
            from pcgrl.app.intent.openai_translator import (
                OpenAICompatibleTranslator,
            )

            inner = OpenAICompatibleTranslator(config)
    else:
        raise ValueError(f"unknown translator provider {settings.provider!r}")
    if settings.mode == "live":
        return inner
    store = TranslationStore(resolve_path(settings.store_path))
    # The store key must distinguish grounded from zero-shot prompts, or a
    # grounding ablation would replay answers produced under the other prompt.
    prompt_version = PROMPT_VERSION if settings.grounding else f"{PROMPT_VERSION}+zeroshot"
    return RecordedTranslator(
        store,
        mode=settings.mode,
        inner=inner,
        model=model,
        prompt_version=prompt_version,
    )
