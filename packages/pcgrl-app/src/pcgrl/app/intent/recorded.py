from __future__ import annotations

from pcgrl.app.intent.store import TranslationStore, translation_key
from pcgrl.app.intent.types import (
    IntentTranslation,
    IntentTranslator,
    TranslationError,
)


class RecordedTranslator:
    """Record/replay wrapper that makes any translator reproducible.

    Description:
        ``record`` mode serves store hits and otherwise calls the inner
        translator and persists the result; ``replay`` mode serves only the
        store and fails loudly on a miss, so offline analyses can never
        silently call the network.
    """

    def __init__(
        self,
        store: TranslationStore,
        *,
        mode: str,
        inner: IntentTranslator | None = None,
        model: str | None = None,
        prompt_version: str,
    ) -> None:
        """Bind the store, replay mode, and optional inner translator.

        Args:
            store: Persistent translation store.
            mode: ``record`` or ``replay``.
            inner: Translator consulted on store misses in record mode.
            model: Model id used for store keys.
            prompt_version: Prompt version used for store keys.

        Raises:
            ValueError: On an unknown mode or a record mode without ``inner``.
        """
        if mode not in ("record", "replay"):
            raise ValueError(f"unknown RecordedTranslator mode {mode!r}")
        if mode == "record" and inner is None:
            raise ValueError("record mode requires an inner translator")
        self._store = store
        self._mode = mode
        self._inner = inner
        self._model = model
        self._prompt_version = prompt_version

    def translate(self, description: str) -> IntentTranslation:
        """Serve from the store, recording through the inner translator if allowed.

        Args:
            description: Natural-language level description.

        Returns:
            Stored or freshly recorded translation.

        Raises:
            TranslationError: On a replay-mode store miss.
        """
        key = translation_key(self._model, self._prompt_version, description)
        hit = self._store.get(key)
        if hit is not None:
            return hit
        if self._mode == "replay" or self._inner is None:
            raise TranslationError(
                f"no recorded translation for {description!r} "
                f"(model={self._model}, prompt={self._prompt_version}); "
                "re-run with translator.mode=record to refresh the store"
            )
        translation = self._inner.translate(description)
        self._store.put(key, translation)
        return translation
