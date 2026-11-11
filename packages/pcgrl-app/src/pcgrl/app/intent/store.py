from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from pcgrl.app.intent.types import IntentTranslation


def translation_key(model: str | None, prompt_version: str, description: str) -> str:
    """Stable store key for one (model, prompt, description) triple."""
    payload = f"{model or 'rule-based'}|{prompt_version}|{description}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class TranslationStore:
    """JSON-file store mapping translation keys to recorded translations.

    Description:
        One online pass records translations; every downstream analysis
        replays them offline and reproducibly. Writes are atomic (temp file +
        rename) so a crashed run never corrupts the committed store.
    """

    def __init__(self, path: Path) -> None:
        """Bind the store file path and load any existing entries."""
        self._path = Path(path)
        self._entries: dict[str, dict[str, object]] = {}
        if self._path.is_file():
            self._entries = json.loads(self._path.read_text(encoding="utf-8"))

    def __len__(self) -> int:
        """Number of recorded translations."""
        return len(self._entries)

    def get(self, key: str) -> IntentTranslation | None:
        """Return the recorded translation for ``key``, or ``None``."""
        raw = self._entries.get(key)
        if raw is None:
            return None
        return IntentTranslation.model_validate(raw)

    def put(self, key: str, translation: IntentTranslation) -> None:
        """Record one translation and persist the store atomically."""
        self._entries[key] = translation.model_dump(mode="json")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        handle, tmp_name = tempfile.mkstemp(
            dir=self._path.parent, prefix=self._path.name, suffix=".tmp"
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as tmp:
                json.dump(self._entries, tmp, indent=2, sort_keys=True)
            os.replace(tmp_name, self._path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
