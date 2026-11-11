from __future__ import annotations

import os
from datetime import UTC, datetime

from pcgrl.app.intent.prompt import PROMPT_VERSION, build_system_prompt
from pcgrl.app.intent.types import (
    IntentTranslation,
    TranslationError,
    TranslatorUnavailableError,
    clamp_vector,
)
from pcgrl.core.settings.schema import RuntimeConfig

_DEFAULT_KEY_ENV = "OPENAI_API_KEY"
_LOCAL_PLACEHOLDER_KEY = "not-needed"


class OpenAICompatibleTranslator:
    """Description -> intent translator over the OpenAI Chat Completions API.

    Description:
        One backend covers every provider that speaks the OpenAI wire format:
        OpenAI itself, Gemini through its OpenAI-compatible endpoint, and
        local servers (Ollama, LM Studio, vLLM) via ``translator.base_url``.
        Structured outputs go through ``chat.completions.parse`` with the
        ``TranslatedIntent`` schema, so the response is the same validated
        set of named floats expected from every backend. No token cap is
        sent — providers disagree on the parameter name (``max_tokens`` vs
        ``max_completion_tokens``) and the response is a tiny JSON object.
    """

    def __init__(self, config: RuntimeConfig) -> None:
        """Build the client against the configured endpoint.

        Args:
            config: Runtime settings with the ``translator`` block.

        Raises:
            TranslatorUnavailableError: When the ``openai`` package is missing,
                or the API key is absent for a non-local endpoint.
        """
        try:
            import openai
        except ImportError as exc:
            raise TranslatorUnavailableError(
                "the 'openai' package is not installed; "
                "install it with: pip install 'pcgrl-app[llm]'"
            ) from exc
        settings = config.translator
        key_env = settings.api_key_env or _DEFAULT_KEY_ENV
        api_key = os.environ.get(key_env)
        if not api_key:
            if settings.base_url is None:
                raise TranslatorUnavailableError(
                    f"{key_env} is not set; export it, point translator.base_url "
                    "at a local server, or use translator.provider=fake"
                )
            api_key = _LOCAL_PLACEHOLDER_KEY
        self._openai = openai
        self._client = openai.OpenAI(api_key=api_key, base_url=settings.base_url)
        self._settings = settings
        self._system_prompt = build_system_prompt(config, grounding=settings.grounding)

    @property
    def system_prompt(self) -> str:
        """Return the grounded system prompt (exposed for prompt tests)."""
        return self._system_prompt

    def translate(self, description: str) -> IntentTranslation:
        """Translate one description through the structured-output endpoint.

        Args:
            description: Natural-language level description.

        Returns:
            Clamped, provenance-stamped translation record.

        Raises:
            TranslationError: On API errors, refusals, or unparseable output.
        """
        from pcgrl.app.intent.types import TranslatedIntent

        completions = self._client.chat.completions
        parse_fn = getattr(completions, "parse", None)
        if parse_fn is None:
            parse_fn = self._client.beta.chat.completions.parse
        try:
            response = parse_fn(
                model=self._settings.model,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": description},
                ],
                response_format=TranslatedIntent,
            )
        except self._openai.OpenAIError as exc:
            raise TranslationError(f"OpenAI-compatible API error: {exc}") from exc
        message = response.choices[0].message
        if getattr(message, "refusal", None):
            raise TranslationError(
                f"the model refused to translate the description: {description!r}"
            )
        parsed = message.parsed
        if parsed is None:
            raise TranslationError("structured output missing from the response")
        vector, clamped = clamp_vector(parsed.vector())
        return IntentTranslation(
            description=description,
            vector=vector,
            source="openai_compatible",
            model=self._settings.model,
            prompt_version=PROMPT_VERSION,
            clamped=clamped,
            rationale=parsed.rationale,
            created_at=datetime.now(tz=UTC).isoformat(),
        )
