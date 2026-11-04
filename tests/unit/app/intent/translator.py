from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest
from pcgrl.app.intent.fake import FakeIntentTranslator
from pcgrl.app.intent.prompt import PROMPT_VERSION, build_system_prompt
from pcgrl.app.intent.recorded import RecordedTranslator
from pcgrl.app.intent.store import TranslationStore, translation_key
from pcgrl.app.intent.types import (
    IntentTranslation,
    TranslatedIntent,
    TranslationError,
    clamp_vector,
)
from pcgrl.core.settings.schema import IntentTargetConfig


def _dims() -> tuple[str, ...]:
    """Intent dimension names in vector order."""
    return IntentTargetConfig.field_names()


def test_translated_intent_field_order_matches_target_config() -> None:
    """The LLM output schema must carry the intent dims in vector order."""
    schema_fields = [
        name for name in TranslatedIntent.model_fields if name != "rationale"
    ]
    assert tuple(schema_fields) == _dims()


def test_clamp_vector_flags_out_of_range_values() -> None:
    """Values outside [0, 1] clamp and raise the flag."""
    vector, clamped = clamp_vector((0.5, -0.2, 1.7))
    assert vector == (0.5, 0.0, 1.0)
    assert clamped is True
    vector, clamped = clamp_vector((0.0, 1.0, 0.3))
    assert clamped is False


def test_fake_translator_moves_mentioned_dimensions(config) -> None:
    """'many enemies, no gaps, flat easy level' -> enemy up, gaps down."""
    translation = FakeIntentTranslator(config).translate(
        "many enemies, no gaps, flat easy level"
    )
    values = dict(zip(_dims(), translation.vector, strict=True))
    defaults = dict(zip(_dims(), config.default_target_vector(), strict=True))
    assert values["enemy_density"] > defaults["enemy_density"]
    assert values["gap_difficulty"] < defaults["gap_difficulty"]
    assert values["verticality"] <= 0.1
    assert values["path_smoothness"] >= 0.9
    assert values["coin_density"] == pytest.approx(defaults["coin_density"])
    assert translation.source == "fake"
    assert len(translation.vector) == config.target_vector_dim


def test_fake_translator_intensity_bands(config) -> None:
    """Intensity words map to their bands; extremes beat the bare 'many'."""
    translator = FakeIntentTranslator(config)
    cases = {
        "no coins": 0.05,
        "few coins": 0.20,
        "some coins": 0.40,
        "many coins": 0.70,
        "as many coins as possible": 0.95,
    }
    coin_index = _dims().index("coin_density")
    for description, expected in cases.items():
        assert translator.translate(description).vector[coin_index] == pytest.approx(
            expected
        ), description


def test_fake_translator_is_deterministic(config) -> None:
    """Same description -> same vector, every time."""
    translator = FakeIntentTranslator(config)
    first = translator.translate("many pipes and few enemies").vector
    second = translator.translate("many pipes and few enemies").vector
    assert first == second


def test_prompt_is_byte_deterministic(config) -> None:
    """The grounded prompt must be byte-stable for cacheability."""
    assert build_system_prompt(config) == build_system_prompt(config)
    grounded = build_system_prompt(config, grounding=True)
    zero_shot = build_system_prompt(config, grounding=False)
    assert "enemy_density" in zero_shot
    assert len(grounded) >= len(zero_shot)


def _sample_translation(description: str = "many enemies") -> IntentTranslation:
    """Minimal valid translation record for store tests."""
    return IntentTranslation(
        description=description,
        vector=tuple([0.5] * 8),
        source="fake",
        model=None,
        prompt_version=PROMPT_VERSION,
        clamped=False,
        rationale=None,
        created_at="2026-06-10T00:00:00+00:00",
    )


def test_store_record_replay_roundtrip(tmp_path) -> None:
    """A recorded translation replays identically from a fresh store."""
    store_path = tmp_path / "store.json"
    store = TranslationStore(store_path)
    key = translation_key(None, PROMPT_VERSION, "many enemies")
    store.put(key, _sample_translation())
    reloaded = TranslationStore(store_path)
    assert len(reloaded) == 1
    hit = reloaded.get(key)
    assert hit is not None
    assert hit.vector == tuple([0.5] * 8)


def test_recorded_translator_records_and_replays(config, tmp_path) -> None:
    """Record mode fills the store; replay mode serves it without the inner."""
    store = TranslationStore(tmp_path / "store.json")
    recorder = RecordedTranslator(
        store,
        mode="record",
        inner=FakeIntentTranslator(config),
        model=None,
        prompt_version=PROMPT_VERSION,
    )
    recorded = recorder.translate("many enemies")
    replayer = RecordedTranslator(
        TranslationStore(tmp_path / "store.json"),
        mode="replay",
        inner=None,
        model=None,
        prompt_version=PROMPT_VERSION,
    )
    assert replayer.translate("many enemies").vector == recorded.vector


def test_recorded_translator_replay_miss_fails_loudly(tmp_path) -> None:
    """A replay-mode store miss raises instead of calling any backend."""
    replayer = RecordedTranslator(
        TranslationStore(tmp_path / "store.json"),
        mode="replay",
        inner=None,
        model="gpt-5.2",
        prompt_version=PROMPT_VERSION,
    )
    with pytest.raises(TranslationError, match="no recorded translation"):
        replayer.translate("never seen before")


def test_factory_fake_provider(config, tmp_path) -> None:
    """The factory wires the fake provider through the record store."""
    from pcgrl.app.intent.factory import build_translator

    runtime = config.model_copy(
        update={
            "translator": config.translator.model_copy(
                update={
                    "provider": "fake",
                    "mode": "record",
                    "store_path": str(tmp_path / "store.json"),
                }
            )
        }
    )
    translator = build_translator(runtime)
    translation = translator.translate("many enemies")
    assert translation.source == "fake"
    assert (tmp_path / "store.json").is_file()


@pytest.fixture
def stub_openai(monkeypatch):
    """Install a stub 'openai' module and return its client mock."""
    module = types.ModuleType("openai")

    class _OpenAIError(Exception):
        """Stub OpenAIError matching the SDK exception surface."""

    client = MagicMock()
    module.OpenAIError = _OpenAIError
    module.OpenAI = MagicMock(return_value=client)
    monkeypatch.setitem(sys.modules, "openai", module)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return module


def _openai_config(config, **updates):
    """Config copy with the openai_compatible translator provider."""
    translator = config.translator.model_copy(
        update={"provider": "openai_compatible", **updates}
    )
    return config.model_copy(update={"translator": translator})


def test_openai_translator_contract(config, stub_openai) -> None:
    """The call uses chat-completions structured outputs against the schema."""
    from pcgrl.app.intent.openai_translator import OpenAICompatibleTranslator

    runtime = _openai_config(config, model="gpt-5.1")
    parsed = TranslatedIntent(
        enemy_density=0.9,
        coin_density=0.4,
        interactive_block_density=0.5,
        pipe_density=0.3,
        gap_difficulty=0.4,
        verticality=0.75,
        path_smoothness=0.75,
        platform_usage=0.35,
        rationale=None,
    )
    message = MagicMock()
    message.refusal = None
    message.parsed = parsed
    response = MagicMock()
    response.choices = [MagicMock(message=message)]
    client = stub_openai.OpenAI.return_value
    client.chat.completions.parse.return_value = response

    translation = OpenAICompatibleTranslator(runtime).translate("many enemies")
    kwargs = client.chat.completions.parse.call_args.kwargs
    assert kwargs["response_format"] is TranslatedIntent
    assert kwargs["model"] == "gpt-5.1"
    assert kwargs["messages"][0]["role"] == "system"
    assert "temperature" not in kwargs
    assert translation.source == "openai_compatible"
    assert stub_openai.OpenAI.call_args.kwargs["api_key"] == "test-key"


def test_openai_translator_refusal_raises(config, stub_openai) -> None:
    """A refusal message maps to TranslationError."""
    from pcgrl.app.intent.openai_translator import OpenAICompatibleTranslator

    message = MagicMock()
    message.refusal = "cannot help"
    response = MagicMock()
    response.choices = [MagicMock(message=message)]
    client = stub_openai.OpenAI.return_value
    client.chat.completions.parse.return_value = response
    with pytest.raises(TranslationError, match="refused"):
        OpenAICompatibleTranslator(_openai_config(config)).translate("anything")


def test_openai_translator_missing_key_without_base_url(config, stub_openai, monkeypatch) -> None:
    """No key and no local endpoint -> remediation-bearing unavailable error."""
    from pcgrl.app.intent.openai_translator import OpenAICompatibleTranslator
    from pcgrl.app.intent.types import TranslatorUnavailableError

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(TranslatorUnavailableError, match="OPENAI_API_KEY"):
        OpenAICompatibleTranslator(_openai_config(config))


def test_openai_translator_local_server_needs_no_key(config, stub_openai, monkeypatch) -> None:
    """A base_url (local server) works without any API key."""
    from pcgrl.app.intent.openai_translator import OpenAICompatibleTranslator

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    runtime = _openai_config(
        config, base_url="http://localhost:11434/v1", model="qwen3:14b"
    )
    OpenAICompatibleTranslator(runtime)
    kwargs = stub_openai.OpenAI.call_args.kwargs
    assert kwargs["base_url"] == "http://localhost:11434/v1"
    assert kwargs["api_key"] == "not-needed"


def test_openai_translator_custom_key_env(config, stub_openai, monkeypatch) -> None:
    """api_key_env selects an arbitrary provider key variable."""
    from pcgrl.app.intent.openai_translator import OpenAICompatibleTranslator

    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    runtime = _openai_config(
        config,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key_env="GEMINI_API_KEY",
        model="gemini-3-pro-preview",
    )
    OpenAICompatibleTranslator(runtime)
    assert stub_openai.OpenAI.call_args.kwargs["api_key"] == "gemini-key"


def test_factory_openai_provider(config, stub_openai, tmp_path) -> None:
    """The factory wires the openai_compatible provider through the store."""
    from pcgrl.app.intent.factory import build_translator
    from pcgrl.app.intent.recorded import RecordedTranslator

    runtime = _openai_config(
        config, mode="record", store_path=str(tmp_path / "store.json")
    )
    translator = build_translator(runtime)
    assert isinstance(translator, RecordedTranslator)


def test_load_env_file_sets_only_missing_vars(tmp_path, monkeypatch) -> None:
    """The .env loader fills gaps but never overrides the shell."""
    from pcgrl.core.helpers.env import load_env_file

    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\n"
        "PCGRL_TEST_NEW=from-file\n"
        "PCGRL_TEST_EXISTING=from-file\n"
        "PCGRL_TEST_EMPTY=\n"
        'PCGRL_TEST_QUOTED="quoted-value"\n',
        encoding="utf-8",
    )
    monkeypatch.delenv("PCGRL_TEST_NEW", raising=False)
    monkeypatch.setenv("PCGRL_TEST_EXISTING", "from-shell")
    monkeypatch.delenv("PCGRL_TEST_EMPTY", raising=False)
    monkeypatch.delenv("PCGRL_TEST_QUOTED", raising=False)
    loaded = load_env_file(env_file)
    import os

    assert os.environ["PCGRL_TEST_NEW"] == "from-file"
    assert os.environ["PCGRL_TEST_EXISTING"] == "from-shell"
    assert "PCGRL_TEST_EMPTY" not in os.environ
    assert os.environ["PCGRL_TEST_QUOTED"] == "quoted-value"
    assert loaded == 2
    for key in ("PCGRL_TEST_NEW", "PCGRL_TEST_QUOTED"):
        monkeypatch.delenv(key, raising=False)
