import pytest

from l0_draft_engine.config import Settings, SettingsError


def test_upstream_models_and_raw_preprocessing_are_defaults(monkeypatch) -> None:
    for name in (
        "LOCAL_ENGINE_GIGAAM_MODEL",
        "LOCAL_ENGINE_PUNCTUATION_MODEL",
        "LOCAL_ENGINE_PREPROCESSING",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings.from_env()

    assert settings.gigaam_model_path == "v3_ctc"
    assert settings.punctuation_model_path == "kontur-ai/sbert_punc_case_ru"
    assert settings.preprocessing == "raw"


def test_max_inflight_requests_defaults_to_three_and_accepts_env_override(
    monkeypatch,
) -> None:
    monkeypatch.delenv("LOCAL_ENGINE_MAX_INFLIGHT_REQUESTS", raising=False)
    assert Settings.from_env().max_inflight_requests == 3

    monkeypatch.setenv("LOCAL_ENGINE_MAX_INFLIGHT_REQUESTS", "7")
    assert Settings.from_env().max_inflight_requests == 7


@pytest.mark.parametrize("value", ["0", "65", "not-an-integer"])
def test_max_inflight_requests_rejects_invalid_env_values(
    monkeypatch, value: str
) -> None:
    monkeypatch.setenv("LOCAL_ENGINE_MAX_INFLIGHT_REQUESTS", value)

    with pytest.raises(SettingsError):
        Settings.from_env()


def test_max_inflight_requests_is_validated_for_direct_settings() -> None:
    with pytest.raises(
        SettingsError,
        match="LOCAL_ENGINE_MAX_INFLIGHT_REQUESTS must be between 1 and 64",
    ):
        Settings(max_inflight_requests=0)
