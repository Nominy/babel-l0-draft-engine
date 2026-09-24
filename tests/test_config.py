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


@pytest.mark.parametrize(
    ("system", "machine", "expected"),
    [
        ("Darwin", "arm64", "mps"),
        ("Darwin", "aarch64", "mps"),
        ("Darwin", "x86_64", "cuda"),
        ("Linux", "aarch64", "cuda"),
        ("Windows", "AMD64", "cuda"),
    ],
)
def test_device_default_selects_metal_only_on_apple_silicon(
    monkeypatch, system: str, machine: str, expected: str
) -> None:
    monkeypatch.delenv("LOCAL_ENGINE_DEVICE", raising=False)
    monkeypatch.setattr("l0_draft_engine.config.platform.system", lambda: system)
    monkeypatch.setattr("l0_draft_engine.config.platform.machine", lambda: machine)

    assert Settings().device == expected
    assert Settings.from_env().device == expected


@pytest.mark.parametrize(
    ("system", "machine", "device"),
    [
        ("Darwin", "arm64", "cpu"),
        ("Darwin", "arm64", "cuda"),
        ("Linux", "x86_64", "mps"),
    ],
)
def test_explicit_device_overrides_platform_default(
    monkeypatch, system: str, machine: str, device: str
) -> None:
    monkeypatch.setattr("l0_draft_engine.config.platform.system", lambda: system)
    monkeypatch.setattr("l0_draft_engine.config.platform.machine", lambda: machine)
    monkeypatch.setenv("LOCAL_ENGINE_DEVICE", f" {device.upper()} ")

    assert Settings(device=device).device == device
    assert Settings.from_env().device == device


def test_device_rejects_unsupported_backend(monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_ENGINE_DEVICE", "xpu")
    with pytest.raises(SettingsError):
        Settings.from_env()
    with pytest.raises(SettingsError):
        Settings(device="xpu")


@pytest.mark.parametrize("value", [0, 86400])
def test_model_idle_timeout_accepts_disabled_and_upper_boundary(
    monkeypatch, value: int
) -> None:
    monkeypatch.setenv("LOCAL_ENGINE_MODEL_IDLE_SECONDS", str(value))

    assert Settings.from_env().model_idle_seconds == value
    assert Settings(model_idle_seconds=value).model_idle_seconds == value


def test_model_idle_timeout_accepts_environment_override(monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_ENGINE_MODEL_IDLE_SECONDS", "17")

    assert Settings.from_env().model_idle_seconds == 17


@pytest.mark.parametrize("value", ["-1", "86401", "not-an-integer", "1.5"])
def test_model_idle_timeout_rejects_invalid_environment_values(
    monkeypatch, value: str
) -> None:
    monkeypatch.setenv("LOCAL_ENGINE_MODEL_IDLE_SECONDS", value)

    with pytest.raises(SettingsError):
        Settings.from_env()


@pytest.mark.parametrize("value", [-1, 86401, 1.5, True, "17"])
def test_model_idle_timeout_validates_direct_settings(value) -> None:
    with pytest.raises(SettingsError):
        Settings(model_idle_seconds=value)
