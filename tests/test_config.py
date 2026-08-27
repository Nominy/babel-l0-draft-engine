from l0_draft_engine.config import Settings


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
