import pytest
from pydantic import ValidationError
from l0_draft_engine.inference_release import RELEASE_HEADERS, RELEASE_ID, validate_punctuated_timing
from l0_draft_engine.release_middleware import InferenceReleaseMiddleware
from l0_draft_engine.schemas import TranscriptionResponse


@pytest.mark.asyncio
async def test_outdated_client_rejected_before_body_is_read():
    async def forbidden_receive():
        raise AssertionError("outdated upload body was read")

    async def forbidden_app(*_args):
        raise AssertionError("outdated client reached inference")

    messages = []
    async def send(message):
        messages.append(message)

    scope = {"type": "http", "method": "POST", "path": "/v1/transcribe", "headers": []}
    await InferenceReleaseMiddleware(forbidden_app, True)(scope, forbidden_receive, send)
    assert messages[0]["status"] == 426
    assert RELEASE_ID.encode() in messages[1]["body"]


@pytest.mark.asyncio
async def test_current_client_and_cors_preflight_preserve_streaming():
    calls = []
    async def app(scope, receive, send):
        calls.append((scope, receive, send))

    middleware = InferenceReleaseMiddleware(app, True)
    current = {"type": "http", "method": "POST", "path": "/v1/transcribe",
               "headers": [(key.lower().encode(), value.encode()) for key, value in RELEASE_HEADERS.items()]}
    await middleware(current, None, None)
    preflight = {**current, "method": "OPTIONS", "headers": []}
    await middleware(preflight, None, None)
    assert calls == [(current, None, None), (preflight, None, None)]


def timing_payload(labels, release=RELEASE_ID):
    return {"taskId": "task", "models": {"release": release}, "summary": {}, "tracks": [{
        "lane": "speaker-1", "sampleRate": 16000, "pcmSha256": "a" * 64,
        "tokens": [{"id": "word", "text": "да", "startSeconds": 0, "endSeconds": 1}],
        "segments": [], "punctuationLabels": labels}, {
        "lane": "speaker-2", "sampleRate": 16000, "pcmSha256": "b" * 64,
        "tokens": [], "segments": [], "punctuationLabels": []}]}


@pytest.mark.parametrize("labels", [None, [], [7], [-1], [True], [1.0]])
def test_completed_punctuation_rejects_invalid_label_contract(labels):
    with pytest.raises((ValueError, ValidationError)):
        validate_punctuated_timing(TranscriptionResponse.model_validate(timing_payload(labels)))


def test_completed_punctuation_accepts_current_contract():
    validate_punctuated_timing(TranscriptionResponse.model_validate(timing_payload([2])))


def test_completed_punctuation_rejects_legacy_release():
    timing = TranscriptionResponse.model_validate(timing_payload([2], "legacy"))
    with pytest.raises(ValueError, match="outdated"):
        validate_punctuated_timing(timing)
