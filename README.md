# L0 Draft Engine

Private two-track drafting service with the existing `POST /v1/draft` multipart contract and `GET /health` endpoint.

## Self-host on Linux

Install Docker Engine, Docker Compose, an NVIDIA driver, and NVIDIA Container Toolkit. Then run:

```sh
docker compose up --build --detach
```

The first draft downloads the original public models and stores them in the persistent `model-cache` volume:

- `salute-developers/GigaAM`, model `v3_ctc`
- `kontur-ai/sbert_punc_case_ru`

The service is published only at `127.0.0.1:8767`. Check it with `docker compose ps` or `curl --fail http://127.0.0.1:8767/health`.

`POST /v1/draft` accepts one `payload` JSON multipart field plus exactly the two mono WAV fields declared by `payload.tracks`; callers must send `X-Babel-Local-Engine: 1`. Responses retain the canonical ordered draft-row schema.

## Private model overrides

No model weights are included in this repository or image. Copy private checkpoints or model directories into `./models`, copy `.env.example` to `.env`, and set container paths such as:

```env
LOCAL_ENGINE_GIGAAM_MODEL=/models/my-gigaam.ckpt
LOCAL_ENGINE_PUNCTUATION_MODEL=/models/my-punctuation-model
```

The model mount is read-only. Hugging Face/GigaAM downloads persist in `model-cache`; multipart scratch space uses a separate named volume and completed requests are removed by the service. `raw` preprocessing is the default; `afftdn` remains an explicit opt-in.
