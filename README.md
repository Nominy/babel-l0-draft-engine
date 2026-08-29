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

## Self-host on Windows

Install 64-bit Python 3.11 and a current NVIDIA driver. From PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\Install-Windows.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\Start-Windows.ps1
```

The installer creates an isolated `.venv`, installs CUDA-enabled PyTorch and the engine, copies `.env.example` to `.env` when needed, and verifies that CUDA is available. The launcher imports `.env`, keeps model downloads under `.cache`, runs a dependency/GPU preflight, and serves only `http://127.0.0.1:8767`.

Verify it from a second PowerShell window:

```powershell
Invoke-RestMethod http://127.0.0.1:8767/health
```

Use `Start-Windows.ps1 -PreflightOnly` to check the environment without starting the server. Pass `-PythonExecutable C:\path\to\python.exe` to the installer when the desired Python is not first on `PATH`. The default PyTorch wheel index targets CUDA 12.8, which supports current RTX generations; override `-TorchIndexUrl` when a different official PyTorch wheel channel is required.


## Private model overrides

No model weights are included in this repository or image. Copy private checkpoints or model directories into `./models`, copy `.env.example` to `.env`, and configure either container paths or absolute Windows paths:

```env
# Docker
LOCAL_ENGINE_GIGAAM_MODEL=/models/my-gigaam.ckpt
LOCAL_ENGINE_PUNCTUATION_MODEL=/models/my-punctuation-model

# Windows
LOCAL_ENGINE_GIGAAM_MODEL=C:\models\my-gigaam.ckpt
LOCAL_ENGINE_PUNCTUATION_MODEL=C:\models\my-punctuation-model
```

Docker mounts `./models` read-only and persists Hugging Face/GigaAM downloads in the `model-cache` volume. Windows keeps downloads under `.cache`; both runtimes remove completed multipart scratch files. `raw` remains the default ASR input; S2 activity detection always uses a separately denoised `afftdn` lane. Choosing `afftdn` also uses that denoised lane for ASR.
