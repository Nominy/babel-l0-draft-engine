# L0 Draft Engine

## Linux: Docker

Install Docker Engine, Docker Compose, an NVIDIA driver, and NVIDIA Container Toolkit. From this directory:

```sh
docker compose up --build --detach
curl --fail http://127.0.0.1:8767/health
docker compose logs --follow
```

The service binds only to `127.0.0.1:8767`. Models download on first use: GigaAM `v3_ctc` and `kontur-ai/sbert_punc_case_ru`. Downloads persist in the `model-cache` volume; local `models/` is mounted read-only at `/models`.

## Windows: Python

Install 64-bit Python 3.11 and a current NVIDIA driver. From PowerShell in this directory:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\Install-Windows.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\Start-Windows.ps1
```

The installer creates `.venv`, installs CUDA-enabled PyTorch and the engine, and creates `.env` from `.env.example` if absent. The launcher reads `.env`, checks CUDA, and starts `http://127.0.0.1:8767`; downloads stay under `.cache`.

From a second terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8767/health
```

Installer options: `-PythonExecutable C:\path\to\python.exe`, `-TorchIndexUrl <official-wheel-index>` (default CUDA 12.8), `-Recreate` (replaces `.venv`). Launcher options: `-PreflightOnly`, `-Port <port>`.

## Configuration

Copy `.env.example` to `.env` before customizing Docker settings. No model weights ship in the repository or image. For private checkpoints, place them in `models/` and set:

```env
LOCAL_ENGINE_GIGAAM_MODEL=/models/my-gigaam.ckpt
LOCAL_ENGINE_PUNCTUATION_MODEL=/models/my-punctuation-model
```

On Windows, use absolute Windows paths instead. `LOCAL_ENGINE_MAX_INFLIGHT_REQUESTS` accepts 1–64 (default 3, including the running request).

Endpoints: `/health`, `/v1/draft`, `/v1/transcribe`. Inference requests require `X-Babel-Local-Engine: 1` and multipart fields containing a `payload` JSON object and its two declared mono WAV tracks. Drafting returns transcript rows; transcription returns per-lane word timings. Model downloads require network access on first use.
