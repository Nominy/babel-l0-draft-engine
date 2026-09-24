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

## macOS: native Apple Silicon Metal

Use an Apple Silicon Mac with macOS 14 or newer, **native arm64 Python 3.11** (the Python 3.11 macOS universal2 installer from python.org works when run natively), and **ffmpeg on PATH**. GigaAM requires ffmpeg even for raw WAV input. Install a native Apple Silicon ffmpeg build; if you use Homebrew, `brew install ffmpeg` supplies it. Run Terminal without Rosetta. No NVIDIA driver, CUDA, or Docker is required; Docker Desktop does not expose the Mac's Metal GPU to this engine.

From this directory:

```sh
./scripts/Install-Mac.sh
./scripts/Start-Mac.sh --preflight-only
./scripts/Start-Mac.sh
```

If `python3.11` is not on PATH, install with `./scripts/Install-Mac.sh --python /absolute/path/to/python3.11`. The installer creates `.venv`, installs official PyPI wheels (`torch>=2.14`, `torchaudio>=2.11` on Apple Silicon) and the editable engine with `setuptools<82`, then verifies actual MPS execution. An existing `.venv` must also use native Python 3.11; move or remove an incompatible environment before reinstalling. The installer creates `.env` only if absent and explicitly selects `LOCAL_ENGINE_DEVICE=mps`; it never overwrites existing configuration.

The launcher binds only to `http://127.0.0.1:8767` (`--port 8768` changes the port). `--preflight-only` imports the inference dependencies and executes FP32 and FP16 Metal operations without downloading models or starting the server. Full model loading and inference happen on the first request, so preflight alone does not verify a model checkpoint.

```sh
curl --fail http://127.0.0.1:8767/health
```

Model downloads persist across launches: Hugging Face under `.cache/huggingface`, Torch under `.cache/torch`, and GigaAM under `~/.cache/gigaam`; temporary files use `.cache/tmp`. `HF_HOME`, `TORCH_HOME`, and `TMPDIR` override those respective launcher defaults. The required ffmpeg installation also supports `LOCAL_ENGINE_PREPROCESSING=afftdn`.

The macOS launcher reads simple `NAME=value` assignments from `.env`, ignores blank lines and full-line `#` comments, and strips matching outer single or double quotes. Values are literal: no shell commands, variable expansion, `export` statements, or inline comments. An already-exported environment variable takes precedence over `.env`. The launcher defaults to `mps` and rejects any other device; both `PYTORCH_ENABLE_MPS_FALLBACK` and `PYTORCH_MPS_FAST_MATH` must be unset or `0` and are forced to `0` before importing PyTorch. Unsupported Metal operations fail visibly instead of silently running on CPU.

### Durable local service

After installing the environment and configuring `.env`, stop any foreground engine and run:

```sh
./scripts/Install-MacService.sh
```

This installs `~/Library/LaunchAgents/com.babel.l0-draft-engine.plist` for your normal login user, without sudo. The service starts immediately and at login, survives terminal closure, and automatically restarts after a crash. It stays loopback-only at `http://127.0.0.1:8767`; `--port 8768` selects another port. Re-running the installer replaces the existing service, but it will not kill an unrelated process occupying the requested port.

The LaunchAgent uses this checkout, its `.venv`, and `.env` directly. Keep them in place. Put private checkpoints under `models/` rather than Downloads/Desktop/Documents, whose background access can be restricted by macOS privacy permissions. The installer records an explicit PATH so Homebrew ffmpeg is available outside your terminal. Shell-only environment overrides are not copied into the service; put persistent settings in `.env`.

```sh
# Inspect the service
launchctl print "gui/$(id -u)/com.babel.l0-draft-engine"

# Restart after editing .env; interrupts any active requests
launchctl kickstart -k "gui/$(id -u)/com.babel.l0-draft-engine"

# Stop and uninstall autostart; preserves models, .env, and logs
./scripts/Install-MacService.sh --remove
```

Logs append to `.logs/engine.out.log` and `.logs/engine.err.log`. The Mac must be awake and your user logged in; this is not a pre-login system daemon and does not prevent sleep.

### MacBook backend (detached)

The public Apache vhost on `ethernetservers` routes L0 requests to the browser inference coordinator (`127.0.0.1:18768`) first. Homeserver (`127.0.0.1:18767`) and Razer (`127.0.0.1:28767`) remain configured as trusted hot standbys. The Mac member (`127.0.0.1:38767`) was removed from the balancer, and its reverse SSH tunnel and inference LaunchAgents were uninstalled. Models, `.env`, and logs remain on the Mac. The gateway configuration before Mac removal is backed up at `/etc/apache2/sites-enabled/reviewgen.ovh.conf.bak-without-mac-20260924T200308Z`.

To restore the Mac as a backend, run `./scripts/Install-MacService.sh` and `./scripts/Install-MacTunnel.sh` as the normal login user, verify gateway loopback port 38767, then re-add its Apache `BalancerMember` and reload Apache after `apache2ctl configtest`. The local engine listens only on `127.0.0.1:8767`; the tunnel binds only gateway loopback port 38767. Neither is needed for a connected browser volunteer to process a request.

`/health` can report `{"ok":true,"service":"coordinator","backendsAvailable":false}` while a volunteer is available. Confirm inference with `/v1/draft` or `/v1/transcribe`, not health alone.

### Browser volunteer inference

The coordinator is installed in `/opt/babel-l0-swarm` on `ethernetservers` and runs as `babel-l0-swarm.service` on loopback port 18768. Deploy updates with the lightweight `requirements-coordinator.txt` and `scripts/babel-l0-swarm.service`; keep exactly one Uvicorn worker because volunteer credentials, queued jobs, and leases are in process memory. `scripts/babel-l0-swarm-apache.conf` shows the gateway balancer member. Check `systemctl status babel-l0-swarm` and the public `/health` `swarm` counts after a restart. If the coordinator fails, Apache routes to the existing trusted backends.

An extension browser with the complete verified model bundle and **Enable downloaded local browser models** checked registers automatically. A volunteer downloads another user's two WAV tracks over HTTPS, computes a draft or transcription locally, and sends one schema-validated result back to the coordinator. Opting out stops new work. If no volunteer is available, a lease expires, or inference fails, the coordinator uses the trusted backend; `/v1/queue/{requestId}` still reports queue state. Each worker processes one leased job at a time. Audio and lease credentials are temporary and are discarded when the request ends. The Options page discloses the audio transfer and device resource use before opt-in.

The new extension must be distributed before other users' browsers can participate: build version `0.2.37` with `npm run build:core` and `npm run build:zip -- --no-build` in `../babel/drafting/gold-drafting-extension`, then use its existing manual Chrome Web Store release workflow. Deploying the coordinator alone does not upgrade installed extensions.

### Idle model memory

`LOCAL_ENGINE_MODEL_IDLE_SECONDS=300` unloads both models after five minutes without an inference request. Accepted uploads, queued work, and running inference keep models resident; `/health` polling does not extend the timeout. Unloading drops model references, collects garbage, and releases the selected GPU allocator's unused cache. The next request transparently reloads the configured models, so the first request after idle has additional loading latency.

The HTTP/Python runtime remains resident; idle memory is not zero. Model files remain cached on disk. `/health` reports `loaded: false` after eviction, while `cached: true` means the local checkpoint still exists. Set the timeout to an integer from 1 to 86400 seconds, or `0` to keep models resident indefinitely. Restart the service after changing `.env`.

### Precision and device selection

CUDA and MPS use the same model precision policy: GigaAM is loaded with `fp16_encoder=False`, retaining FP32 weights and preprocessing, while the upstream encoder uses FP16 autocast on GPU. Punctuation uses FP16 on both CUDA and MPS. CPU uses FP32. There is no quantization, blanket CPU fallback, or reduced-precision fast-math mode in the macOS launcher. Matching dtypes does **not** guarantee bit-identical cross-backend results; CUDA and Metal kernels can produce numerical differences and occasionally different transcripts or punctuation.

For direct Python/uvicorn launches, `LOCAL_ENGINE_DEVICE` accepts `mps`, `cuda`, or `cpu`; the default is `mps` on native Apple Silicon macOS and `cuda` elsewhere. CPU is explicit opt-in, not an automatic fallback. `Start-Mac.sh` intentionally requires `mps`; to deliberately use CPU, run the application directly, for example:

```sh
LOCAL_ENGINE_DEVICE=cpu .venv/bin/python -m uvicorn l0_draft_engine.app:app --host 127.0.0.1 --port 8767
```

Direct uvicorn launches do not import `.env`; export any other required settings yourself.

## Configuration

Copy `.env.example` to `.env` before customizing Docker settings. No model weights ship in the repository or image. For private checkpoints, place them in `models/` and set:

```env
LOCAL_ENGINE_GIGAAM_MODEL=/models/my-gigaam.ckpt
LOCAL_ENGINE_PUNCTUATION_MODEL=/models/my-punctuation-model
```

On Windows, use absolute Windows paths instead; on native macOS, use absolute macOS paths (for example `/Users/you/models/my-gigaam.ckpt`), not Docker's `/models` path. `LOCAL_ENGINE_MAX_INFLIGHT_REQUESTS` accepts 1–64 (default 3, including the running request).

Endpoints: `/health`, `/v1/draft`, `/v1/transcribe`. Inference requests require `X-Babel-Local-Engine: 1` and multipart fields containing a `payload` JSON object and its two declared mono WAV tracks. Drafting returns transcript rows; transcription returns per-lane word timings. Model downloads require network access on first use.
