# L0 Draft Engine

## Current official WebGPU runtime

The current release uses the shared browser GigaAM/C-denoise runtime for L0 and the packaged ZipEnhancer GPU runtime for denoising. Build/package it from Gold Drafting and follow the [current-release deployment guide](../gold-drafting-extension/deploy/inference-release/README.md). Install `requirements-webgpu.lock` and the pinned Playwright package, preserve a dedicated Chromium profile, and set `LOCAL_ENGINE_RUNTIME=c-denoise-webgpu` plus `LOCAL_ENGINE_REQUIRE_CURRENT_RELEASE=1`. Startup must pass real hardware inference for both capabilities; software adapters and CPU neural fallback are rejected.

The Docker, `Start-Windows.ps1`, and `Start-Mac.sh` instructions below describe the legacy GigaAM/SBERT stack, not this release. Use `Start-WebGPU.ps1` for the current Windows deployment. The public coordinator needs only `requirements-coordinator.txt`, never model weights, Chromium, or a GPU.

All uncached operations try official services before the volunteer swarm. Official enhancement first checks `/health` for the exact model ID, implementation SHA, source-graph SHA, and hardware provider; ASR-only endpoints receive no enhancement audio. Returned pairs are bounded and checked for source identity, order, clocks, canonical WAVs and digests before release. Original/output temporary files live only through the request/response; the requester retains its existing one-pair cache.

Successful validated work updates an in-memory per-operation throughput average (audio seconds per elapsed second, or draft tokens per second; 75% previous / 25% newest). Official endpoints use that ranking, retaining configured order for equal/unmeasured capacity. Swarm polls prefer a strictly faster compatible idle worker that polled within three seconds; unavailable, busy, stale, undersized, wrong-model and same-client workers cannot block another eligible worker. No client-declared GPU speed is trusted.

The finite owner deadline reserves up to the configured swarm queue-plus-lease duration, capped at half the remaining time. Official attempts share the earlier phase so one stalled endpoint cannot consume every fallback opportunity. Transport failures, invalid results, HTTP 408/429 and server errors allow fallback; ordinary input-validation errors are not retried through the swarm.

### Managed deployments and home maintenance

| Deployment | Supervision | Local / public-loopback port | State |
|---|---|---|---|
| Windows combined L0 + ZipEnhancer 0.2.50 | `Babel Official GPU Engine 0.2.50`, `Babel Official GPU Reviewgen Tunnel 0.2.50` | 8772 / 28772 | Verified public HTTP inference with packaged full-FP32 numerical recovery |
| Previous combined 0.2.49 service | `Babel Official GPU Engine`, `Babel Official GPU Reviewgen Tunnel` | 8771 / 28771 | Retained for the previous model hash during rollout |
| Previous current-release ASR service | Existing service/tunnel retained | public 28770 | L0 fallback; not enhancement-capable |
| Home combined service | `babel-official-inference.service`, `babel-official-inference-tunnel.service` | 8769 / 38770 | Inference stopped/disabled and excluded pending numerical validation |

The active public pool is `COORDINATOR_BACKEND_URLS=http://127.0.0.1:28772,http://127.0.0.1:28771,http://127.0.0.1:28770`. The coordinator service/drop-in is `babel-cdenoise-r2-coordinator.service`; its timing cache is preserved across updates. The current Windows deployment is `D:/babel_deployments/official-inference-20261012-r1`; the previous directory remains intact. Independent Gold 0.2.50, 0.2.49 and 0.2.48 swarm donors coexist during Store rollout. The recovery-capable model hash is `d98d4e0b60124a5d8bb3a524562f41685ca4d2dcdad9ea978c064cd734ac1384`.

Home maintenance upgraded the pinned driver to NVIDIA 615.71.09 open-kernel DKMS, preserved kernel 6.12.95 and the VPN module, and rebooted successfully. CfT 157.0.8097.0 exposes hardware `shader-f16` without a bypass. Desktop/Sunshine, the old Docker GPU service and tunnels were restored before a later host/network outage. Rollback packages, initramfs/config backups and the stopped pre-migration browser profile are retained under `/var/backups/babel-nvidia-20261011`.

Functional preflight and public-sample HTTP inference passed on home, but its enhanced waveform still differs from the Windows reference by 4.565% relative RMS error; a quiet input and silent-tail case also differ. All 54 deployed core files, 424 generated shaders and 541 uploaded weight/input buffers matched across platforms. The numerical cause is not yet established; do not equate adapter availability or HTTP 200 with acceptance, weaken validators, or add port 38770 to the pool. Direct, relay and private SSH routes subsequently became unreachable, and the operator deferred further home work until physical/network access is restored.



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

The public Apache vhost on `ethernetservers` routes current-release requests through `babel-cdenoise-r2-coordinator.service` on loopback port 18770. Trusted engines must not be Apache hot standbys: they accept JSON timing, not the public task-ID/capability contract. `COORDINATOR_BACKEND_URLS` selects the official pool; add an endpoint only after real two-lane L0/draft and enhancement verification, not merely host or container health. Current ports and activation status are listed in the managed deployments table above.

The Mac backend (`127.0.0.1:38767`) remains detached. Restore public participation only with the current shared WebGPU backend, a dedicated profile/tunnel, and real release-compatible transcription plus completed-label drafting. The legacy Metal/SBERT launcher does not satisfy this contract. Add a verified endpoint to `COORDINATOR_BACKEND_URLS`, never directly to the public Apache balancer.

`/health` can report `{"ok":true,"service":"coordinator","backendsAvailable":false}` while a volunteer is available. Confirm inference with `/v1/transcribe` followed by `/v1/draft`, not health alone.

### Browser volunteer inference

Deploy the coordinator into an immutable release directory on `ethernetservers`; the active path is selected by the `babel-cdenoise-r2-coordinator.service` drop-in. Keep exactly one Uvicorn worker because worker credentials, throughput measurements, queued jobs, leases and in-flight timing joins are in process memory. Production timing cache remains `/var/cache/babel-cdenoise-r2/timing` (rotating LRU, at most 4 GiB); `COORDINATOR_CACHE_DIR` and `COORDINATOR_CACHE_MAX_BYTES` configure it. Preserve that private writable directory across coordinator updates. Inspect `systemctl cat/status babel-cdenoise-r2-coordinator` and the public `/health` swarm counts, then exercise both operations.

The extension generates a random per-task bearer capability before its first timing upload and persists it by canonical `taskId`; send `Authorization: Bearer <token>` on all three public inference requests. `POST /v1/transcribe` still accepts exactly two mono WAV tracks and multipart `payload` with the canonical task ID. It returns two tracks with word tokens, S2 segments, sample rate, and processed PCM digest, plus the echoed `accessToken`. `POST /v1/timing/lookup` with JSON `{"taskId":"..."}` returns cached timing (404 for a missing or unauthorized task); it also waits for an authenticated in-flight timing job. Cached and in-flight timing are private to the pair of task ID and capability digest: another browser capability can independently transcribe the same task without accessing or blocking the first browser's timing. Within that pair, repeat uploads reuse timing only when the ordered lane/audio-file digest mapping and requested preprocessing match; different input returns 409. Omitted preprocessing is distinct from either explicit mode because the backend selects its default. `POST /v1/draft` with JSON `{"taskId":"..."}` (optionally `{"options":{"preserveRows":[...]}}`) waits for the same private first stage and dispatches punctuation only, without audio. Timing is not exposed to a caller merely knowing the task ID.

The coordinator writes version-2 timing cache records with no raw bearer token. On startup it deletes the old task-ID-only cache files rather than migrating records that lack lane and preprocessing identity. Existing browser capabilities remain valid, but their first lookup after this cutover returns 404 until they upload audio again.

Volunteering is a separate explicit opt-in, not a consequence of selecting local transcription or remote enhancement. Current workers use protocol 3, bundle schema v3 and the enforced model release. L0 additionally needs its verified downloaded bundle; ZipEnhancer advertises its exact packaged model capability. After official services fail, an eligible volunteer downloads the two original WAVs for transcription or enhancement; a draft lease uses cached timing and completed punctuation labels without rerunning ASR. Each worker owns one lease at a time. Enhanced uploads/results are request-scoped and never enter the donor's one-pair cache. Per-offscreen client IDs prevent a requester from leasing its own enhancement. `/v1/queue/{requestId}` forwards official execution progress or reports the swarm state. Audio and lease credentials are discarded when the request ends.

Distribute the timing-first extension from the same coordinated release as the coordinator and trusted engine: run `npm run build:core` and `npm run build:zip -- --no-build` in `drafting/gold-drafting-extension` from the workspace root, then use its existing manual Chrome Web Store release workflow. Deploying the coordinator alone does not upgrade installed extensions or their volunteer protocol.

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

Trusted endpoints are `/health`, `/v1/transcribe`, `/v1/draft`, and `/v1/enhance` when the current WebGPU capability is available. Transcription requires `X-Babel-Local-Engine: 1` and multipart `payload` plus its two declared mono WAVs. Drafting uses the same proxy header and JSON `{"timing":<transcription response>,"options":<optional draft options>}`. Current-release calls also carry `X-Babel-Inference-Release`. Enhancement accepts its strict model/source/clock manifest plus two original WAVs and returns a verified multipart result plus two enhanced WAVs; it preserves sample rate/frame count. Use a unique request ID for progress. Models download on first preparation; readiness is not advertised before both real GPU preflights succeed.
