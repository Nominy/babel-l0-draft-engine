[CmdletBinding()]
param([ValidateRange(1, 65535)][int]$Port = 8769)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$backendRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $backendRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw "Install the backend's pinned Python dependencies into .venv first." }
$env:LOCAL_ENGINE_RUNTIME = "c-denoise-webgpu"
$env:LOCAL_ENGINE_REQUIRE_CURRENT_RELEASE = "1"
$env:BABEL_INFERENCE_WORKER = Join-Path $backendRoot "worker.mjs"
$env:BABEL_INFERENCE_PROFILE = Join-Path $backendRoot "chromium"
$env:BABEL_INFERENCE_RPC_SECONDS = "840"
$env:PYTHONUTF8 = "1"
Set-Location -LiteralPath $backendRoot
& $python -m uvicorn l0_draft_engine.app:app --host 127.0.0.1 --port $Port --workers 1
exit $LASTEXITCODE
