[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][ValidateSet("engine", "tunnel")][string]$Mode,
    [ValidateRange(1, 65535)][int]$Port = 8767,
    # Deployed revision to serve. The failover must match the homeserver's
    # committed checkout, never an in-progress working tree.
    [string]$EngineRoot = "",
    [string]$TunnelTarget = "root@93.127.223.38",
    [ValidateRange(1, 65535)][int]$TunnelRemotePort = 28767,
    [ValidateRange(1, 3600)][int]$MinBackoffSeconds = 5,
    [ValidateRange(1, 3600)][int]$MaxBackoffSeconds = 60
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$LogRoot = Join-Path $RepoRoot ".logs"
New-Item -ItemType Directory -Force $LogRoot | Out-Null
$LogPath = Join-Path $LogRoot "$Mode.log"
$OutLogPath = Join-Path $LogRoot "$Mode.out.log"
$ErrLogPath = Join-Path $LogRoot "$Mode.err.log"

# The healing trigger re-runs this task every few minutes, so exactly one
# supervisor per mode must survive; the rest exit immediately.
$LockPath = Join-Path $LogRoot "$Mode.lock"
try {
    $script:Lock = [System.IO.File]::Open(
        $LockPath,
        [System.IO.FileMode]::OpenOrCreate,
        [System.IO.FileAccess]::Write,
        [System.IO.FileShare]::None
    )
} catch {
    exit 0
}

function Write-Log {
    param([Parameter(Mandatory = $true)][string]$Message)

    $line = "{0} [{1}] {2}" -f (Get-Date -Format "yyyy-MM-ddTHH:mm:ssK"), $Mode, $Message
    Add-Content -LiteralPath $LogPath -Value $line
}

function Get-Attempt {
    if ($Mode -eq "engine") {
        $engineHome = if ($EngineRoot) { $EngineRoot } else { $RepoRoot }
        $startScript = Join-Path $engineHome "scripts\Start-Windows.ps1"
        if (-not (Test-Path $startScript)) {
            throw "Engine launcher is missing: $startScript"
        }
        return @{
            FilePath  = (Get-Command powershell.exe).Source
            Arguments = @(
                "-NoProfile",
                "-ExecutionPolicy", "Bypass",
                "-File", $startScript,
                "-Port", $Port
            )
        }
    }

    $ssh = Get-Command ssh.exe -ErrorAction SilentlyContinue
    if (-not $ssh) {
        throw "ssh.exe was not found on PATH; install the Windows OpenSSH client."
    }
    return @{
        FilePath  = $ssh.Source
        Arguments = @(
            "-N",
            "-o", "BatchMode=yes",
            "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3",
            "-R", "127.0.0.1:${TunnelRemotePort}:127.0.0.1:${Port}",
            $TunnelTarget
        )
    }
}

$backoff = $MinBackoffSeconds
Write-Log "supervisor started (pid $PID)"

while ($true) {
    $attempt = Get-Attempt
    $startedAt = Get-Date
    Write-Log "starting: $($attempt.FilePath) $($attempt.Arguments -join ' ')"

    try {
        $process = Start-Process `
            -FilePath $attempt.FilePath `
            -ArgumentList $attempt.Arguments `
            -WindowStyle Hidden `
            -RedirectStandardOutput $OutLogPath `
            -RedirectStandardError $ErrLogPath `
            -PassThru
        $process.WaitForExit()
        $exitCode = $process.ExitCode
    } catch {
        $exitCode = -1
        Write-Log "launch failed: $($_.Exception.Message)"
    }

    $ranSeconds = [int]((Get-Date) - $startedAt).TotalSeconds
    Write-Log "exited code=$exitCode after ${ranSeconds}s"

    if ($ranSeconds -ge 60) {
        $backoff = $MinBackoffSeconds
    } else {
        $backoff = [Math]::Min($backoff * 2, $MaxBackoffSeconds)
    }

    Write-Log "restarting in ${backoff}s"
    Start-Sleep -Seconds $backoff
}
