[CmdletBinding()]
param(
    [string]$EngineTaskName = "Babel L0 Engine",
    [string]$TunnelTaskName = "Babel L0 Reviewgen Tunnel",
    [ValidateRange(1, 65535)][int]$Port = 8767,
    # Pristine deployed checkout the failover serves; defaults to this repo.
    [string]$EngineRoot = "",
    [string]$TunnelTarget = "root@93.127.223.38",
    [ValidateRange(1, 65535)][int]$TunnelRemotePort = 28767,
    [switch]$Remove
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Runner = Join-Path $PSScriptRoot "Run-Durable.ps1"
if (-not (Test-Path $Runner)) {
    throw "Supervisor script is missing: $Runner"
}
$Launcher = Join-Path $PSScriptRoot "run-hidden.vbs"
if (-not (Test-Path $Launcher)) {
    throw "Windowless launcher is missing: $Launcher"
}


function Remove-TaskIfPresent {
    param([Parameter(Mandatory = $true)][string]$Name)

    if (Get-ScheduledTask -TaskName $Name -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $Name -Confirm:$false
        Write-Host "Removed scheduled task: $Name"
    }
}

if ($Remove) {
    Remove-TaskIfPresent -Name $EngineTaskName
    Remove-TaskIfPresent -Name $TunnelTaskName
    return
}

function Register-DurableTask {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string[]]$RunnerArguments
    )

    Remove-TaskIfPresent -Name $Name

    # wscript.exe never allocates a console, so neither the supervisor nor its
    # children can flash a window, and closing any console cannot kill them.
    $launcherArguments = @("`"$Launcher`"", "`"$Runner`"")
    foreach ($argument in $RunnerArguments) {
        $launcherArguments += "`"$argument`""
    }

    $action = New-ScheduledTaskAction -Execute "wscript.exe" -Argument ($launcherArguments -join " ") -WorkingDirectory $RepoRoot
    $triggers = @(
        (New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME),
        (New-ScheduledTaskTrigger -Once -At (Get-Date).Date.AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 5))
    )
    $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -DontStopOnIdleEnd `
        -RestartCount 999 `
        -RestartInterval (New-TimeSpan -Minutes 1) `
        -MultipleInstances IgnoreNew `
        -ExecutionTimeLimit ([TimeSpan]::Zero)

    Register-ScheduledTask -TaskName $Name -Action $action -Trigger $triggers -Principal $principal -Settings $settings | Out-Null
    Start-ScheduledTask -TaskName $Name
    Write-Host "Registered and started scheduled task: $Name"
}

$engineArguments = @("-Mode", "engine", "-Port", $Port)
if ($EngineRoot) {
    $resolvedEngineRoot = (Resolve-Path -LiteralPath $EngineRoot).Path
    $engineArguments += @("-EngineRoot", $resolvedEngineRoot)
}
Register-DurableTask -Name $EngineTaskName -RunnerArguments $engineArguments
Register-DurableTask -Name $TunnelTaskName -RunnerArguments @(
    "-Mode", "tunnel",
    "-Port", $Port,
    "-TunnelTarget", $TunnelTarget,
    "-TunnelRemotePort", $TunnelRemotePort
)

Write-Host ""
Write-Host "Durable supervision is active:"
Write-Host "  engine  -> http://127.0.0.1:$Port from $(if ($EngineRoot) { $EngineRoot } else { $RepoRoot })"
Write-Host "  tunnel  -> ${TunnelTarget} 127.0.0.1:${TunnelRemotePort}"
Write-Host "  logs    -> $(Join-Path $RepoRoot '.logs')"
