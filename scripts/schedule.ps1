<#
.SYNOPSIS
    Register (or remove) a Windows scheduled task that runs `python -m src.cli run` on weekdays.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\schedule.ps1                # weekdays 4:30 PM
    powershell -ExecutionPolicy Bypass -File scripts\schedule.ps1 -Time 13:45    # pick a time
    powershell -ExecutionPolicy Bypass -File scripts\schedule.ps1 -Remove

.NOTES
    -Time is in this PC's local time. US markets close at 4:00 PM New York time, so pick a
    time a little after that in your time zone (e.g. 13:30 on the West Coast).
    The task runs only while you're logged in, which is what desktop toasts need. If the PC
    was asleep at that time, it runs as soon as it wakes. Output goes to reports\logs\run.log.
#>
param(
    [string]$Time = "16:30",
    [string]$TaskName = "Portfolio Analyzer",
    [switch]$Remove
)

$ErrorActionPreference = "Stop"

if ($Remove) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Output "Removed scheduled task '$TaskName'."
    return
}

$repo = Split-Path -Parent $PSScriptRoot
$python = (Get-Command python -ErrorAction Stop).Source
New-Item -ItemType Directory -Force (Join-Path $repo "reports\logs") | Out-Null

# cmd.exe so output can be appended to the log; stdin from NUL so a login prompt fails
# fast instead of hanging; UTF-8 so the log can hold rich's box-drawing characters.
$cmd = "set `"PYTHONIOENCODING=utf-8`" &&`"$python`" -m src.cli run >> reports\logs\run.log 2>&1 < NUL"
$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c $cmd" -WorkingDirectory $repo
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At $Time
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 30)

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
    -Description "Portfolio snapshot, news, reports and alerts (python -m src.cli run)" -Force | Out-Null

Write-Output "Scheduled '$TaskName' for weekdays at $Time (local time)."
Write-Output "Run it now:  Start-ScheduledTask -TaskName '$TaskName'"
Write-Output "Log:         $repo\reports\logs\run.log"
