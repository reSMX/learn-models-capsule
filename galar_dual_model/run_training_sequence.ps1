param(
    [int]$WaitForAnatomyPid = 0
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv-win\Scripts\python.exe'
$cache = 'C:\Users\Maxx\Galar_256_cache'
$runs = Join-Path $PSScriptRoot 'runs'
$logs = Join-Path $runs 'sequence_logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null

function Invoke-Training {
    param(
        [string]$Task,
        [string]$OutputName
    )

    $script = Join-Path $PSScriptRoot "train_$Task.py"
    $output = Join-Path $runs $OutputName
    $stdout = Join-Path $logs "$OutputName.stdout.log"
    $stderr = Join-Path $logs "$OutputName.stderr.log"
    $arguments = @(
        '-B', $script,
        '--image-cache-root', $cache,
        '--output', $output,
        '--workers', '2',
        '--prefetch-factor', '2',
        '--batch-size', '64',
        '--ram-buffer-gb', '0',
        '--read-block-size', '512',
        '--device', 'cuda'
    )
    $process = Start-Process `
        -FilePath $python `
        -ArgumentList $arguments `
        -WorkingDirectory $projectRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -Wait `
        -PassThru
    return $process.ExitCode
}

$anatomySucceeded = $false
if ($WaitForAnatomyPid -gt 0) {
    try {
        $anatomy = [System.Diagnostics.Process]::GetProcessById($WaitForAnatomyPid)
        $anatomy.WaitForExit()
        $anatomySucceeded = $anatomy.ExitCode -eq 0
    }
    catch {
        $anatomySucceeded = $false
    }
}

if (-not $anatomySucceeded) {
    $anatomyOutput = if ($WaitForAnatomyPid -gt 0) {
        'anatomy_ssd_q95_recovery'
    }
    else {
        'anatomy_ssd_q95'
    }
    $anatomyExit = Invoke-Training -Task 'anatomy' -OutputName $anatomyOutput
    if ($anatomyExit -ne 0) {
        exit $anatomyExit
    }
}

$pathologyExit = Invoke-Training -Task 'pathology' -OutputName 'pathology_ssd_q95'
exit $pathologyExit
