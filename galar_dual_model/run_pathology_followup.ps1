param(
    [Parameter(Mandatory = $true)]
    [int]$WaitForPathologyPid,
    [Parameter(Mandatory = $true)]
    [string]$PathologyRun,
    [Parameter(Mandatory = $true)]
    [string]$AnatomyCheckpoint,
    [double]$BaselineMacroF1 = 0.1449952249723189
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv-win\Scripts\python.exe'
$cache = 'C:\Users\Maxx\Galar_256_cache'
$runs = Join-Path $PSScriptRoot 'runs'
$logs = Join-Path $runs 'sequence_logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null

if (-not [System.IO.Path]::IsPathRooted($PathologyRun)) {
    $PathologyRun = Join-Path $projectRoot $PathologyRun
}
if (-not [System.IO.Path]::IsPathRooted($AnatomyCheckpoint)) {
    $AnatomyCheckpoint = Join-Path $projectRoot $AnatomyCheckpoint
}

$pathologyProcess = Get-Process -Id $WaitForPathologyPid -ErrorAction SilentlyContinue
if ($null -ne $pathologyProcess) {
    $pathologyProcess.WaitForExit()
}
$historyPath = Join-Path $PathologyRun 'history.csv'
$bestCheckpoint = Join-Path $PathologyRun 'best.pt'
if (-not (Test-Path -LiteralPath $historyPath)) {
    throw "Pathology history was not written: $historyPath"
}
$history = @(Import-Csv -LiteralPath $historyPath)
if ($history.Count -eq 0) {
    throw "Pathology history is empty: $historyPath"
}

$best = $history |
    Sort-Object { [double]$_.validation_supported_macro_f1 } -Descending |
    Select-Object -First 1
$last = $history[-1]
$bestMacroF1 = [double]$best.validation_supported_macro_f1
$runImproved = $bestMacroF1 -gt $BaselineMacroF1
$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$decision = [ordered]@{
    timestamp = (Get-Date).ToString('o')
    pathology_pid = $WaitForPathologyPid
    source_run = $PathologyRun
    last_epoch = [int]$last.epoch
    best_epoch = [int]$best.epoch
    best_validation_macro_f1 = $bestMacroF1
    baseline_validation_macro_f1 = $BaselineMacroF1
    improved_over_baseline = $runImproved
    launch_sqrt_bce = -not $runImproved
}
$decisionPath = Join-Path $logs "pathology_decision_$timestamp.json"
$decision | ConvertTo-Json | Set-Content -LiteralPath $decisionPath -Encoding UTF8

if ($runImproved) {
    exit 0
}

$activeTraining = @(
    Get-CimInstance Win32_Process |
        Where-Object {
            $_.CommandLine -match 'galar_dual_model[\\/]train_(anatomy|pathology)\.py'
        }
)
if ($activeTraining.Count -gt 0) {
    $activePids = ($activeTraining.ProcessId -join ', ')
    throw "Refusing to duplicate active Galar training process(es): $activePids"
}
if (-not (Test-Path -LiteralPath $AnatomyCheckpoint)) {
    throw "Anatomy initialization checkpoint is missing: $AnatomyCheckpoint"
}

$outputName = "pathology_sqrt_bce_anatomy_init_$timestamp"
$output = Join-Path $runs $outputName
$stdout = Join-Path $logs "$outputName.stdout.log"
$stderr = Join-Path $logs "$outputName.stderr.log"
$arguments = @(
    '-u',
    '-B',
    (Join-Path $PSScriptRoot 'train_pathology.py'),
    '--image-cache-root',
    $cache,
    '--output',
    $output,
    '--workers',
    '0',
    '--batch-size',
    '64',
    '--ram-buffer-gb',
    '2',
    '--read-block-size',
    '512',
    '--device',
    'cuda',
    '--loss',
    'bce',
    '--pos-weight-power',
    '0.5',
    '--max-pos-weight',
    '20',
    '--classifier-dropout',
    '0.4',
    '--initial-backbone-checkpoint',
    $AnatomyCheckpoint
)
$followup = Start-Process `
    -FilePath $python `
    -ArgumentList $arguments `
    -WorkingDirectory $projectRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -PassThru

$followupRecord = [ordered]@{
    timestamp = (Get-Date).ToString('o')
    task = 'pathology'
    strategy = 'sqrt_smoothed_bce_with_anatomy_backbone_initialization'
    pid = $followup.Id
    output = $output
    stdout = $stdout
    stderr = $stderr
    source_decision = $decisionPath
    previous_best_checkpoint = $bestCheckpoint
}
$followupPath = Join-Path $logs "pathology_followup_$timestamp.json"
$followupRecord | ConvertTo-Json | Set-Content -LiteralPath $followupPath -Encoding UTF8
