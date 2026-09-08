param(
    [Parameter(Mandatory = $true)]
    [int]$WaitForAnatomyPid,
    [string]$AnatomyRun = 'galar_dual_model\runs\anatomy_workers0_20260907',
    [double]$MinimumMacroF1 = 0.40,
    [int]$ContinuationEpochs = 10
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv-win\Scripts\python.exe'
$cache = 'C:\Users\Maxx\Galar_256_cache'
$runs = Join-Path $PSScriptRoot 'runs'
$logs = Join-Path $runs 'sequence_logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null

if ([System.IO.Path]::IsPathRooted($AnatomyRun)) {
    $anatomyOutput = $AnatomyRun
}
else {
    $anatomyOutput = Join-Path $projectRoot $AnatomyRun
}
$anatomyHistory = Join-Path $anatomyOutput 'history.csv'
$anatomyBest = Join-Path $anatomyOutput 'best.pt'
$anatomyStdout = "$anatomyOutput.stdout.log"

function Start-GalarTraining {
    param(
        [Parameter(Mandatory = $true)]
        [ValidateSet('anatomy', 'pathology')]
        [string]$Task,
        [Parameter(Mandatory = $true)]
        [string]$OutputName,
        [string[]]$ExtraArguments = @()
    )

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

    $script = Join-Path $PSScriptRoot "train_$Task.py"
    $output = Join-Path $runs $OutputName
    if (Test-Path -LiteralPath $output) {
        $existing = @(Get-ChildItem -LiteralPath $output -Force)
        if ($existing.Count -gt 0) {
            throw "Refusing to overwrite non-empty run directory: $output"
        }
    }

    $stdout = Join-Path $logs "$OutputName.stdout.log"
    $stderr = Join-Path $logs "$OutputName.stderr.log"
    $arguments = @(
        '-u',
        '-B',
        $script,
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
        'cuda'
    ) + $ExtraArguments

    $process = Start-Process `
        -FilePath $python `
        -ArgumentList $arguments `
        -WorkingDirectory $projectRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -PassThru

    return @{
        task = $Task
        pid = $process.Id
        output = $output
        stdout = $stdout
        stderr = $stderr
    }
}

function Commit-GalarChanges {
    $allowedFiles = @(
        'galar_dual_model/README.md',
        'galar_dual_model/evaluate_thresholds.py',
        'galar_dual_model/run_pathology_followup.ps1',
        'galar_dual_model/run_training_sequence.ps1',
        'galar_dual_model/train_common.py'
    )
    Push-Location $projectRoot
    try {
        $alreadyStaged = @(git diff --cached --name-only)
        $unexpected = @(
            $alreadyStaged | Where-Object { $_ -notin $allowedFiles }
        )
        if ($unexpected.Count -gt 0) {
            return "skipped: unrelated staged files: $($unexpected -join ', ')"
        }

        git add -- $allowedFiles
        if ($LASTEXITCODE -ne 0) {
            return "failed: git add exited with code $LASTEXITCODE"
        }
        git diff --cached --check -- $allowedFiles
        if ($LASTEXITCODE -ne 0) {
            return "failed: git diff --check exited with code $LASTEXITCODE"
        }
        git diff --cached --quiet -- $allowedFiles
        if ($LASTEXITCODE -eq 0) {
            return 'no changes to commit'
        }
        if ($LASTEXITCODE -ne 1) {
            return "failed: git diff --quiet exited with code $LASTEXITCODE"
        }

        git commit -m 'Improve Galar continuation and pathology training'
        if ($LASTEXITCODE -ne 0) {
            return "failed: git commit exited with code $LASTEXITCODE"
        }
        git push
        if ($LASTEXITCODE -ne 0) {
            return "committed locally; git push exited with code $LASTEXITCODE"
        }
        return 'committed and pushed'
    }
    finally {
        Pop-Location
    }
}

$anatomyProcess = Get-Process -Id $WaitForAnatomyPid -ErrorAction Stop
$anatomyProcess.WaitForExit()
$anatomyExitCode = $anatomyProcess.ExitCode

if (-not (Test-Path -LiteralPath $anatomyHistory)) {
    throw "Anatomy history was not written: $anatomyHistory"
}
$history = @(Import-Csv -LiteralPath $anatomyHistory)
if ($history.Count -eq 0) {
    throw "Anatomy history is empty: $anatomyHistory"
}

$last = $history[-1]
$best = $history |
    Sort-Object { [double]$_.validation_supported_macro_f1 } -Descending |
    Select-Object -First 1
$lastEpoch = [int]$last.epoch
$bestEpoch = [int]$best.epoch
$bestMacroF1 = [double]$best.validation_supported_macro_f1
$lastValidationF1 = [double]$last.validation_supported_macro_f1
$finalGeneralizationGap = (
    [double]$last.train_supported_macro_f1 - $lastValidationF1
)
$earlyStopped = (
    (Test-Path -LiteralPath $anatomyStdout) -and
    (Select-String -LiteralPath $anatomyStdout -Pattern 'Early stopping at epoch' -Quiet)
)
$completionSummaryWritten = (
    (Test-Path -LiteralPath $anatomyStdout) -and
    (Select-String -LiteralPath $anatomyStdout -Pattern 'Best validation supported macro-F1=' -Quiet)
)

# Early stopping already enforces recency of the best epoch. The extra checks
# reject a continuation when validation has materially fallen from its best or
# when the final train/validation macro-F1 gap is plainly excessive.
$clearOverfitting = (
    $lastValidationF1 -lt ($bestMacroF1 - 0.03) -or
    $finalGeneralizationGap -gt 0.45
)
$reachedConfiguredEnd = $lastEpoch -eq 20
$normalCompletion = ($anatomyExitCode -eq 0) -or $completionSummaryWritten
$continueAnatomy = (
    $normalCompletion -and
    $reachedConfiguredEnd -and
    -not $earlyStopped -and
    $bestMacroF1 -gt $MinimumMacroF1 -and
    -not $clearOverfitting -and
    (Test-Path -LiteralPath $anatomyBest)
)

if (-not $normalCompletion) {
    $reason = "Anatomy process exited with code $anatomyExitCode"
}
elseif ($earlyStopped) {
    $reason = 'Anatomy stopped automatically after four epochs without improvement'
}
elseif (-not $reachedConfiguredEnd) {
    $reason = "Anatomy ended before epoch 20 (last epoch: $lastEpoch)"
}
elseif ($bestMacroF1 -le $MinimumMacroF1) {
    $reason = "best validation macro-F1 did not exceed $MinimumMacroF1"
}
elseif ($clearOverfitting) {
    $reason = 'Anatomy showed a clear train/validation generalization gap'
}
elseif (-not (Test-Path -LiteralPath $anatomyBest)) {
    $reason = 'Anatomy best checkpoint is missing'
}
else {
    $reason = 'Anatomy reached epoch 20 above the quality gate without clear overfitting'
}

$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$decision = [ordered]@{
    timestamp = (Get-Date).ToString('o')
    anatomy_pid = $WaitForAnatomyPid
    anatomy_exit_code = $anatomyExitCode
    completion_summary_written = $completionSummaryWritten
    last_epoch = $lastEpoch
    best_epoch = $bestEpoch
    best_validation_macro_f1 = $bestMacroF1
    last_validation_macro_f1 = $lastValidationF1
    final_generalization_gap = $finalGeneralizationGap
    early_stopped = $earlyStopped
    clear_overfitting = $clearOverfitting
    continue_anatomy = $continueAnatomy
    reason = $reason
}
$decisionPath = Join-Path $logs "decision_$timestamp.json"
$decision | ConvertTo-Json | Set-Content -LiteralPath $decisionPath -Encoding UTF8

if ($continueAnatomy) {
    $outputName = "anatomy_continuation_$timestamp"
    $followup = Start-GalarTraining `
        -Task 'anatomy' `
        -OutputName $outputName `
        -ExtraArguments @(
            '--epochs',
            "$ContinuationEpochs",
            '--initial-checkpoint',
            $anatomyBest,
            '--freeze-backbone-epochs',
            '0',
            '--unfreeze-stages-per-epoch',
            '0',
            '--head-lr',
            '0.0001',
            '--backbone-lr',
            '0.00001'
        )
}
else {
    $outputName = "pathology_asl_anatomy_init_$timestamp"
    $pathologyArguments = @(
        '--loss',
        'asymmetric',
        '--asymmetric-gamma-neg',
        '4',
        '--asymmetric-gamma-pos',
        '1',
        '--asymmetric-clip',
        '0.05',
        '--classifier-dropout',
        '0.4'
    )
    if (Test-Path -LiteralPath $anatomyBest) {
        $pathologyArguments += @('--initial-backbone-checkpoint', $anatomyBest)
    }
    $followup = Start-GalarTraining `
        -Task 'pathology' `
        -OutputName $outputName `
        -ExtraArguments $pathologyArguments
}

$followup['decision'] = $decisionPath
$followup['reason'] = $reason
$followup['git'] = if ($reachedConfiguredEnd) {
    Commit-GalarChanges
}
else {
    'skipped: Anatomy did not reach epoch 20'
}
$followupPath = Join-Path $logs "followup_$timestamp.json"
$followup | ConvertTo-Json | Set-Content -LiteralPath $followupPath -Encoding UTF8
