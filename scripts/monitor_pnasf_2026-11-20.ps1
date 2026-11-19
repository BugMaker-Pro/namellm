$ErrorActionPreference = "Continue"

$repo = "D:\namellm"
$python = Join-Path $repo ".venv\Scripts\python.exe"
$kaggle = Join-Path $repo "scripts\kaggle_cli_ipv4.py"
$kernelRef = "yyyyhhhhwww/ntl-0704-pnasf"
$logPath = "D:\tmp\namellm_pnasf_monitor.log"
$submissionRefs = @(
    "58570156",
    "58570469",
    "58571708",
    "58578599"
)

New-Item -ItemType Directory -Path (Split-Path $logPath) -Force | Out-Null
$lastStatus = @{}

function Write-Transition {
    param([string]$Key, [string]$Value)
    if ($script:lastStatus[$Key] -ne $Value) {
        $line = "{0} {1}={2}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Key, $Value
        Add-Content -LiteralPath $logPath -Value $line -Encoding utf8
        $script:lastStatus[$Key] = $Value
    }
}

function Get-KernelStatus {
    $output = & $python $kaggle kernels status $kernelRef 2>&1
    if ($LASTEXITCODE -ne 0) {
        return "STATUS_ERROR"
    }
    return ($output | Select-Object -Last 1).ToString().Trim()
}

function Get-SubmissionStates {
    $rows = & $python $kaggle competitions submissions -c name-that-llm --csv 2>&1
    if ($LASTEXITCODE -ne 0) {
        foreach ($ref in $submissionRefs) {
            Write-Transition "submission_$ref" "STATUS_ERROR"
        }
        return
    }
    $records = $rows | ConvertFrom-Csv
    foreach ($ref in $submissionRefs) {
        $record = $records | Where-Object { $_.ref -eq $ref } | Select-Object -First 1
        if ($null -eq $record) {
            Write-Transition "submission_$ref" "NOT_LISTED"
        }
        else {
            Write-Transition "submission_$ref" "$($record.status) score=$($record.publicScore)"
        }
    }
}

for ($attempt = 0; $attempt -lt 72; $attempt++) {
    Write-Transition "pnasf_kernel" (Get-KernelStatus)
    Get-SubmissionStates

    $kernelDone = $script:lastStatus["pnasf_kernel"] -match "COMPLETE|ERROR|CANCEL"
    $submissionsDone = $true
    foreach ($ref in $submissionRefs) {
        if ($script:lastStatus["submission_$ref"] -notmatch "COMPLETE|ERROR") {
            $submissionsDone = $false
        }
    }
    if ($kernelDone -and $submissionsDone) {
        Add-Content -LiteralPath $logPath -Value "MONITOR_DONE" -Encoding utf8
        break
    }
    if ($attempt -eq 71) {
        Add-Content -LiteralPath $logPath -Value "MONITOR_TIMEOUT" -Encoding utf8
    }
    Start-Sleep -Seconds 600
}
