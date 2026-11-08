$ErrorActionPreference = 'Stop'
$statusPath = Join-Path $PSScriptRoot 'status.json'
if (-not (Test-Path -LiteralPath $statusPath)) {
    Write-Output 'Training has not been launched by this supervisor yet.'
    exit 0
}
$state = Get-Content -LiteralPath $statusPath -Raw -Encoding UTF8 | ConvertFrom-Json
$state | Format-List state,started_at,ended_at,training_pid,supervisor_pid,exit_code,stdout,stderr
if ($state.state -eq 'running') {
    $active = Get-Process -Id $state.training_pid -ErrorAction SilentlyContinue
    if ($null -eq $active) {
        Write-Output 'Recorded training process is absent. Inspect stderr and resume from saved checkpoints.'
    } else {
        $active | Select-Object Id,ProcessName,CPU,@{Name='RAM_MB';Expression={[math]::Round($_.WorkingSet64 / 1MB)}} | Format-Table
    }
}
Write-Output 'Recent output:'
Get-Content -LiteralPath $state.stdout -Tail 8 -Encoding UTF8
Write-Output 'Recent diagnostics:'
Get-Content -LiteralPath $state.stderr -Tail 8 -Encoding UTF8
$validationPath = Join-Path $PSScriptRoot 'prediction_validation_20261029_v2\status.json'
if (Test-Path -LiteralPath $validationPath) {
    Write-Output 'Separate CPU prediction validation:'
    $validation = Get-Content -LiteralPath $validationPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $validation | Format-List state,stage,updated_at,complete,total,region,model,horizon,fold,total_folds,cpu_threads,cuda_used,report,error
}
$diagnosticsPath = Join-Path $PSScriptRoot 'prediction_diagnostics_20261029_v2\status.json'
if (Test-Path -LiteralPath $diagnosticsPath) {
    Write-Output 'Separate CPU event and probability-calibration diagnostics:'
    $diagnostics = Get-Content -LiteralPath $diagnosticsPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $diagnostics | Format-List state,stage,updated_at,complete,total,model,fold,total_folds,cpu_threads,cuda_used,report,error
}
$predictionPilotPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'prediction_improvement\runs\next_outage_1s_v1\status.json'
if (Test-Path -LiteralPath $predictionPilotPath) {
    Write-Output 'Isolated next-outage timing prediction pilot:'
    $predictionPilot = Get-Content -LiteralPath $predictionPilotPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $predictionPilot | Format-List state,stage,updated_at,complete,total,split,region,step,target,loss,score,best_step,cpu_threads,cuda_used,error
}
