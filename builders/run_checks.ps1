$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Python .venv tidak ditemukan: $python"
}

Push-Location $projectRoot
$previousErrorActionPreference = $ErrorActionPreference
try {
    $ErrorActionPreference = "Continue"
    & $python (Join-Path $projectRoot "test_pipeline_logic.py") 2>&1
    $logicExitCode = $LASTEXITCODE
    if ($logicExitCode -ne 0) { throw "test_pipeline_logic.py gagal dengan exit code $logicExitCode" }
    & $python (Join-Path $projectRoot "builders\verify_dashboard_states.py") 2>&1
    $dashboardExitCode = $LASTEXITCODE
    if ($dashboardExitCode -ne 0) { throw "verify_dashboard_states.py gagal dengan exit code $dashboardExitCode" }
    & $python (Join-Path $projectRoot "builders\notebook_gate.py") --check-all 2>&1
    $notebookExitCode = $LASTEXITCODE
    if ($notebookExitCode -ne 0) { throw "notebook builder/source gate gagal dengan exit code $notebookExitCode" }
}
finally {
    $ErrorActionPreference = $previousErrorActionPreference
    Pop-Location
}

Write-Host "Semua pemeriksaan logic, dashboard, dan sinkronisasi notebook lolos." -ForegroundColor Green
