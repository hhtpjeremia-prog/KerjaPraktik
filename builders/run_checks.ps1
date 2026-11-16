$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Python .venv tidak ditemukan: $python"
}

Push-Location $projectRoot
try {
    & $python (Join-Path $projectRoot "test_pipeline_logic.py")
    if ($LASTEXITCODE -ne 0) { throw "test_pipeline_logic.py gagal dengan exit code $LASTEXITCODE" }
    & $python (Join-Path $projectRoot "builders\verify_dashboard_states.py")
    if ($LASTEXITCODE -ne 0) { throw "verify_dashboard_states.py gagal dengan exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
}

Write-Host "Semua pemeriksaan pipeline dan dashboard lolos." -ForegroundColor Green
