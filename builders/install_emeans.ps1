$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$vendorRoot = Join-Path $projectRoot "vendor\faiss_e_means"
$stagingRoot = Join-Path $projectRoot ("vendor\.emeans-stage-" + [guid]::NewGuid().ToString("N"))
$sourceRoot = "https://raw.githubusercontent.com/facebookresearch/faiss/v1.16.0"
$files = @("base.py", "e_means_torch.py", "torch_backend.py")

if (-not (Test-Path -LiteralPath (Join-Path $vendorRoot "__init__.py"))) {
    throw "Shim vendor tidak ditemukan: $vendorRoot"
}

New-Item -ItemType Directory -Path $stagingRoot -Force | Out-Null
try {
    foreach ($file in $files) {
        Invoke-WebRequest -Uri "$sourceRoot/contrib/e_means/$file" `
            -OutFile (Join-Path $stagingRoot $file)
    }
    Invoke-WebRequest -Uri "$sourceRoot/LICENSE" -OutFile (Join-Path $stagingRoot "LICENSE")
    $required = @("base.py", "e_means_torch.py", "torch_backend.py", "LICENSE")
    foreach ($file in $required) {
        if ((Get-Item -LiteralPath (Join-Path $stagingRoot $file)).Length -lt 100) {
            throw "Sumber resmi e-means tidak lengkap: $file"
        }
    }
    foreach ($file in $files) {
        Copy-Item -LiteralPath (Join-Path $stagingRoot $file) `
            -Destination (Join-Path $vendorRoot $file) -Force
    }
    Copy-Item -LiteralPath (Join-Path $stagingRoot "LICENSE") `
        -Destination (Join-Path $vendorRoot "LICENSE") -Force
    $resolvedVendor = (Resolve-Path -LiteralPath $vendorRoot).Path
    $resolvedStage = (Resolve-Path -LiteralPath $stagingRoot).Path
    if (-not $resolvedStage.StartsWith((Join-Path $projectRoot "vendor"), [StringComparison]::OrdinalIgnoreCase)) {
        throw "Folder staging keluar dari vendor project; tidak dihapus: $resolvedStage"
    }
    Remove-Item -LiteralPath $resolvedStage -Recurse -Force
    Write-Host "e-means resmi FAISS v1.16.0 terpasang di $resolvedVendor"
}
catch {
    Write-Error "Pemasangan gagal. File staging dipertahankan untuk inspeksi: $stagingRoot"
    throw
}
