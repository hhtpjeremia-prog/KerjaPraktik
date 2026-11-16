$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$vendorRoot = Join-Path $projectRoot "vendor\faiss_e_means"
$stagingRoot = Join-Path $projectRoot ("vendor\.emeans-stage-" + [guid]::NewGuid().ToString("N"))
$sourceCommit = "6ccfd5f9be2283ebc037564bb45c6ba63b71e96a"
$sourceRoot = "https://raw.githubusercontent.com/facebookresearch/faiss/$sourceCommit"
$files = @("base.py", "e_means_torch.py", "torch_backend.py")
$expectedSha256 = @{
    "base.py" = "BCFFD7829EBFC25E1354308770B49E6F99D8566BA5E474A41A22E9EC414B17CD"
    "e_means_torch.py" = "CDDC5ED1F1121C96406055FE6F50E93C049D0E09BE27DC07DBE03E5C8E7EF4D3"
    "torch_backend.py" = "55757D984D9B6FCF8AC905C4FFD7B74B84467351DD7B212AA46CEE91BBC044D2"
    "LICENSE" = "52412D7BC7CE4157EA628BBAACB8829E0A9CB3C58F57F99176126BC8CF2BFC85"
}

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
        $actualHash = (Get-FileHash -LiteralPath (Join-Path $stagingRoot $file) -Algorithm SHA256).Hash
        if ($actualHash -ne $expectedSha256[$file]) {
            throw "SHA256 sumber e-means tidak cocok untuk $file; unduhan tidak dipasang."
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
