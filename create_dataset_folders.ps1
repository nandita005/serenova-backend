# ============================================================
# SERENOVA - DATASET DIRECTORY SETUP
# ============================================================

$Root = "D:\serenova_backend\data\raw\external"

Write-Host ""
Write-Host "Creating Serenova dataset directories..." -ForegroundColor Cyan
Write-Host "Root: $Root"
Write-Host ""

# Create root
New-Item -ItemType Directory -Force -Path $Root | Out-Null

# Dataset directories
$folders = @(
    "bidmc",
    "ctu_uhb_ctg",
    "fetal_ecg",
    "fetal_ecg\nifecgdb",
    "fetal_ecg\nifeadb",
    "shiraz_fetal_heart_sound",
    "mit_bih_nsr",
    "maternal_health_risk",
    "diabetes",
    "fetal_health",
    "postnatal"
)

foreach ($folder in $folders) {

    $path = Join-Path $Root $folder

    New-Item -ItemType Directory -Force -Path $path | Out-Null

    Write-Host "[CREATED] $path" -ForegroundColor Green
}

Write-Host ""
Write-Host "============================================================"
Write-Host " SERENOVA DATASET DIRECTORIES CREATED"
Write-Host "============================================================"
Write-Host ""

# Display structure
tree $Root /F

Write-Host ""
Write-Host "Place your downloaded datasets into the corresponding folders."
Write-Host ""