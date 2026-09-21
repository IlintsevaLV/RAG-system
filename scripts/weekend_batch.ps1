# Weekend batch for work PC — full text extract + garbage scan (+ optional regions).
# Paste into PowerShell OR:  powershell -ExecutionPolicy Bypass -File scripts\weekend_batch.ps1
#
# Does NOT start VLM. Leave VLM for a separate targeted run if llama-server is up.

$ErrorActionPreference = "Continue"
$Root = "C:\Users\l.ilyintseva\Desktop\RAG-system"
Set-Location $Root

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogDir = Join-Path $Root "data\ir\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogFile = Join-Path $LogDir "weekend_$stamp.log"

function Write-Log([string]$msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Write-Host $line
    Add-Content -Path $LogFile -Value $line -Encoding UTF8
}

Write-Log "START weekend batch"
Write-Log "Log: $LogFile"

# --- venv ---
$Activate = Join-Path $Root ".venv\Scripts\Activate.ps1"
if (-not (Test-Path $Activate)) {
    Write-Log "ERROR: .venv not found at $Activate"
    exit 1
}
. $Activate
Write-Log "venv activated: $((Get-Command python).Source)"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

# --- resume-friendly: keep finished *_pages.json; only remove obviously incomplete ---
$PagesDir = Join-Path $Root "data\ir\pages"
$BackupDir = Join-Path $Root "data\ir\pages_backup_$stamp"
New-Item -ItemType Directory -Force -Path $BackupDir | Out-Null
$moved = 0
Get-ChildItem -Path $PagesDir -Filter "*_pages.json" -ErrorAction SilentlyContinue | ForEach-Object {
    try {
        $n = (Get-Content $_.FullName -Raw -Encoding UTF8 | ConvertFrom-Json).pages.Count
    } catch {
        $n = 0
    }
    # Incomplete sample runs were ~1-8 pages; keep full docs (e.g. _1954=254)
    if ($n -gt 0 -and $n -lt 20) {
        Move-Item $_.FullName -Destination $BackupDir -Force
        $moved++
        Write-Log "Moved incomplete $($_.Name) (pages=$n) -> backup"
    } else {
        Write-Log "Keep $($_.Name) (pages=$n) for --resume"
    }
}
Write-Log "Moved $moved incomplete JSON files"

# --- 1) FULL text extract (all pages of all PDFs in data\raw) ---
# No --pages = entire document. No --enable-vlm = OCR fallback for C/D.
Write-Log "=== STEP 1: run_extract_pages (full corpus, no VLM, --resume) ==="
python -m ingestion.run_extract_pages data\raw --resume 2>&1 | Tee-Object -FilePath $LogFile -Append -Encoding utf8
$code1 = $LASTEXITCODE
Write-Log "STEP 1 exit code: $code1"

# --- 2) Full garbage text-layer scan ---
Write-Log "=== STEP 2: scan_garbage_text_layer --full --no-visual ==="
python -m ingestion.scan_garbage_text_layer data\raw --full --no-visual 2>&1 | Tee-Object -FilePath $LogFile -Append -Encoding utf8
$code2 = $LASTEXITCODE
Write-Log "STEP 2 exit code: $code2"

# --- 3) Regions on known hard pages (Docling tables; formulas empty without UniMERNet/VLM) ---
Write-Log "=== STEP 3: run_regions (selected pages) ==="
$regionJobs = @(
    @{ Pdf = "data\raw\US_Army_Materiel_Command_Engineering_Design_Handbook.pdf"; Pages = "83,246,572" }
)
foreach ($job in $regionJobs) {
    if (Test-Path $job.Pdf) {
        Write-Log "regions: $($job.Pdf) pages=$($job.Pages)"
        python -m ingestion.run_regions $job.Pdf --pages $job.Pages 2>&1 | Tee-Object -FilePath $LogFile -Append -Encoding utf8
        Write-Log "regions exit: $LASTEXITCODE"
    } else {
        Write-Log "SKIP missing PDF: $($job.Pdf)"
    }
}

Write-Log "DONE weekend batch"
Write-Log "Results:"
Write-Log "  data\ir\pages\*_pages.json + page_*.txt"
Write-Log "  data\ir\garbage_text_layer_candidates.csv"
Write-Log "  data\ir\garbage_text_layer_report.json"
Write-Log "  data\ir\regions\*_regions.json"
Write-Log "On Monday: git add -f data/ir/pages/*_pages.json data/ir/regions/*.json data/ir/garbage_text_layer_*.csv data/ir/garbage_text_layer_*.json ; commit ; push"

# Keep window open if double-clicked
if ($Host.Name -eq "ConsoleHost") {
    Write-Host ""
    Write-Host "Finished. Log: $LogFile"
}
