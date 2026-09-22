# scripts/run_page_tests.ps1
# Прогон тестовых страниц _1954.pdf по типам.

$ErrorActionPreference = "Continue"
$Root = "C:\Users\l.ilyintseva\Desktop\RAG-system"
Set-Location $Root

$PythonMain = ".\.venv\Scripts\python.exe"
$Pdf = "data\raw\_1954.pdf"

if (-not (Test-Path $Pdf)) {
    Write-Host "FATAL: PDF not found: $Pdf"
    exit 1
}

# env
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:ENABLE_VLM = "false"
$env:ENABLE_UNIMERNET = "true"
$env:UNIMERNET_CONFIG_PATH = ".\data\models\unimernet\unimernet_tiny.yaml"
$env:OCR_ENABLE_LATIN = "true"
$env:OCR_ENABLE_GREEK = "true"

# сон отключить
powercfg /change standby-timeout-ac 0 | Out-Null
powercfg /change hibernate-timeout-ac 0 | Out-Null

New-Item -ItemType Directory -Force "data\ir\logs" | Out-Null
New-Item -ItemType Directory -Force "data\ir\page_tests" | Out-Null

$Tests = @(
    @{ Name = "T1_title";         Pages = "1,2,3";         Args = @() },
    @{ Name = "T2_table_typo";    Pages = "3";             Args = @() },
    @{ Name = "T3_equations";     Pages = "34,36,37,40,41,44"; Args = @("--enable-unimernet") },
    @{ Name = "T4_greek";         Pages = "47,48,51";      Args = @("--enable-unimernet") },
    @{ Name = "T5_figures";       Pages = "9,11,15,20,25"; Args = @() },
    @{ Name = "T6_real_tables";   Pages = "61,68,69,189";  Args = @() },
    @{ Name = "T7_bibliography";  Pages = "246,247,250";   Args = @("--enable-unimernet") },
    @{ Name = "T8_toc";           Pages = "252,253,254";   Args = @() },
    @{ Name = "T9_twocolumn";     Pages = "23,24";         Args = @() },
    @{ Name = "T10_dense_text";   Pages = "10,30,102,190"; Args = @() }
)

$summary = @()

foreach ($t in $Tests) {
    $out = "data\ir\page_tests\$($t.Name)"
    $log = "data\ir\logs\page_tests_$($t.Name)_$(Get-Date -Format yyyyMMdd_HHmmss).log"

    Write-Host ""
    Write-Host "============================================"
    Write-Host "TEST $($t.Name) — pages $($t.Pages)"
    Write-Host "out: $out"
    Write-Host "log: $log"
    Write-Host "============================================"

    $start = Get-Date

    $args = @(
        "-m", "ingestion.run_regions",
        $Pdf,
        "--pages", $t.Pages,
        "--out-dir", $out
    ) + $t.Args

    & $PythonMain @args 2>&1 | Tee-Object -FilePath $log

    $elapsed = (Get-Date) - $start
    $summary += [PSCustomObject]@{
        Test = $t.Name
        Pages = $t.Pages
        Seconds = [math]::Round($elapsed.TotalSeconds, 1)
        OutDir = $out
        Log = $log
    }
}

Write-Host ""
Write-Host "===== SUMMARY ====="
$summary | Format-Table -AutoSize

$summary | Export-Csv -NoTypeInformation -Encoding UTF8 "data\ir\page_tests\_summary.csv"
Write-Host "Summary: data\ir\page_tests\_summary.csv"