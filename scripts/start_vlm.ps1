# Start local VLM for rag_system (llama.cpp llama-server).
# Run in PowerShell on the WORK PC. Keep this window open while extracting.
#
# Prereqs:
#   1) llama-server on PATH (or set $env:LLAMA_SERVER_BIN)
#   2) GPU drivers / enough VRAM (3B Q4 ~6–8 GB; 7B Q4 ~10–14 GB)
#   3) Internet once for -hf download, OR local GGUF + mmproj paths in .env
#
# Quickest (auto-download from Hugging Face via llama.cpp):
#   .\scripts\start_vlm.ps1
#
# Offline / corporate network: put files under data\models\vlm\ and:
#   $env:VLM_GGUF_PATH="data\models\vlm\Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"
#   $env:VLM_MMPROJ_PATH="data\models\vlm\mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf"
#   .\scripts\start_vlm.ps1

$ErrorActionPreference = "Stop"

$Bin = if ($env:LLAMA_SERVER_BIN) { $env:LLAMA_SERVER_BIN } else { "llama-server" }
$HostAddr = if ($env:MODEL_HOST) { $env:MODEL_HOST } else { "127.0.0.1" }
$Port = if ($env:MODEL_PORT) { $env:MODEL_PORT } else { "8080" }
$Ctx = if ($env:VLM_CTX) { $env:VLM_CTX } else { "8192" }
$GpuLayers = if ($env:VLM_NGL) { $env:VLM_NGL } else { "99" }

# Default: small VL model suitable as recovery layer (roadmap: modest GPU).
$HfRepo = if ($env:VLM_HF_REPO) { $env:VLM_HF_REPO } else { "ggml-org/Qwen2.5-VL-3B-Instruct-GGUF" }

Write-Host "Binary : $Bin"
Write-Host "Listen : http://${HostAddr}:${Port}"

if ($env:VLM_GGUF_PATH -and $env:VLM_MMPROJ_PATH) {
    if (-not (Test-Path $env:VLM_GGUF_PATH)) { throw "VLM_GGUF_PATH not found: $($env:VLM_GGUF_PATH)" }
    if (-not (Test-Path $env:VLM_MMPROJ_PATH)) { throw "VLM_MMPROJ_PATH not found: $($env:VLM_MMPROJ_PATH)" }
    Write-Host "Mode   : local GGUF + mmproj"
    Write-Host "Model  : $($env:VLM_GGUF_PATH)"
    Write-Host "Mmproj : $($env:VLM_MMPROJ_PATH)"
    & $Bin `
        -m $env:VLM_GGUF_PATH `
        --mmproj $env:VLM_MMPROJ_PATH `
        --host $HostAddr `
        --port $Port `
        -c $Ctx `
        -ngl $GpuLayers
}
else {
    Write-Host "Mode   : Hugging Face auto (-hf $HfRepo)"
    Write-Host "Tip    : for 7B set `$env:VLM_HF_REPO='ggml-org/Qwen2.5-VL-7B-Instruct-GGUF'"
    & $Bin `
        -hf $HfRepo `
        --host $HostAddr `
        --port $Port `
        -c $Ctx `
        -ngl $GpuLayers
}
