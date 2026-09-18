"""Application settings (pydantic-settings)."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "dev"
    log_level: str = "INFO"

    normativka_dir: Path = Path("./data/raw")
    ir_dir: Path = Path("./data/ir")
    cache_dir: Path = Path("./data/cache")

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "normativka"

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "changeme"

    llm_gguf_path: str = ""
    vlm_gguf_path: str = ""
    vlm_mmproj_path: str = ""
    llama_server_bin: str = "llama-server"
    model_host: str = "127.0.0.1"
    model_port: int = 8080

    enable_vlm: bool = False
    enable_gpu_ocr: bool = False  # CUDA EP often silent-fails; enable after smoke on work PC
    page_confidence_threshold: float = 0.85

    # Source analysis / preprocess (roadmap_new §1–2)
    tlq_threshold: float = 0.65
    tlq_enable_visual_agreement: bool = True
    render_dpi: int = 200
    render_dpi_bad: int = 300
    preprocess_enable_binarize: bool = False

    # Lexical garbage detection for embedded text layer
    # Calibrated so OCR-like layers (long-token OOV ~0.30) veto; clean layers (~0.06) pass.
    lexical_veto_threshold: float = 0.75
    lexical_min_tokens: int = 25

    # Page classes A/B/C/D extraction
    class_b_lex_min: float = 0.62
    class_b_dict_min: float = 0.58
    ab_always_ocr_check: bool = False
    ab_enable_vlm_check: bool = True  # used only if ENABLE_VLM=true
    ab_ocr_agree_min: float = 0.45
    ab_escalate_agree: float = 0.50  # B→C if layer vs OCR Jaccard below this
    cd_ocr_fallback: bool = True  # if VLM off/fails, RapidOCR after preprocess
    cd_fail_ocr_conf: float = 0.55
    vlm_timeout_s: float = 180.0

    # RapidOCR
    ocr_model_dir: Path = Path("./data/models/ocr")
    ocr_max_side_len: int = 4000
    ocr_band_trigger_px: int = 2800
    ocr_band_height: int = 1600

    # Special regions: formulas / tables / figures
    enable_unimernet: bool = False
    unimernet_model: str = "Wanderhub/UniMERNet"
    formula_vlm_recovery: bool = True
    table_enable_docling: bool = True
    table_enable_ppstructure: bool = True
    table_enable_img2table: bool = True
    table_accept_threshold: float = 0.45
    table_vlm_recovery: bool = True
    figure_vlm_caption: bool = True

    embedding_model: str = "intfloat/multilingual-e5-small"
    rerank_enabled: bool = False

    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1])

    def ensure_dirs(self) -> None:
        for path in (self.normativka_dir, self.ir_dir, self.cache_dir):
            path.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    return Settings()
