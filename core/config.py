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
    enable_gpu_ocr: bool = True
    page_confidence_threshold: float = 0.85

    # Source analysis / preprocess (roadmap_new §1–2)
    tlq_threshold: float = 0.65
    tlq_enable_visual_agreement: bool = True
    render_dpi: int = 200
    render_dpi_bad: int = 300
    preprocess_enable_binarize: bool = False

    embedding_model: str = "intfloat/multilingual-e5-small"
    rerank_enabled: bool = False

    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1])

    def ensure_dirs(self) -> None:
        for path in (self.normativka_dir, self.ir_dir, self.cache_dir):
            path.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    return Settings()
