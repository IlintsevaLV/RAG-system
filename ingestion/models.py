"""Source-analysis, preprocess and OCR result models (roadmap_new §1–2)."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class PageRoute(str, Enum):
    TEXT_LAYER = "text_layer"
    OCR = "ocr"


class PageClass(str, Enum):
    """Page handling class after TLQ + lexical analysis."""

    A = "A"  # clean text layer
    B = "B"  # usable layer with light damage / layout quirks
    C = "C"  # garbage/missing layer → preprocess + VLM
    D = "D"  # unreadable after VLM/OCR (set by quality gate)


class ExtractStatus(str, Enum):
    OK = "ok"
    SUSPICIOUS = "suspicious"
    FAILED = "failed"
    NEEDS_VLM = "needs_vlm"


class ImageQualityLevel(str, Enum):
    GOOD = "good"
    MEDIUM = "medium"
    BAD = "bad"


class TLQComponents(BaseModel):
    printable_ratio: float
    language_score: float
    geometry_score: float
    text_density_score: float
    visual_agreement: float | None = None
    lexical_quality: float | None = None
    garbage_score: float | None = None
    dict_hit: float | None = None
    typo_ratio: float | None = None
    oov_hard: float | None = None


class TLQResult(BaseModel):
    score: float
    components: TLQComponents
    has_text_layer: bool
    char_count: int
    span_count: int
    route: PageRoute
    garbage_veto: bool = False
    lang_hint: str | None = None
    sample_bad_tokens: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class TextSpan(BaseModel):
    text: str
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1 in PDF points
    font_size: float | None = None


class IQSComponents(BaseModel):
    resolution_score: float
    blur_score: float
    noise_score: float
    contrast_score: float
    skew_score: float
    compression_score: float


class IQSResult(BaseModel):
    score: float
    level: ImageQualityLevel
    components: IQSComponents
    estimated_skew_deg: float = 0.0
    notes: list[str] = Field(default_factory=list)


class OCRLine(BaseModel):
    text: str
    text_raw: str = ""
    bbox_px: tuple[float, float, float, float]
    bbox_pt: tuple[float, float, float, float]
    confidence: float = 0.0
    engine: str = "rapidocr"


class OCRPageResult(BaseModel):
    doc_id: str
    page: int
    lines: list[OCRLine] = Field(default_factory=list)
    text: str = ""
    mean_confidence: float = 0.0
    line_count: int = 0
    used_bands: bool = False
    image_size: tuple[int, int] | None = None  # width, height px
    dpi: int | None = None


class PageSourceAnalysis(BaseModel):
    doc_id: str
    page: int  # 1-based
    width_pt: float
    height_pt: float
    tlq: TLQResult
    text_spans: list[TextSpan] = Field(default_factory=list)
    extracted_text: str = ""
    iqs: IQSResult | None = None
    preprocess_plan: list[str] = Field(default_factory=list)
    ocr: OCRPageResult | None = None


class DocumentSourceAnalysis(BaseModel):
    doc_id: str
    source_path: str
    page_count: int
    tlq_threshold: float
    pages: list[PageSourceAnalysis]
    summary: dict[str, Any] = Field(default_factory=dict)


class LayoutHints(BaseModel):
    n_columns: int = 1
    likely_table: bool = False
    unusual_layout: bool = False
    notes: list[str] = Field(default_factory=list)
    vlm_notes: str | None = None


class PageTextExtract(BaseModel):
    """RAG-ready page text after class A/B/C/D extraction."""

    doc_id: str
    page: int
    page_class: PageClass
    status: ExtractStatus
    text: str = ""
    text_source: str = ""  # layer | layer+ocr | vlm | ocr_fallback
    layout: LayoutHints = Field(default_factory=LayoutHints)
    tlq_score: float | None = None
    lexical_quality: float | None = None
    iqs_level: str | None = None
    confidence: float | None = None
    preview_path: str | None = None
    notes: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)


class DocumentTextExtract(BaseModel):
    doc_id: str
    source_path: str
    page_count: int
    pages: list[PageTextExtract]
    summary: dict[str, Any] = Field(default_factory=dict)
