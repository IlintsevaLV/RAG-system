"""Source-analysis and preprocess result models (roadmap_new §1–2)."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class PageRoute(str, Enum):
    TEXT_LAYER = "text_layer"
    OCR = "ocr"


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


class TLQResult(BaseModel):
    score: float
    components: TLQComponents
    has_text_layer: bool
    char_count: int
    span_count: int
    route: PageRoute
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


class DocumentSourceAnalysis(BaseModel):
    doc_id: str
    source_path: str
    page_count: int
    tlq_threshold: float
    pages: list[PageSourceAnalysis]
    summary: dict[str, Any] = Field(default_factory=dict)
