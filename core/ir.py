"""Unified Document IR envelopes (roadmap_new §8)."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class BlockType(str, Enum):
    TEXT = "text"
    FORMULA = "formula"
    TABLE = "table"
    FIGURE = "figure"  # generic image / photo / diagram (non-chart)
    CHART = "chart"
    DRAWING = "drawing"


class BBox(BaseModel):
    """Page coordinates in PDF points (origin top-left unless noted)."""

    x1: float
    y1: float
    x2: float
    y2: float


class Provenance(BaseModel):
    model: str | None = None
    version: str | None = None
    method: str | None = None
    confidence: float | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class QualitySignals(BaseModel):
    """Multi-signal quality; do not rely on a single confidence."""

    confidence: float | None = None
    coverage: float | None = None
    structural_ok: bool | None = None
    suspicion: float | None = None
    notes: list[str] = Field(default_factory=list)


class RegionBlock(BaseModel):
    """Common envelope for every layout region."""

    doc_id: str
    page: int
    region_id: str
    type: BlockType
    bbox: BBox
    tiles: list[BBox] = Field(default_factory=list)
    source: str = "raster"  # text_layer | vector | raster | mixed
    quality: QualitySignals = Field(default_factory=QualitySignals)
    content: dict[str, Any] = Field(default_factory=dict)
    provenance: Provenance = Field(default_factory=Provenance)


class Citation(BaseModel):
    """What the RAG answer must return to the user."""

    doc_id: str
    document_name: str
    page: int
    bbox: BBox | None = None
    region_id: str | None = None
    quote: str | None = None
