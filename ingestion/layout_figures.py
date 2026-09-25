"""Figure boxes from Docling's layout model (RT-DETR), not caption heuristics.

Layout runs on the rendered page. OCR and table structure stay off: we only
need Picture and Caption clusters. The converter is loaded once.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.ir import BBox
from core.logger import get_logger
from ingestion.preprocess import imwrite_unicode

log = get_logger("layout_figures")

_CONVERTER = None
_LOAD_FAILED = False


@dataclass
class LayoutHit:
    label: str  # picture | caption
    bbox_pt: BBox
    score: float
    text: str = ""


def _label_name(label: object) -> str:
    raw = getattr(label, "value", None) or getattr(label, "name", None) or str(label)
    name = str(raw).lower()
    if "picture" in name or name.endswith("figure"):
        return "picture"
    if "caption" in name:
        return "caption"
    return name


def _get_converter():
    global _CONVERTER, _LOAD_FAILED
    if _CONVERTER is not None or _LOAD_FAILED:
        return _CONVERTER
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter

        opts = PdfPipelineOptions()
        opts.do_ocr = False
        opts.do_table_structure = False
        format_options = {}
        try:
            from docling.document_converter import ImageFormatOption

            format_options[InputFormat.IMAGE] = ImageFormatOption(pipeline_options=opts)
        except ImportError:
            from docling.document_converter import PdfFormatOption

            format_options[InputFormat.IMAGE] = PdfFormatOption(pipeline_options=opts)
        _CONVERTER = DocumentConverter(format_options=format_options)
    except Exception as exc:  # noqa: BLE001
        _LOAD_FAILED = True
        log.warning("docling_layout_unavailable", error=str(exc))
        _CONVERTER = None
    return _CONVERTER


def _to_top_left(bbox: object, page_h: float) -> tuple[float, float, float, float] | None:
    if hasattr(bbox, "to_top_left_origin"):
        try:
            bbox = bbox.to_top_left_origin(page_h)
        except Exception:
            pass
    l = getattr(bbox, "l", None)
    t = getattr(bbox, "t", None)
    r = getattr(bbox, "r", None)
    b = getattr(bbox, "b", None)
    if None in (l, t, r, b):
        return None
    x1, x2 = float(min(l, r)), float(max(l, r))
    y1, y2 = float(min(t, b)), float(max(t, b))
    return x1, y1, x2, y2


def _scale_box(
    box: tuple[float, float, float, float],
    *,
    src_w: float,
    src_h: float,
    page_w: float,
    page_h: float,
) -> BBox:
    sx = page_w / src_w if src_w else 1.0
    sy = page_h / src_h if src_h else 1.0
    x1, y1, x2, y2 = box
    return BBox(x1=x1 * sx, y1=y1 * sy, x2=x2 * sx, y2=y2 * sy)


def _cluster_text(cluster: object) -> str:
    parts: list[str] = []
    for cell in getattr(cluster, "cells", None) or []:
        text = getattr(cell, "text", None) or ""
        if str(text).strip():
            parts.append(str(text).strip())
    return " ".join(parts)[:200]


def docling_layout_hits(
    image_bgr: np.ndarray,
    *,
    page_w: float,
    page_h: float,
) -> list[LayoutHit]:
    """Picture and Caption boxes in PDF points, origin top-left."""
    converter = _get_converter()
    if converter is None or image_bgr is None or image_bgr.size == 0:
        return []
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    try:
        if not imwrite_unicode(tmp_path, image_bgr):
            return []
        result = converter.convert(str(tmp_path))
    except Exception as exc:  # noqa: BLE001
        log.warning("docling_layout_fail", error=str(exc))
        return []
    finally:
        tmp_path.unlink(missing_ok=True)

    pages = list(getattr(result, "pages", None) or [])
    hits: list[LayoutHit] = []
    if pages:
        page = pages[0]
        size = getattr(page, "size", None)
        src_w = float(getattr(size, "width", 0) or image_bgr.shape[1])
        src_h = float(getattr(size, "height", 0) or image_bgr.shape[0])
        layout = getattr(getattr(page, "predictions", None), "layout", None)
        clusters = list(getattr(layout, "clusters", None) or [])
        for cluster in clusters:
            kind = _label_name(getattr(cluster, "label", ""))
            if kind not in {"picture", "caption"}:
                continue
            raw = _to_top_left(cluster.bbox, src_h)
            if raw is None:
                continue
            hits.append(
                LayoutHit(
                    label=kind,
                    bbox_pt=_scale_box(
                        raw, src_w=src_w, src_h=src_h, page_w=page_w, page_h=page_h
                    ),
                    score=float(getattr(cluster, "confidence", 0.7) or 0.7),
                    text=_cluster_text(cluster),
                )
            )
    if any(h.label == "picture" for h in hits):
        return hits

    # Assembled pictures, if the pipeline dropped raw clusters.
    doc = getattr(result, "document", None)
    for picture in getattr(doc, "pictures", None) or []:
        for prov in getattr(picture, "prov", None) or []:
            raw = _to_top_left(getattr(prov, "bbox", None), page_h)
            if raw is None:
                continue
            hits.append(
                LayoutHit(label="picture", bbox_pt=BBox(x1=raw[0], y1=raw[1], x2=raw[2], y2=raw[3]), score=0.7)
            )
    return hits
