"""Detect formula / table / figure regions on a rendered page."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import cv2
import fitz
import numpy as np

from core.ir import BBox, BlockType, Provenance, QualitySignals, RegionBlock
from ingestion.models import TextSpan
from ingestion.preprocess import render_page
from ingestion.source_analysis import _extract_spans

_MATH_CHARS = set("=+-×÷·±≠≈≤≥<>∑∫∏√∞∂∇αβγδθλμπσφω^_{}\\/|()[]")
_MATH_TOKEN = re.compile(
    r"(\\frac|\\sum|\\int|\\sqrt|\\left|\\right|\\mathrm|\\text|"
    r"[Σ∑∫√∞±≤≥≠≈]|[=^_]|\d+[a-zA-Zа-яА-Я]|[a-zA-Z]\d)"
)


@dataclass
class DetectedRegion:
    type: BlockType
    bbox_pt: BBox  # PDF points
    bbox_px: tuple[int, int, int, int] | None = None  # x1,y1,x2,y2 on render
    score: float = 0.0
    method: str = "heuristic"
    notes: list[str] = field(default_factory=list)


def _pt_to_px(bbox: BBox, dpi: int) -> tuple[int, int, int, int]:
    s = dpi / 72.0
    return (
        int(bbox.x1 * s),
        int(bbox.y1 * s),
        int(bbox.x2 * s),
        int(bbox.y2 * s),
    )


def _px_to_pt(x1: int, y1: int, x2: int, y2: int, dpi: int) -> BBox:
    s = 72.0 / dpi
    return BBox(x1=x1 * s, y1=y1 * s, x2=x2 * s, y2=y2 * s)


def _line_math_score(text: str) -> float:
    t = text.strip()
    if len(t) < 2:
        return 0.0
    if _MATH_TOKEN.search(t):
        return 0.9
    letters = sum(ch.isalpha() for ch in t)
    mathish = sum(ch in _MATH_CHARS or ch.isdigit() for ch in t)
    if letters + mathish == 0:
        return 0.0
    ratio = mathish / (letters + mathish)
    # short equation-like lines
    if len(t) <= 80 and ratio >= 0.35 and ("=" in t or "^" in t or "_" in t):
        return min(1.0, 0.5 + ratio)
    if ratio >= 0.55 and letters < 12:
        return min(1.0, ratio)
    return 0.0


def detect_formula_regions_from_spans(
    spans: list[TextSpan],
    *,
    dpi: int,
    page_w: float,
    page_h: float,
) -> list[DetectedRegion]:
    """Merge contiguous math-like text spans into formula boxes."""
    scored: list[tuple[TextSpan, float]] = []
    for sp in spans:
        sc = _line_math_score(sp.text)
        if sc >= 0.55:
            scored.append((sp, sc))
    if not scored:
        return []

    # cluster by vertical proximity
    scored.sort(key=lambda x: x[0].bbox[1])
    clusters: list[list[tuple[TextSpan, float]]] = []
    cur: list[tuple[TextSpan, float]] = []
    last_y2 = -1e9
    for sp, sc in scored:
        y1 = sp.bbox[1]
        if cur and y1 - last_y2 > 18:
            clusters.append(cur)
            cur = []
        cur.append((sp, sc))
        last_y2 = sp.bbox[3]
    if cur:
        clusters.append(cur)

    out: list[DetectedRegion] = []
    for cl in clusters:
        xs0 = [s.bbox[0] for s, _ in cl]
        ys0 = [s.bbox[1] for s, _ in cl]
        xs1 = [s.bbox[2] for s, _ in cl]
        ys1 = [s.bbox[3] for s, _ in cl]
        pad = 4.0
        bbox = BBox(
            x1=max(0.0, min(xs0) - pad),
            y1=max(0.0, min(ys0) - pad),
            x2=min(page_w, max(xs1) + pad),
            y2=min(page_h, max(ys1) + pad),
        )
        # skip huge "math" regions (likely misclassified text)
        area = (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1)
        if area > 0.25 * page_w * page_h:
            continue
        score = float(np.mean([sc for _, sc in cl]))
        out.append(
            DetectedRegion(
                type=BlockType.FORMULA,
                bbox_pt=bbox,
                bbox_px=_pt_to_px(bbox, dpi),
                score=score,
                method="span_math_heuristic",
                notes=[f"n_spans={len(cl)}"],
            )
        )
    return out


def detect_table_regions_cv(
    image_bgr: np.ndarray,
    *,
    dpi: int,
    page_w: float,
    page_h: float,
) -> list[DetectedRegion]:
    """Find table-like grids via line detection (OpenCV)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10
    )
    # horizontal / vertical lines
    hk = max(w // 40, 20)
    vk = max(h // 40, 20)
    hor = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))
    )
    ver = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk))
    )
    grid = cv2.add(hor, ver)
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[DetectedRegion] = []
    page_area = float(w * h)
    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        area = ww * hh
        if area < 0.02 * page_area or area > 0.65 * page_area:
            continue
        if ww < w * 0.25 or hh < h * 0.05:
            continue
        # require both h and v line energy inside box
        roi_h = hor[y : y + hh, x : x + ww]
        roi_v = ver[y : y + hh, x : x + ww]
        if roi_h.mean() < 2 and roi_v.mean() < 2:
            continue
        if roi_h.mean() < 0.5 or roi_v.mean() < 0.5:
            continue
        bbox_pt = _px_to_pt(x, y, x + ww, y + hh, dpi)
        out.append(
            DetectedRegion(
                type=BlockType.TABLE,
                bbox_pt=bbox_pt,
                bbox_px=(x, y, x + ww, y + hh),
                score=min(1.0, 0.4 + (roi_h.mean() + roi_v.mean()) / 40.0),
                method="cv_line_grid",
            )
        )
    return out


def detect_table_regions_from_spans(
    spans: list[TextSpan],
    *,
    dpi: int,
    page_w: float,
    page_h: float,
) -> list[DetectedRegion]:
    """Column-aligned short tokens → table candidate bbox."""
    if len(spans) < 25 or page_w <= 0:
        return []
    xs = np.array([0.5 * (s.bbox[0] + s.bbox[2]) for s in spans])
    hist, edges = np.histogram(xs, bins=14)
    peaks = [i for i, v in enumerate(hist) if v >= max(4, hist.max() * 0.4)]
    short = [s for s in spans if len(s.text.strip()) <= 14]
    if len(peaks) < 3 or len(short) / len(spans) < 0.4:
        return []
    ys0 = [s.bbox[1] for s in short]
    ys1 = [s.bbox[3] for s in short]
    xs0 = [s.bbox[0] for s in short]
    xs1 = [s.bbox[2] for s in short]
    bbox = BBox(
        x1=max(0.0, min(xs0) - 6),
        y1=max(0.0, min(ys0) - 6),
        x2=min(page_w, max(xs1) + 6),
        y2=min(page_h, max(ys1) + 6),
    )
    # Span alignment across most of a scanned page is usually body text,
    # not a table. Keep only compact candidates; grid detection handles
    # genuinely large ruled tables separately.
    area = (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1)
    if area > 0.45 * page_w * page_h:
        return []
    return [
        DetectedRegion(
            type=BlockType.TABLE,
            bbox_pt=bbox,
            bbox_px=_pt_to_px(bbox, dpi),
            score=0.65,
            method="span_alignment",
            notes=[f"peaks={len(peaks)} short={len(short)}"],
        )
    ]


def detect_figure_regions(
    image_bgr: np.ndarray,
    text_mask_spans: list[TextSpan],
    *,
    dpi: int,
    page_w: float,
) -> list[DetectedRegion]:
    """Large ink regions with little overlapping text → figure candidates."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    # ink
    ink = gray < np.median(gray) - 15
    # dilate text boxes to exclude
    text_mask = np.zeros((h, w), dtype=np.uint8)
    s = dpi / 72.0
    for sp in text_mask_spans:
        x0, y0, x1, y1 = [int(v * s) for v in sp.bbox]
        x0, y0 = max(0, x0 - 2), max(0, y0 - 2)
        x1, y1 = min(w, x1 + 2), min(h, y1 + 2)
        if x1 > x0 and y1 > y0:
            text_mask[y0:y1, x0:x1] = 1
    nontext_ink = ink & (text_mask == 0)
    # connected components on non-text ink
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        nontext_ink.astype(np.uint8) * 255, connectivity=8
    )
    out: list[DetectedRegion] = []
    page_area = float(w * h)
    for i in range(1, num):
        x, y, ww, hh, area = stats[i]
        if area < 0.03 * page_area:
            continue
        if ww < 80 or hh < 80:
            continue
        # skip full-page blobs
        if area > 0.55 * page_area:
            continue
        # aspect: figures often not extremely thin
        if ww / max(hh, 1) > 8 or hh / max(ww, 1) > 8:
            continue
        bbox_pt = _px_to_pt(x, y, x + ww, y + hh, dpi)
        out.append(
            DetectedRegion(
                type=BlockType.FIGURE,
                bbox_pt=bbox_pt,
                bbox_px=(x, y, x + ww, y + hh),
                score=min(1.0, area / (0.15 * page_area)),
                method="nontext_ink_cc",
            )
        )
    # keep top-N by area
    out.sort(key=lambda r: (r.bbox_px[2] - r.bbox_px[0]) * (r.bbox_px[3] - r.bbox_px[1]), reverse=True)
    return out[:5]


def _iou_pt(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    bb = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    return inter / (aa + bb - inter + 1e-9)


def merge_regions(regions: list[DetectedRegion], iou_thr: float = 0.4) -> list[DetectedRegion]:
    """NMS-like merge within the same type."""
    kept: list[DetectedRegion] = []
    for r in sorted(regions, key=lambda x: -x.score):
        drop = False
        for k in kept:
            if k.type == r.type and _iou_pt(k.bbox_pt, r.bbox_pt) >= iou_thr:
                drop = True
                break
        if not drop:
            kept.append(r)
    return kept


def detect_page_regions(
    doc: fitz.Document,
    page_number: int,
    *,
    doc_id: str,
    dpi: int = 200,
    use_ocr_fallback: bool = True,
) -> tuple[np.ndarray, list[DetectedRegion], list[RegionBlock]]:
    """Detect formula/table/figure regions; return render, raw detections, IR blocks."""
    rendered = render_page(doc, page_number, dpi=dpi)
    page = doc[page_number - 1]
    page_w, page_h = float(page.rect.width), float(page.rect.height)
    _, spans = _extract_spans(page)

    # Scanned pages often have empty text layer — build pseudo-spans via OCR
    if use_ocr_fallback and len(spans) < 5:
        spans = _ocr_pseudo_spans(rendered.image_bgr, dpi=dpi, doc_id=doc_id, page=page_number)

    regions: list[DetectedRegion] = []
    regions.extend(
        detect_formula_regions_from_spans(spans, dpi=dpi, page_w=page_w, page_h=page_h)
    )
    regions.extend(
        detect_table_regions_from_spans(spans, dpi=dpi, page_w=page_w, page_h=page_h)
    )
    regions.extend(
        detect_table_regions_cv(
            rendered.image_bgr, dpi=dpi, page_w=page_w, page_h=page_h
        )
    )
    regions.extend(
        detect_figure_regions(
            rendered.image_bgr, spans, dpi=dpi, page_w=page_w
        )
    )
    regions = merge_regions(regions)

    blocks: list[RegionBlock] = []
    for i, r in enumerate(regions):
        blocks.append(
            RegionBlock(
                doc_id=doc_id,
                page=page_number,
                region_id=f"{r.type.value}_{i:02d}",
                type=r.type,
                bbox=r.bbox_pt,
                source="mixed",
                quality=QualitySignals(confidence=r.score),
                content={"detection_notes": r.notes},
                provenance=Provenance(method=r.method, confidence=r.score),
            )
        )
    return rendered.image_bgr, regions, blocks


def _ocr_pseudo_spans(
    image_bgr: np.ndarray,
    *,
    dpi: int,
    doc_id: str,
    page: int,
) -> list[TextSpan]:
    try:
        from ingestion.ocr_rapid import RapidOCRConfig, recognize_page_image
        from core.config import get_settings

        settings = get_settings()
        ocr = recognize_page_image(
            image_bgr,
            dpi=dpi,
            page=page,
            doc_id=doc_id,
            cfg=RapidOCRConfig(
                use_gpu=settings.enable_gpu_ocr,
                max_side_len=settings.ocr_max_side_len,
                model_dir=str(settings.ocr_model_dir),
            ),
        )
        spans: list[TextSpan] = []
        for ln in ocr.lines:
            spans.append(
                TextSpan(
                    text=ln.text,
                    bbox=ln.bbox_pt,
                    font_size=None,
                )
            )
        return spans
    except Exception:
        return []


def crop_region_bgr(
    image_bgr: np.ndarray,
    region: DetectedRegion,
    *,
    pad: int = 8,
) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    if region.bbox_px is None:
        raise ValueError("bbox_px required for crop")
    x1, y1, x2, y2 = region.bbox_px
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    return image_bgr[y1:y2, x1:x2].copy()
