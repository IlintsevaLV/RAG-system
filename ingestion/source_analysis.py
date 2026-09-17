"""Text-layer quality (TLQ) and PDF source analysis before preprocess.

TLQ =
  0.30 * printable_ratio
+ 0.20 * language_score
+ 0.20 * geometry_score
+ 0.15 * text_density_score
+ 0.15 * visual_agreement

Volume of text is NOT a quality signal by itself.
"""

from __future__ import annotations

import re
import string
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np

from ingestion.models import (
    DocumentSourceAnalysis,
    PageRoute,
    PageSourceAnalysis,
    TextSpan,
    TLQComponents,
    TLQResult,
)
from ingestion.text_layer_quality import lexical_quality

# Weights from roadmap_new
W_PRINTABLE = 0.30
W_LANGUAGE = 0.20
W_GEOMETRY = 0.20
W_DENSITY = 0.15
W_VISUAL = 0.15

_CYRILLIC = re.compile(r"[\u0400-\u04FF]")
_LATIN = re.compile(r"[A-Za-z]")
_PRINTABLE = set(string.printable) | set(
    "«»—–−°±×÷№§…ёЁАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюя"
)


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def printable_ratio(text: str) -> float:
    if not text:
        return 0.0
    ok = sum(1 for ch in text if ch in _PRINTABLE or ch.isspace())
    return _clamp01(ok / len(text))


def language_score(text: str) -> float:
    """Score coherent Cyrillic and/or Latin technical text (not symbol soup)."""
    if not text or not text.strip():
        return 0.0
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.35
    cyr = sum(1 for ch in letters if _CYRILLIC.match(ch))
    lat = sum(1 for ch in letters if _LATIN.match(ch))
    total = cyr + lat
    if total == 0:
        return 0.2
    cyr_share = cyr / total
    lat_share = lat / total

    # Russian body (+ Latin designations)
    if cyr_share >= 0.45:
        return _clamp01(0.65 + 0.35 * cyr_share)

    # English / Latin technical docs
    if lat_share >= 0.75:
        # Prefer word-like tokens over single-letter garbage
        tokens = re.findall(r"[A-Za-z]{2,}", text)
        if len(tokens) >= 3:
            return _clamp01(0.75 + 0.2 * min(1.0, len(tokens) / 20.0))
        return 0.45

    # Mixed / ambiguous
    if cyr_share >= 0.2:
        return 0.55
    return 0.3


def geometry_score(spans: list[TextSpan], page_w: float, page_h: float) -> float:
    if not spans:
        return 0.0
    if page_w <= 0 or page_h <= 0:
        return 0.0

    valid = 0
    area_sum = 0.0
    centers_y: list[float] = []
    for sp in spans:
        x0, y0, x1, y1 = sp.bbox
        if x1 > x0 and y1 > y0 and x0 >= -1 and y0 >= -1 and x1 <= page_w + 1 and y1 <= page_h + 1:
            valid += 1
            area_sum += (x1 - x0) * (y1 - y0)
            centers_y.append(0.5 * (y0 + y1))

    valid_ratio = valid / len(spans)

    # Reading order: y should be mostly non-decreasing when sorted by extraction order
    order_ok = 1.0
    if len(centers_y) >= 3:
        decreases = sum(1 for i in range(1, len(centers_y)) if centers_y[i] + 2 < centers_y[i - 1])
        order_ok = _clamp01(1.0 - decreases / (len(centers_y) - 1))

    # Duplicate / stacked garbage: many nearly identical bboxes
    dup_penalty = 0.0
    if len(spans) >= 8:
        rounded = [tuple(round(v, 1) for v in sp.bbox) for sp in spans]
        unique = len(set(rounded))
        dup_ratio = 1.0 - unique / len(rounded)
        dup_penalty = _clamp01(dup_ratio)

    # Unreasonable coverage (text boxes covering almost whole page many times)
    page_area = page_w * page_h
    coverage = area_sum / page_area if page_area else 0.0
    coverage_score = 1.0
    if coverage > 3.0:
        coverage_score = 0.2
    elif coverage > 1.5:
        coverage_score = 0.5
    elif coverage < 0.001 and len(spans) > 5:
        coverage_score = 0.4

    return _clamp01(0.45 * valid_ratio + 0.30 * order_ok + 0.25 * coverage_score * (1.0 - dup_penalty))


def text_density_score(char_count: int, page_w: float, page_h: float) -> float:
    """Reasonable density — not empty, not absurdly packed garbage."""
    if page_w <= 0 or page_h <= 0:
        return 0.0
    # PDF points; A4 ≈ 500k pt². Use chars per 1000 pt².
    area = page_w * page_h
    dens = char_count / (area / 1000.0)
    if char_count == 0:
        return 0.0
    # Empirically: sparse notes ~0.5–2, normal text ~3–25, garbage OCR dump >> 80
    if 2.0 <= dens <= 40.0:
        return 1.0
    if 0.3 <= dens < 2.0 or 40.0 < dens <= 80.0:
        return 0.55
    if dens < 0.3:
        return 0.25
    return 0.15


def visual_agreement_score(
    page: fitz.Page,
    spans: list[TextSpan],
    *,
    dpi: int = 72,
) -> tuple[float, list[str]]:
    """Cheap ink vs text-layer coverage agreement (not full OCR)."""
    notes: list[str] = []
    if not spans:
        return 0.0, ["no spans for visual agreement"]

    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n >= 3:
        gray = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]).astype(np.float32)
    else:
        gray = arr[:, :, 0].astype(np.float32)

    # Ink = darker than adaptive threshold near median
    med = float(np.median(gray))
    ink = gray < (med - 12.0)
    ink_ratio = float(ink.mean())

    # Text-layer mask from span bboxes
    mask = np.zeros((pix.height, pix.width), dtype=bool)
    for sp in spans:
        x0, y0, x1, y1 = sp.bbox
        xa, ya = int(x0 * zoom), int(y0 * zoom)
        xb, yb = int(x1 * zoom), int(y1 * zoom)
        xa, xb = max(0, min(xa, xb)), min(pix.width, max(xa, xb))
        ya, yb = max(0, min(ya, yb)), min(pix.height, max(ya, yb))
        if xb > xa and yb > ya:
            mask[ya:yb, xa:xb] = True

    text_cov = float(mask.mean())
    if text_cov < 1e-6:
        return 0.2, ["empty text mask"]

    ink_in_text = float((ink & mask).sum()) / float(mask.sum())
    # If text boxes contain almost no ink → ghost/wrong layer
    # If ink exists but text covers nothing of ink → incomplete layer
    ink_covered = float((ink & mask).sum()) / float(ink.sum() + 1e-6)

    score = _clamp01(0.55 * ink_in_text + 0.45 * min(1.0, ink_covered * 3.0))
    notes.append(f"ink_ratio={ink_ratio:.4f} text_cov={text_cov:.4f} ink_in_text={ink_in_text:.3f}")
    # Scanned pages often have high ink and empty/bad layer → already low
    if ink_ratio > 0.02 and text_cov < 0.005:
        score = min(score, 0.25)
        notes.append("ink present but almost no text coverage")
    return score, notes


def compute_tlq(
    text: str,
    spans: list[TextSpan],
    page_w: float,
    page_h: float,
    *,
    visual: float | None,
    threshold: float,
    lexical_veto_threshold: float = 0.75,
    lexical_min_tokens: int = 25,
) -> TLQResult:
    notes: list[str] = []
    has_layer = bool(text.strip()) or bool(spans)
    lex = lexical_quality(
        text,
        min_tokens_for_veto=lexical_min_tokens,
        veto_threshold=lexical_veto_threshold,
    )
    comp = TLQComponents(
        printable_ratio=printable_ratio(text),
        language_score=language_score(text),
        geometry_score=geometry_score(spans, page_w, page_h),
        text_density_score=text_density_score(len(text), page_w, page_h),
        visual_agreement=visual,
        lexical_quality=lex.lexical_quality,
        garbage_score=lex.garbage_score,
        dict_hit=lex.dict_hit,
        typo_ratio=lex.typo_ratio,
        oov_hard=lex.oov_hard,
    )
    notes.extend(lex.notes)

    if visual is None:
        # Renormalize without visual term
        w_sum = W_PRINTABLE + W_LANGUAGE + W_GEOMETRY + W_DENSITY
        score = (
            W_PRINTABLE * comp.printable_ratio
            + W_LANGUAGE * comp.language_score
            + W_GEOMETRY * comp.geometry_score
            + W_DENSITY * comp.text_density_score
        ) / w_sum
        notes.append("visual_agreement skipped; weights renormalized")
    else:
        score = (
            W_PRINTABLE * comp.printable_ratio
            + W_LANGUAGE * comp.language_score
            + W_GEOMETRY * comp.geometry_score
            + W_DENSITY * comp.text_density_score
            + W_VISUAL * visual
        )

    score = _clamp01(score)
    if not has_layer:
        score = 0.0
        notes.append("no text layer")

    # Volume alone must not force accept
    if len(text) > 2000 and score < threshold:
        notes.append("large text volume but TLQ below threshold (possible garbage layer)")

    route = PageRoute.TEXT_LAYER if has_layer and score >= threshold else PageRoute.OCR
    garbage_veto = bool(has_layer and lex.garbage_veto)
    if garbage_veto:
        route = PageRoute.OCR
        # Keep structural score visible, but mark veto in notes
        notes.append(
            f"garbage_veto overridden route→ocr "
            f"(lexical_quality={lex.lexical_quality:.3f}, weighted_tlq={score:.3f})"
        )

    return TLQResult(
        score=score,
        components=comp,
        has_text_layer=has_layer,
        char_count=len(text),
        span_count=len(spans),
        route=route,
        garbage_veto=garbage_veto,
        lang_hint=lex.lang_hint,
        sample_bad_tokens=lex.sample_bad_tokens,
        notes=notes,
    )


def _extract_spans(page: fitz.Page) -> tuple[str, list[TextSpan]]:
    data = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
    spans: list[TextSpan] = []
    parts: list[str] = []
    for block in data.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []):
            for sp in line.get("spans", []):
                t = sp.get("text") or ""
                if not t:
                    continue
                bbox = tuple(float(x) for x in sp["bbox"])  # type: ignore[misc]
                spans.append(
                    TextSpan(
                        text=t,
                        bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                        font_size=float(sp.get("size") or 0) or None,
                    )
                )
                parts.append(t)
            parts.append("\n")
    text = "".join(parts).strip()
    if not text:
        # Fallback plain extract
        text = (page.get_text("text") or "").strip()
    return text, spans


def analyze_page(
    page: fitz.Page,
    *,
    doc_id: str,
    page_number: int,
    tlq_threshold: float,
    enable_visual: bool,
    visual_dpi: int = 72,
    lexical_veto_threshold: float = 0.75,
    lexical_min_tokens: int = 25,
) -> PageSourceAnalysis:
    rect = page.rect
    w, h = float(rect.width), float(rect.height)
    text, spans = _extract_spans(page)
    notes_extra: list[str] = []
    visual: float | None = None
    if enable_visual and spans:
        visual, vnotes = visual_agreement_score(page, spans, dpi=visual_dpi)
        notes_extra.extend(vnotes)
    elif enable_visual and not spans:
        visual = 0.0
        notes_extra.append("visual_agreement=0 (no spans)")

    tlq = compute_tlq(
        text,
        spans,
        w,
        h,
        visual=visual,
        threshold=tlq_threshold,
        lexical_veto_threshold=lexical_veto_threshold,
        lexical_min_tokens=lexical_min_tokens,
    )
    tlq.notes.extend(notes_extra)

    return PageSourceAnalysis(
        doc_id=doc_id,
        page=page_number,
        width_pt=w,
        height_pt=h,
        tlq=tlq,
        text_spans=spans if tlq.route == PageRoute.TEXT_LAYER else [],
        extracted_text=text if tlq.route == PageRoute.TEXT_LAYER else text[:500],
    )


def analyze_pdf(
    path: str | Path,
    *,
    doc_id: str | None = None,
    pages: list[int] | None = None,
    tlq_threshold: float = 0.65,
    enable_visual: bool = True,
    lexical_veto_threshold: float = 0.75,
    lexical_min_tokens: int = 25,
) -> DocumentSourceAnalysis:
    path = Path(path)
    doc_id = doc_id or path.stem
    doc = fitz.open(path)
    try:
        total = doc.page_count
        if pages is None:
            page_nums = list(range(1, total + 1))
        else:
            page_nums = [p for p in pages if 1 <= p <= total]

        results: list[PageSourceAnalysis] = []
        for pno in page_nums:
            results.append(
                analyze_page(
                    doc[pno - 1],
                    doc_id=doc_id,
                    page_number=pno,
                    tlq_threshold=tlq_threshold,
                    enable_visual=enable_visual,
                    lexical_veto_threshold=lexical_veto_threshold,
                    lexical_min_tokens=lexical_min_tokens,
                )
            )

        routes = [p.tlq.route.value for p in results]
        summary = {
            "pages_analyzed": len(results),
            "route_text_layer": routes.count(PageRoute.TEXT_LAYER.value),
            "route_ocr": routes.count(PageRoute.OCR.value),
            "garbage_veto_pages": sum(1 for p in results if p.tlq.garbage_veto),
            "mean_tlq": round(float(np.mean([p.tlq.score for p in results])), 4) if results else 0.0,
            "min_tlq": round(float(min(p.tlq.score for p in results)), 4) if results else 0.0,
            "mean_lexical": round(
                float(
                    np.mean(
                        [
                            p.tlq.components.lexical_quality
                            for p in results
                            if p.tlq.components.lexical_quality is not None
                        ]
                    )
                ),
                4,
            )
            if any(p.tlq.components.lexical_quality is not None for p in results)
            else None,
        }
        return DocumentSourceAnalysis(
            doc_id=doc_id,
            source_path=str(path.resolve()),
            page_count=total,
            tlq_threshold=tlq_threshold,
            pages=results,
            summary=summary,
        )
    finally:
        doc.close()
