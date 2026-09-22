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
    r"[Σ∑∫√∞±≤≥≠≈]|[=^_]|"
    r"[A-Za-zА-Яа-яЁё]\s*[\^_]\s*[A-Za-zА-Яа-яЁё0-9])"
)
_MATH_STRONG = re.compile(
    r"(=|\\frac|\\sum|\\int|\\sqrt|[Σ∑∫√∞±≤≥≠≈]|"
    r"[A-Za-zА-Яа-яЁё]\s*[\^_]\s*[A-Za-zА-Яа-яЁё0-9])"
)
_WORD = re.compile(r"[A-Za-zА-Яа-яЁё]{2,}")

_TOC_LEADER = re.compile(r"(\.{2,}|…|\. \. \.|\u2026)")
_TOC_END_PAGE = re.compile(r"(\d{1,3})\s*$")
_TOC_TITLE = re.compile(
    r"(содержани[ея]|оглавлени[ея]|содержание|contents|table of contents)",
    re.IGNORECASE,
)
_NUMERICISH = re.compile(r"\d")


def _is_numericish(text: str) -> bool:
    """Numeric/formula cell, not an arbitrary word matching a broad charset."""
    t = text.strip()
    return bool(t and _NUMERICISH.search(t))

REGION_PRIORITY = {
    BlockType.FIGURE: 3,
    BlockType.TABLE: 3,
    BlockType.FORMULA: 2,
}


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
    if len(t) < 2 or len(t) > 240:
        return 0.0
    words = _WORD.findall(t)
    letters = sum(ch.isalpha() for ch in t)
    mathish = sum(ch in _MATH_CHARS or ch.isdigit() for ch in t)
    if letters + mathish == 0:
        return 0.0
    ratio = mathish / (letters + mathish)
    strong = bool(_MATH_STRONG.search(t))
    if not strong and (len(words) >= 3 or len(t) > 40):
        return 0.0
    if strong and ratio >= 0.20:
        return min(1.0, 0.55 + ratio * 0.45)
    if _MATH_TOKEN.search(t) and ratio >= 0.55 and len(words) <= 3:
        return min(1.0, ratio)
    return 0.0


def _is_formula_fragment(span: TextSpan) -> bool:
    """Return whether a short OCR span can belong to a broken formula.

    OCR commonly emits a numerator, denominator, Greek token, or single
    variable as separate spans.  Do not treat ordinary words as fragments:
    only short spans containing digits, Greek/math glyphs, or compact Latin
    tokens qualify.
    """
    text = span.text.strip()
    if not text or len(text) > 24:
        return False
    words = _WORD.findall(text)
    if len(words) > 3:
        return False
    greek_or_math = sum(
        ch in _MATH_CHARS or ("\u0370" <= ch <= "\u03ff") for ch in text
    )
    digits = sum(ch.isdigit() for ch in text)
    latin = sum("A" <= ch <= "Z" or "a" <= ch <= "z" for ch in text)
    # A lone Cyrillic OCR token ("с", "где", ...) is not evidence by itself.
    return bool(
        greek_or_math
        or digits
        or (latin >= 1 and len(text) <= 8)
        or _MATH_STRONG.search(text)
    )


def merge_formula_spans(
    spans: list[TextSpan],
    *,
    y_tol: float = 40.0,
    x_gap_tol: float = 80.0,
) -> list[TextSpan]:
    """Build synthetic spans for formulas fragmented by OCR.

    This is intentionally additive: original spans remain untouched for
    tables and reading order.  A connected cluster must contain at least two
    formula signals and either a Greek/math glyph or both digits and Latin
    symbols, which prevents ordinary prose columns from becoming formulas.
    """
    candidates = [sp for sp in spans if _is_formula_fragment(sp)]
    if len(candidates) < 2:
        return []
    candidates.sort(key=lambda sp: (sp.bbox[1], sp.bbox[0]))

    groups: list[list[TextSpan]] = []
    for span in candidates:
        x1, y1, x2, y2 = span.bbox
        matched: list[int] = []
        for i, group in enumerate(groups):
            gx1 = min(s.bbox[0] for s in group)
            gy1 = min(s.bbox[1] for s in group)
            gx2 = max(s.bbox[2] for s in group)
            gy2 = max(s.bbox[3] for s in group)
            vertical_gap = max(0.0, max(gy1, y1) - min(gy2, y2))
            horizontal_gap = max(0.0, max(gx1, x1) - min(gx2, x2))
            if vertical_gap <= y_tol and horizontal_gap <= x_gap_tol:
                matched.append(i)
        if not matched:
            groups.append([span])
            continue
        base = groups[matched[0]]
        base.append(span)
        for i in reversed(matched[1:]):
            base.extend(groups.pop(i))

    merged: list[TextSpan] = []
    for group in groups:
        if len(group) < 2:
            continue
        text = " ".join(
            s.text.strip()
            for s in sorted(group, key=lambda s: (_span_center(s)[1], s.bbox[0]))
            if s.text.strip()
        )
        if not text:
            continue
        signal_count = sum(
            ch.isdigit() or ch in _MATH_CHARS or ("\u0370" <= ch <= "\u03ff")
            for ch in text
        )
        has_greek_or_math = any(
            ch in _MATH_CHARS or ("\u0370" <= ch <= "\u03ff") for ch in text
        )
        has_digit_and_latin = any(ch.isdigit() for ch in text) and any(
            "A" <= ch <= "Z" or "a" <= ch <= "z" for ch in text
        )
        if signal_count < 2 or not (
            has_greek_or_math
            or has_digit_and_latin
            or (
                len(group) >= 2
                and any("\u0370" <= ch <= "\u03ff" for ch in text)
                and any("A" <= ch <= "Z" or "a" <= ch <= "z" for ch in text)
            )
        ):
            continue
        bbox = (
            min(s.bbox[0] for s in group),
            min(s.bbox[1] for s in group),
            max(s.bbox[2] for s in group),
            max(s.bbox[3] for s in group),
        )
        merged.append(TextSpan(text=text, bbox=bbox, font_size=None))
    return merged


def detect_formula_regions_from_spans(
    spans: list[TextSpan],
    *,
    dpi: int,
    page_w: float,
    page_h: float,
) -> list[DetectedRegion]:
    """Merge contiguous math-like text spans into formula boxes."""
    scored: list[tuple[TextSpan, float]] = []
    merged_spans = merge_formula_spans(spans)
    formula_spans = list(spans) + merged_spans
    merged_ids = {id(sp) for sp in merged_spans}
    for sp in formula_spans:
        sc = _line_math_score(sp.text)
        if id(sp) in merged_ids and sc < 0.55:
            # Fragmented OCR may contain only variables, digits and Greek
            # glyphs, with the "=" lost as a separate box.  The cluster
            # checks above are the evidence in that case.
            signal_ratio = sum(
                ch.isdigit() or ch in _MATH_CHARS or ("\u0370" <= ch <= "\u03ff")
                for ch in sp.text
            ) / max(1, sum(ch.isalpha() or ch.isdigit() for ch in sp.text))
            sc = min(0.82, 0.56 + 0.25 * min(1.0, signal_ratio))
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
        if min(ys0) < 60.0 or max(ys1) > page_h - 60.0:
            continue
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


def _span_center(span: TextSpan) -> tuple[float, float]:
    x1, y1, x2, y2 = span.bbox
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2)


def _cluster_span_rows(
    spans: list[TextSpan], *, y_tol: float = 10.0
) -> list[list[TextSpan]]:
    items = sorted(spans, key=lambda s: (_span_center(s)[1], s.bbox[0]))
    rows: list[list[TextSpan]] = []
    for sp in items:
        cy = _span_center(sp)[1]
        if not rows or abs(_span_center(rows[-1][0])[1] - cy) > y_tol:
            rows.append([sp])
        else:
            rows[-1].append(sp)
    return rows


def _is_toc_like_text(texts: list[str]) -> bool:
    if not texts:
        return False
    leader_hits = sum(1 for t in texts if _TOC_LEADER.search(t))
    # Page-number hits only count on lines that also look like TOC entries
    # (leaders or long title + trailing digits), not bare section numbers.
    page_hits = sum(
        1
        for t in texts
        if _TOC_END_PAGE.search(t.strip())
        and ( _TOC_LEADER.search(t) or len(t.strip()) >= 24 )
    )
    title_hit = any(
        bool(_TOC_TITLE.search(t)) and len(t.strip()) <= 48 for t in texts[:12]
    )
    if title_hit and leader_hits >= 2:
        return True
    if leader_hits >= max(2, len(texts) // 3):
        return True
    if page_hits >= max(3, int(0.45 * len(texts))) and leader_hits >= 1:
        return True
    return False


def _is_two_column_prose(
    rows: list[list[TextSpan]], *, page_w: float
) -> bool:
    """Two wide text columns are not a data table."""
    if page_w <= 0 or len(rows) < 4:
        return False
    centers: list[float] = []
    widths: list[float] = []
    lens: list[int] = []
    for row in rows:
        for sp in row:
            t = sp.text.strip()
            if not t:
                continue
            cx, _ = _span_center(sp)
            centers.append(cx)
            widths.append(max(0.0, sp.bbox[2] - sp.bbox[0]))
            lens.append(len(t))
    if len(centers) < 12:
        return False
    hist, _ = np.histogram(centers, bins=12)
    peak_ids = [i for i, v in enumerate(hist) if v >= max(3, hist.max() * 0.45)]
    if len(peak_ids) != 2:
        return False
    if abs(peak_ids[1] - peak_ids[0]) < 3:
        return False
    med_w = float(np.median(widths))
    med_len = float(np.median(lens))
    # Prose columns are wide and wordy; true tables have short/narrow cells.
    return med_w > 0.22 * page_w and med_len >= 28


def _column_x_peaks(xs: list[float], *, min_peaks: int = 3) -> list[float]:
    if len(xs) < min_peaks:
        return []
    arr = np.asarray(xs, dtype=float)
    bins = max(8, min(18, int(len(arr) / 2)))
    hist, edges = np.histogram(arr, bins=bins)
    thr = max(2.0, float(hist.max()) * 0.35)
    peaks = []
    for i, v in enumerate(hist):
        if v < thr:
            continue
        peaks.append(0.5 * (edges[i] + edges[i + 1]))
    # merge close peaks
    merged: list[float] = []
    for p in peaks:
        if not merged or abs(p - merged[-1]) > 18.0:
            merged.append(p)
        else:
            merged[-1] = 0.5 * (merged[-1] + p)
    return merged if len(merged) >= min_peaks else []


def detect_table_regions_cv(
    image_bgr: np.ndarray,
    *,
    dpi: int,
    page_w: float,
    page_h: float,
) -> list[DetectedRegion]:
    """Find ruled tables via intersecting horizontal/vertical line grids."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10
    )
    hk = max(w // 35, 22)
    vk = max(h // 35, 22)
    hor = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))
    )
    ver = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk))
    )
    grid = cv2.bitwise_and(hor, ver)
    # Dilate intersections so a ruled table becomes one connected component.
    grid = cv2.dilate(
        grid, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=2
    )
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[DetectedRegion] = []
    page_area = float(w * h)
    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        area = ww * hh
        if area < 0.015 * page_area or area > 0.55 * page_area:
            continue
        if ww < w * 0.20 or hh < h * 0.04:
            continue
        roi_h = hor[y : y + hh, x : x + ww]
        roi_v = ver[y : y + hh, x : x + ww]
        if roi_h.mean() < 1.2 or roi_v.mean() < 1.2:
            continue

        def _line_peaks(energy: np.ndarray, axis: int, minimum: float) -> int:
            vals = energy.sum(axis=axis) / 255.0
            ids = np.flatnonzero(vals >= minimum).tolist()
            groups: list[list[int]] = []
            for idx in ids:
                if not groups or idx - groups[-1][-1] > 6:
                    groups.append([idx])
                else:
                    groups[-1].append(idx)
            return len(groups)

        n_h = _line_peaks(roi_h, 1, max(3.0, ww * 0.12))
        n_v = _line_peaks(roi_v, 0, max(3.0, hh * 0.12))
        # Need a real grid, not a single column gutter or figure frame.
        if n_h < 3 or n_v < 3:
            continue
        score = min(
            1.0,
            0.35
            + 0.08 * min(n_h, 8)
            + 0.08 * min(n_v, 8)
            + (roi_h.mean() + roi_v.mean()) / 50.0,
        )
        bbox_pt = _px_to_pt(x, y, x + ww, y + hh, dpi)
        out.append(
            DetectedRegion(
                type=BlockType.TABLE,
                bbox_pt=bbox_pt,
                bbox_px=(x, y, x + ww, y + hh),
                score=score,
                method="cv_line_grid",
                notes=[f"grid_lines=h{n_h}_v{n_v}"],
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
    """Detect compact cell grids from OCR/text spans (incl. unruled tables).

    Explicitly rejects:
    - table-of-contents / dotted-leader pages
    - two-column body text
    - page-wide clouds of short OCR fragments
    """
    if len(spans) < 12 or page_w <= 0 or page_h <= 0:
        return []

    # Drop repeating scan header/footer.
    body = [
        s
        for s in spans
        if 40.0 < _span_center(s)[1] < page_h - 40.0 and s.text.strip()
    ]
    if len(body) < 12:
        return []

    all_text = [s.text for s in body]
    if _is_toc_like_text(all_text):
        return []

    rows = _cluster_span_rows(body, y_tol=max(8.0, page_h * 0.012))
    if _is_two_column_prose(rows, page_w=page_w):
        return []

    # Keep only rows that look like table rows: >=3 short/narrow cells.
    cell_rows: list[list[TextSpan]] = []
    for row in rows:
        cells = []
        for sp in sorted(row, key=lambda s: s.bbox[0]):
            t = sp.text.strip()
            width = sp.bbox[2] - sp.bbox[0]
            if not t:
                continue
            short = len(t) <= 18 or width <= 0.18 * page_w
            numeric = _is_numericish(t)
            if short or numeric:
                cells.append(sp)
        if len(cells) >= 3:
            # reject TOC-style rows inside an otherwise normal page
            if _is_toc_like_text([c.text for c in cells]):
                continue
            # reject a row that is really one prose sentence split poorly
            avg_len = sum(len(c.text.strip()) for c in cells) / len(cells)
            if avg_len > 22 and len(cells) <= 3:
                continue
            cell_rows.append(cells)

    if len(cell_rows) < 3:
        return []

    # Group consecutive cell-rows into candidate table bands.
    bands: list[list[list[TextSpan]]] = []
    current: list[list[TextSpan]] = [cell_rows[0]]
    for prev, row in zip(cell_rows, cell_rows[1:]):
        gap = _span_center(row[0])[1] - _span_center(prev[0])[1]
        if gap > max(42.0, page_h * 0.06):
            if len(current) >= 3:
                bands.append(current)
            current = [row]
        else:
            current.append(row)
    if len(current) >= 3:
        bands.append(current)

    out: list[DetectedRegion] = []
    for band in bands:
        xs: list[float] = []
        ys0: list[float] = []
        ys1: list[float] = []
        xs0: list[float] = []
        xs1: list[float] = []
        numeric_cells = 0
        total_cells = 0
        for row in band:
            for sp in row:
                cx, _ = _span_center(sp)
                xs.append(cx)
                xs0.append(sp.bbox[0])
                xs1.append(sp.bbox[2])
                ys0.append(sp.bbox[1])
                ys1.append(sp.bbox[3])
                total_cells += 1
                if _is_numericish(sp.text):
                    numeric_cells += 1
        peaks = _column_x_peaks(xs, min_peaks=3)
        if len(peaks) < 3:
            continue
        # Stable columns: each peak should appear in most rows.
        aligned_rows = 0
        for row in band:
            row_xs = [_span_center(sp)[0] for sp in row]
            hits = sum(any(abs(rx - p) <= 22.0 for rx in row_xs) for p in peaks)
            if hits >= min(3, len(peaks)):
                aligned_rows += 1
        if aligned_rows < max(3, int(0.6 * len(band))):
            continue

        bbox = BBox(
            x1=max(0.0, min(xs0) - 6),
            y1=max(0.0, min(ys0) - 6),
            x2=min(page_w, max(xs1) + 6),
            y2=min(page_h, max(ys1) + 6),
        )
        area = (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1)
        # Local table bands should stay compact; full-page text is not a table.
        if area > 0.40 * page_w * page_h:
            continue
        if (bbox.y2 - bbox.y1) < 0.035 * page_h:
            continue
        if (bbox.x2 - bbox.x1) < 0.25 * page_w:
            continue

        numeric_ratio = numeric_cells / max(1, total_cells)
        nonempty_ratio = total_cells and len(
            [sp for row in band for sp in row if sp.text.strip()]
        ) / total_cells
        # A real unruled table has stable repeated columns and a reasonably
        # filled grid. Formula fragments and prose columns do not.
        if nonempty_ratio < 0.50:
            continue
        if len(peaks) < 3 or len(band) < 3:
            continue
        # Span-based tables without a numeric backbone are usually prose
        # fragments or formula paragraphs. Text-only typo tables are handled
        # by the ruled-grid detector, not this heuristic.
        if numeric_ratio < 0.30:
            continue
        score = min(
            0.95,
            0.40
            + 0.08 * min(len(peaks), 6)
            + 0.05 * min(len(band), 8)
            + 0.20 * numeric_ratio,
        )
        out.append(
            DetectedRegion(
                type=BlockType.TABLE,
                bbox_pt=bbox,
                bbox_px=_pt_to_px(bbox, dpi),
                score=score,
                method="span_cell_grid",
                notes=[
                    f"rows={len(band)}",
                    f"cols={len(peaks)}",
                    f"numeric_ratio={numeric_ratio:.2f}",
                    f"nonempty_ratio={nonempty_ratio:.2f}",
                ],
            )
        )
    return out


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
        notes: list[str] = []
        roi = gray[y : y + hh, x : x + ww]
        if roi.size:
            edges = cv2.Canny(roi, 50, 150)
            horizontal = cv2.morphologyEx(
                edges, cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_RECT, (max(12, ww // 8), 1)),
            )
            vertical = cv2.morphologyEx(
                edges, cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(12, hh // 8))),
            )
            if horizontal.mean() > 1.0 and vertical.mean() > 1.0:
                notes.append("chart_candidate")
        out.append(
            DetectedRegion(
                type=BlockType.FIGURE,
                bbox_pt=bbox_pt,
                bbox_px=(x, y, x + ww, y + hh),
                score=min(1.0, area / (0.15 * page_area)),
                method="nontext_ink_cc",
                notes=notes,
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


def _is_formula_figure_conflict(
    figure: DetectedRegion,
    formula: DetectedRegion,
    *,
    page_area: float,
) -> bool:
    """Reject a small ink blob when it is actually a formula crop."""
    f_area = max(
        0.0,
        (figure.bbox_pt.x2 - figure.bbox_pt.x1)
        * (figure.bbox_pt.y2 - figure.bbox_pt.y1),
    )
    contained = (
        formula.bbox_pt.x1 >= figure.bbox_pt.x1
        and formula.bbox_pt.y1 >= figure.bbox_pt.y1
        and formula.bbox_pt.x2 <= figure.bbox_pt.x2
        and formula.bbox_pt.y2 <= figure.bbox_pt.y2
    )
    return (
        formula.method == "span_math_heuristic"
        and formula.score >= 0.70
        and contained
        and f_area < 0.15 * page_area
    )


def merge_regions(regions: list[DetectedRegion], iou_thr: float = 0.4) -> list[DetectedRegion]:
    """NMS with object priority: figure/table > formula > residual text."""
    kept: list[DetectedRegion] = []
    for r in sorted(regions, key=lambda x: (-REGION_PRIORITY.get(x.type, 0), -x.score)):
        drop = False
        for k in kept:
            iou = _iou_pt(k.bbox_pt, r.bbox_pt)
            if k.type == r.type and iou >= iou_thr:
                drop = True
                break
            if (
                k.type != r.type
                and REGION_PRIORITY.get(k.type, 0) > REGION_PRIORITY.get(r.type, 0)
                and iou >= 0.20
            ):
                drop = True
                break
            # A formula fully inside a table is table content, even when the
            # table is much larger and IoU is therefore below 0.20.
            if (
                k.type == BlockType.TABLE
                and r.type == BlockType.FORMULA
                and r.bbox_pt.x1 >= k.bbox_pt.x1
                and r.bbox_pt.y1 >= k.bbox_pt.y1
                and r.bbox_pt.x2 <= k.bbox_pt.x2
                and r.bbox_pt.y2 <= k.bbox_pt.y2
            ):
                drop = True
                break
        if not drop:
            kept.append(r)
    return sorted(kept, key=lambda x: (x.bbox_pt.y1, x.bbox_pt.x1))


def detect_page_regions(
    doc: fitz.Document,
    page_number: int,
    *,
    doc_id: str,
    dpi: int = 200,
    use_ocr_fallback: bool = True,
) -> tuple[np.ndarray, list[DetectedRegion], list[RegionBlock], list[TextSpan]]:
    """Detect formula/table/figure regions; return render, detections, IR blocks, spans."""
    rendered = render_page(doc, page_number, dpi=dpi)
    page = doc[page_number - 1]
    page_w, page_h = float(page.rect.width), float(page.rect.height)
    _, spans = _extract_spans(page)

    # Scanned pages often have empty text layer — build pseudo-spans via OCR
    if use_ocr_fallback and len(spans) < 5:
        spans = _ocr_pseudo_spans(rendered.image_bgr, dpi=dpi, doc_id=doc_id, page=page_number)

    formula_regions = detect_formula_regions_from_spans(
        spans, dpi=dpi, page_w=page_w, page_h=page_h
    )
    # A page may have a usable text layer while the image-only formula band is
    # absent from it.  OCR just the upper part in that case; this keeps the
    # recovery targeted and avoids replacing the page's normal text spans.
    upper_formula = [
        r for r in formula_regions if r.bbox_pt.y1 < 0.45 * page_h
    ]
    if not upper_formula and use_ocr_fallback and len(spans) >= 5:
        ocr_top = _ocr_pseudo_spans(
            rendered.image_bgr,
            dpi=dpi,
            doc_id=doc_id,
            page=page_number,
            y_range=(0.0, 0.45),
        )
        if ocr_top:
            formula_regions = detect_formula_regions_from_spans(
                spans + ocr_top, dpi=dpi, page_w=page_w, page_h=page_h
            )

    regions: list[DetectedRegion] = []
    regions.extend(formula_regions)
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
    page_area_pt = page_w * page_h
    formulas = [r for r in regions if r.type == BlockType.FORMULA]
    regions = [
        r
        for r in regions
        if r.type != BlockType.FIGURE
        or not any(
            _is_formula_figure_conflict(r, f, page_area=page_area_pt)
            for f in formulas
        )
    ]
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
    return rendered.image_bgr, regions, blocks, spans


def _ocr_pseudo_spans(
    image_bgr: np.ndarray,
    *,
    dpi: int,
    doc_id: str,
    page: int,
    y_range: tuple[float, float] | None = None,
) -> list[TextSpan]:
    try:
        from ingestion.ocr_rapid import RapidOCRConfig, recognize_page_image
        from core.config import get_settings

        settings = get_settings()
        source_h = image_bgr.shape[0]
        y0_px = 0
        y1_px = source_h
        if y_range is not None:
            y0_px = max(0, int(source_h * y_range[0]))
            y1_px = min(source_h, int(source_h * y_range[1]))
            if y1_px <= y0_px:
                return []
        crop = image_bgr[y0_px:y1_px]
        ocr = recognize_page_image(
            crop,
            dpi=dpi,
            page=page,
            doc_id=doc_id,
            cfg=RapidOCRConfig(
                use_gpu=settings.enable_gpu_ocr,
                max_side_len=settings.ocr_max_side_len,
                model_dir=str(settings.ocr_model_dir),
                enable_latin=settings.ocr_enable_latin,
                enable_greek=settings.ocr_enable_greek,
            ),
        )
        spans: list[TextSpan] = []
        for ln in ocr.lines:
            offset_pt = y0_px * 72.0 / float(dpi)
            spans.append(
                TextSpan(
                    text=ln.text,
                    bbox=(
                        ln.bbox_pt[0],
                        ln.bbox_pt[1] + offset_pt,
                        ln.bbox_pt[2],
                        ln.bbox_pt[3] + offset_pt,
                    ),
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
