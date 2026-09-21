"""Reading-order reconstruction for OCR word/line boxes.

RapidOCR often returns word-level quads. Naively joining them with newlines
(sorted only by y,x) produces the “backwards / lost lines” effect seen on
scanned books like _1954: wrap-around figures, spaced headings, formula shards.
"""

from __future__ import annotations

from dataclasses import dataclass

from ingestion.models import OCRLine


@dataclass
class ReadingOrderStats:
    n_columns: int = 1
    n_text_lines: int = 0
    short_fragment_ratio: float = 0.0
    formula_suspect: bool = False
    notes: list[str] | None = None


def _cy(ln: OCRLine) -> float:
    return 0.5 * (ln.bbox_px[1] + ln.bbox_px[3])


def _cx(ln: OCRLine) -> float:
    return 0.5 * (ln.bbox_px[0] + ln.bbox_px[2])


def _h(ln: OCRLine) -> float:
    return max(1.0, ln.bbox_px[3] - ln.bbox_px[1])


def detect_two_columns(
    lines: list[OCRLine],
    page_width: float,
    *,
    min_lines: int = 24,
) -> bool:
    """Heuristic 2-column split from OCR box x-centers."""
    if len(lines) < min_lines or page_width <= 0:
        return False
    mid = page_width / 2.0
    xs = [_cx(ln) for ln in lines]
    left = sum(1 for x in xs if x < mid)
    right = len(xs) - left
    left_frac = left / len(xs)
    if not (0.28 <= left_frac <= 0.72):
        return False
    # gap: few boxes straddling the mid gutter
    gutter = page_width * 0.06
    in_gutter = sum(1 for x in xs if abs(x - mid) < gutter)
    if in_gutter / len(xs) > 0.12:
        return False
    # both sides need mass
    return left >= 8 and right >= 8


def cluster_into_rows(
    lines: list[OCRLine],
    *,
    y_tol_ratio: float = 0.6,
) -> list[list[OCRLine]]:
    """Group boxes whose vertical centers overlap into reading rows."""
    if not lines:
        return []
    ordered = sorted(lines, key=lambda ln: (_cy(ln), _cx(ln)))
    rows: list[list[OCRLine]] = []
    cur: list[OCRLine] = [ordered[0]]
    cur_y = _cy(ordered[0])
    cur_h = _h(ordered[0])
    for ln in ordered[1:]:
        y = _cy(ln)
        h = _h(ln)
        tol = y_tol_ratio * max(cur_h, h)
        if abs(y - cur_y) <= tol:
            cur.append(ln)
            ys = [_cy(x) for x in cur]
            hs = [_h(x) for x in cur]
            cur_y = sum(ys) / len(ys)
            cur_h = sum(hs) / len(hs)
        else:
            rows.append(sorted(cur, key=_cx))
            cur = [ln]
            cur_y = y
            cur_h = h
    if cur:
        rows.append(sorted(cur, key=_cx))
    return rows


def rows_to_text(rows: list[list[OCRLine]]) -> str:
    out_lines: list[str] = []
    for row in rows:
        parts = [ln.text.strip() for ln in row if ln.text.strip()]
        if not parts:
            continue
        out_lines.append(" ".join(parts))
    return "\n".join(out_lines)


def rebuild_ocr_text(
    lines: list[OCRLine],
    page_width_px: float,
) -> tuple[str, list[OCRLine], ReadingOrderStats]:
    """Return reading-order text + stably sorted lines + stats."""
    stats = ReadingOrderStats(notes=[])
    if not lines:
        return "", [], stats

    short = sum(1 for ln in lines if len(ln.text.strip()) <= 3)
    stats.short_fragment_ratio = short / max(1, len(lines))
    stats.formula_suspect = stats.short_fragment_ratio >= 0.28 and len(lines) >= 20
    if stats.formula_suspect:
        stats.notes.append(
            f"formula_suspect short_ratio={stats.short_fragment_ratio:.2f}"
        )

    two_col = detect_two_columns(lines, page_width_px)
    if two_col:
        mid = page_width_px / 2.0
        left = [ln for ln in lines if _cx(ln) < mid]
        right = [ln for ln in lines if _cx(ln) >= mid]
        left_rows = cluster_into_rows(left)
        right_rows = cluster_into_rows(right)
        text = rows_to_text(left_rows)
        if right_rows:
            text = text + "\n\n" + rows_to_text(right_rows)
        ordered_lines = [ln for row in left_rows for ln in row] + [
            ln for row in right_rows for ln in row
        ]
        stats.n_columns = 2
        stats.n_text_lines = len(left_rows) + len(right_rows)
        stats.notes.append(
            f"ocr_two_column left={len(left)} right={len(right)}"
        )
    else:
        rows = cluster_into_rows(lines)
        text = rows_to_text(rows)
        ordered_lines = [ln for row in rows for ln in row]
        stats.n_columns = 1
        stats.n_text_lines = len(rows)

    return text, ordered_lines, stats
