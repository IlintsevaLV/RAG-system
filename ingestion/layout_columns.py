"""Cheap layout hints: columns / table-ish density from text spans."""

from __future__ import annotations

import numpy as np

from ingestion.models import LayoutHints, TextSpan


def reorder_spans_by_columns(
    spans: list[TextSpan],
    page_width: float,
    *,
    min_col_gap_ratio: float = 0.08,
) -> tuple[str, LayoutHints]:
    """Detect 1–2 columns and return reading-order text (col1 then col2)."""
    hints = LayoutHints()
    if not spans or page_width <= 0:
        return "", hints

    xs = np.array([0.5 * (s.bbox[0] + s.bbox[2]) for s in spans], dtype=float)
    mid = page_width / 2.0
    near_mid = float(np.mean(np.abs(xs - mid) < page_width * min_col_gap_ratio))
    left_frac = float(np.mean(xs < mid))

    two_col = left_frac > 0.25 and left_frac < 0.75 and near_mid < 0.20
    if not two_col:
        # single column: sort by y, then x
        ordered = sorted(spans, key=lambda s: (s.bbox[1], s.bbox[0]))
        text = _join_spans(ordered)
        hints.n_columns = 1
        return text, hints

    left = [s for s in spans if 0.5 * (s.bbox[0] + s.bbox[2]) < mid]
    right = [s for s in spans if 0.5 * (s.bbox[0] + s.bbox[2]) >= mid]
    left.sort(key=lambda s: (s.bbox[1], s.bbox[0]))
    right.sort(key=lambda s: (s.bbox[1], s.bbox[0]))
    text = _join_spans(left) + "\n\n" + _join_spans(right)
    hints.n_columns = 2
    hints.notes.append(f"two_column_split mid={mid:.1f} left={len(left)} right={len(right)}")
    return text, hints


def _join_spans(spans: list[TextSpan]) -> str:
    if not spans:
        return ""
    lines: list[str] = []
    buf: list[str] = []
    last_y: float | None = None
    for s in spans:
        y = s.bbox[1]
        if last_y is not None and abs(y - last_y) > 4.0 and buf:
            lines.append(" ".join(buf))
            buf = []
        buf.append(s.text.strip())
        last_y = y
    if buf:
        lines.append(" ".join(buf))
    return "\n".join(x for x in lines if x)


def table_likelihood(spans: list[TextSpan], page_width: float) -> bool:
    """Heuristic: many short aligned tokens in a grid-like x histogram."""
    if len(spans) < 30 or page_width <= 0:
        return False
    xs = [0.5 * (s.bbox[0] + s.bbox[2]) for s in spans]
    hist, _ = np.histogram(xs, bins=12)
    peaks = int(np.sum(hist >= max(3, hist.max() * 0.45)))
    short = sum(1 for s in spans if len(s.text.strip()) <= 12)
    return peaks >= 3 and short / len(spans) >= 0.45


def enrich_layout_hints(
    spans: list[TextSpan],
    page_width: float,
    base: LayoutHints | None = None,
) -> LayoutHints:
    text, hints = reorder_spans_by_columns(spans, page_width)
    if base:
        hints.vlm_notes = base.vlm_notes
        hints.notes = list(dict.fromkeys([*base.notes, *hints.notes]))
    if table_likelihood(spans, page_width):
        hints.likely_table = True
        hints.notes.append("table_like_span_density")
    # unused text var — caller uses reorder separately
    _ = text
    return hints
