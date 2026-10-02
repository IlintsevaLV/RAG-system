"""Cheap layout hints: columns / table-ish density from text spans."""

from __future__ import annotations

import numpy as np

from ingestion.models import LayoutHints, TextSpan


_LINE_Y = 4.0


def reorder_spans_by_columns(
    spans: list[TextSpan],
    page_width: float,
    *,
    min_col_gap_ratio: float = 0.08,
) -> tuple[str, LayoutHints]:
    """Reading order: one visual line stays one line.

    A full-width paragraph has words on both sides of the page middle. Splitting
    those words into a left pile and a right pile tears every line in half.
    Two columns are used only when the same line has a wide gap across the middle.
    """
    hints = LayoutHints()
    if not spans or page_width <= 0:
        return "", hints

    lines = _group_lines(spans)
    gutter_lines = [line for line in lines if _gutter_at(line, page_width, min_col_gap_ratio) is not None]
    two_col = len(gutter_lines) >= 2 and len(gutter_lines) / len(lines) >= 0.35
    hints.n_columns = 2 if two_col else 1
    if not two_col:
        text = "\n".join(t for t in (_line_text(line) for line in lines) if t)
        return text, hints

    parts: list[str] = []
    left_buf: list[str] = []
    right_buf: list[str] = []

    def flush() -> None:
        nonlocal left_buf, right_buf
        left = "\n".join(x for x in left_buf if x)
        right = "\n".join(x for x in right_buf if x)
        if left:
            parts.append(left)
        if right:
            parts.append(right)
        left_buf, right_buf = [], []

    for line in lines:
        split_at = _gutter_at(line, page_width, min_col_gap_ratio)
        if split_at is None:
            flush()
            text = _line_text(line)
            if text:
                parts.append(text)
            continue
        left_buf.append(_line_text(line[: split_at + 1]))
        right_buf.append(_line_text(line[split_at + 1 :]))
    flush()
    hints.notes.append(f"two_column_split lines={len(gutter_lines)}")
    return "\n\n".join(parts), hints


def _group_lines(spans: list[TextSpan]) -> list[list[TextSpan]]:
    """Words whose tops sit within 4pt belong to one visual line, then left to right."""
    ordered = sorted(spans, key=lambda s: (s.bbox[1], s.bbox[0]))
    lines: list[list[TextSpan]] = []
    current: list[TextSpan] = []
    anchor: float | None = None
    for span in ordered:
        y = span.bbox[1]
        if anchor is None or abs(y - anchor) <= _LINE_Y:
            if anchor is None:
                anchor = y
            current.append(span)
            continue
        lines.append(sorted(current, key=lambda s: s.bbox[0]))
        current = [span]
        anchor = y
    if current:
        lines.append(sorted(current, key=lambda s: s.bbox[0]))
    return lines


def _gutter_at(line: list[TextSpan], page_width: float, gap_ratio: float) -> int | None:
    """Index of the left span where a gap crosses the page middle. None if the line is one column."""
    if len(line) < 2 or page_width <= 0:
        return None
    need = page_width * gap_ratio
    mid = page_width / 2.0
    best_i: int | None = None
    best_gap = 0.0
    for i in range(len(line) - 1):
        gap = line[i + 1].bbox[0] - line[i].bbox[2]
        straddles = line[i].bbox[2] <= mid + need * 0.25 and line[i + 1].bbox[0] >= mid - need * 0.25
        if gap >= need and straddles and gap > best_gap:
            best_gap = gap
            best_i = i
    return best_i


def _line_text(spans: list[TextSpan]) -> str:
    return " ".join(s.text.strip() for s in spans if s.text and s.text.strip())


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
