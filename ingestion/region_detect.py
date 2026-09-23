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
# Strict markers for recovering OCR-fragmented formulas.  Digits, brackets
# and plain letters are deliberately absent: "(42)" or "88 89 90" must not
# count as math.
_FORMULA_MARKERS = set("=+−-×÷^_√∫∑∏∂∇∞±≤≥≠≈")
# Greek, extended Greek and letterlike symbols (ℏ).
_MATH_SYMBOL_RE = re.compile(r"[\u0370-\u03FF\u1F00-\u1FFF\u2100-\u214F]")
_LATEX_CMD_RE = re.compile(r"\\[a-zA-Z]+")
_SLASH_FRACTION_RE = re.compile(r"(?<=[\w)])/(?=[\w(])")
# "=" is a relation between two sides; a stacked fraction is a division bar
# that OCR cannot emit.  Both carry more evidence than a single operator; a
# bar confirmed on the image is as strong as an explicit \frac.
_RELATION_WEIGHT = 2
_STACKED_FRACTION_WEIGHT = 2
_BAR_FRACTION_WEIGHT = 3
_VARIABLE_RE = re.compile(r"[A-Za-z\u0370-\u03FF\u1F00-\u1FFF\u2100-\u214F]")
_WORD_RUN_RE = re.compile(r"[^\W\d_]{4,}")
# Latin/Greek glyphs the OCR heads substitute for Cyrillic letters.
_CYR_LOOKALIKES = {
    "a": "а", "A": "А", "B": "В", "b": "в", "c": "с", "C": "С", "e": "е",
    "E": "Е", "H": "Н", "k": "к", "K": "К", "m": "м", "M": "М", "o": "о",
    "O": "О", "p": "р", "P": "Р", "T": "Т", "t": "т", "x": "х", "X": "Х",
    "y": "у", "α": "а", "Α": "А", "Β": "В", "ε": "е", "Ε": "Е", "Η": "Н",
    "ι": "и", "κ": "к", "Κ": "К", "μ": "м", "Μ": "М", "ο": "о", "Ο": "О",
    "Π": "П", "π": "п", "ρ": "р", "Ρ": "Р", "τ": "т", "Τ": "Т", "χ": "х",
    "Χ": "Х", "γ": "у",
}
_FORMULA_STOPWORDS_RE = re.compile(
    r"\b(и|в|на|с|для|при|от|до|или|но|как|что|это|же|бы|не|по|из|к|о|об|"
    r"за|над|под|про|через|между|где|если|тогда|and|the|for|with|from)\b",
    re.IGNORECASE,
)
_CYR_WORD_RE = re.compile(r"[А-Яа-яЁё]{2,}")
_LONG_WORD_RE = re.compile(r"[А-Яа-яЁёA-Za-z]{11,}")
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


def count_formula_markers(text: str) -> int:
    """Strict math-marker count used to validate merged OCR fragments."""
    count = 0
    for ch in text:
        if ch == "=":
            count += _RELATION_WEIGHT
        elif ch in _FORMULA_MARKERS:
            count += 1
    count += len(_MATH_SYMBOL_RE.findall(text))
    count += len(_LATEX_CMD_RE.findall(text))
    count += len(_SLASH_FRACTION_RE.findall(text))
    return count


def _has_operator_evidence(text: str) -> bool:
    return bool(
        any(ch in _FORMULA_MARKERS for ch in text)
        or _LATEX_CMD_RE.search(text)
        or _SLASH_FRACTION_RE.search(text)
    )


def is_real_formula(text: str, *, layout_markers: int = 0) -> bool:
    """Return whether merged fragment text is a formula rather than prose/OCR noise.

    ``layout_markers`` is structural evidence from geometry and pixels
    (stacked fractions, fraction bars) that OCR cannot put into the text.
    """
    text = text.strip()
    if len(text) < 4:
        return False
    if not _VARIABLE_RE.search(text):
        # Axis ticks and page numbers: "0 -8 -12", "88 89 90".
        return False

    markers = count_formula_markers(text) + layout_markers
    if markers < 3:
        return False
    # Greek letters alone are not enough: the Greek OCR head turns Cyrillic
    # prose into Greek lookalikes.  Require an operator or a fraction.
    if not _has_operator_evidence(text) and layout_markers == 0:
        return False

    non_space = [ch for ch in text if not ch.isspace()]
    if not non_space or markers / len(non_space) < 0.30:
        return False

    # Single-letter stopwords ("с", "в") are also OCR'd variables (c_N);
    # they only indicate prose next to another Cyrillic word.
    for match in _FORMULA_STOPWORDS_RE.finditer(text):
        if len(match.group(0)) > 1 or _CYR_WORD_RE.search(text):
            return False

    if _LONG_WORD_RE.search(text):
        return False
    # Formula tokens are short variables; four or more consecutive letters in
    # any script is a word misread by OCR ("Myφτa", "Bepτoπera").
    if _WORD_RUN_RE.search(text):
        return False
    if any(len(_MATH_SYMBOL_RE.findall(t)) >= 3 for t in text.split()):
        return False

    # OCR noise: "b H i c l U M t e", "\mathrm { c h e C r 2 ~ E }".
    tokens = text.split()
    single_letters = sum(
        1 for t in tokens if len(t.strip("\\{}")) == 1 and t.strip("\\{}").isalpha()
    )
    if len(tokens) >= 4 and single_letters / len(tokens) > 0.5:
        return False
    return True


def _is_lookalike_stopword(token: str) -> bool:
    """Latin/Greek OCR of a Cyrillic stopword: "kak" -> "как", "Πpι" -> "при"."""
    if len(token) < 2 or not token.isalpha():
        return False
    if any(ch not in _CYR_LOOKALIKES for ch in token):
        return False
    cyr = "".join(_CYR_LOOKALIKES[ch] for ch in token)
    return bool(_FORMULA_STOPWORDS_RE.fullmatch(cyr))


def _is_lhs_fragment(text: str) -> bool:
    """Left-hand side cut off by OCR: "χ =", "M=" (the rest is a fraction)."""
    return bool(re.fullmatch(r"[^\s=]{1,4}\s*=", text.strip()))


def _is_formula_fragment(span: TextSpan) -> bool:
    """Return whether a short OCR span can be a piece of a broken formula.

    Numerators, denominators, Greek tokens and compact variables qualify;
    words (Cyrillic words, 4+ letter runs, lookalike stopwords) do not, so
    prose lines cannot chain fragments together.
    """
    text = span.text.strip()
    if not text or len(text) > 24:
        return False
    if _CYR_WORD_RE.search(text) or _WORD_RUN_RE.search(text):
        return False
    if any(_is_lookalike_stopword(t.strip(".,;:()")) for t in text.split()):
        return False
    if _line_math_score(text) >= 0.55 and not _is_lhs_fragment(text):
        # Complete one-span formulas are handled by the single-span heuristic.
        return False
    return bool(
        _MATH_SYMBOL_RE.search(text)
        or any(ch.isdigit() or ch in _FORMULA_MARKERS for ch in text)
        or re.search(r"[A-Za-z]", text)
    )


def _y_overlap(a: TextSpan, b: TextSpan) -> bool:
    return min(a.bbox[3], b.bbox[3]) > max(a.bbox[1], b.bbox[1])


def _can_merge_fragments(
    a: TextSpan, b: TextSpan, *, page_w: float, page_h: float
) -> bool:
    ax1, ay1, ax2, ay2 = a.bbox
    bx1, by1, bx2, by2 = b.bbox
    y_gap = max(0.0, max(ay1, by1) - min(ay2, by2))
    if y_gap > 0.04 * page_h:
        return False
    x_gap = max(0.0, max(ax1, bx1) - min(ax2, bx2))
    if x_gap > 0.15 * page_w:
        return False
    if (ay2 < by1 or by2 < ay1) and y_gap > 0.02 * page_h:
        return False
    return True


def _is_stacked_pair(top: TextSpan, bot: TextSpan) -> float | None:
    """Return the vertical gap if ``top`` sits directly above ``bot``."""
    tx1, ty1, tx2, ty2 = top.bbox
    bx1, by1, bx2, by2 = bot.bbox
    th, bh = max(1.0, ty2 - ty1), max(1.0, by2 - by1)
    if (ty1 + ty2) / 2 >= (by1 + by2) / 2:
        return None
    gap = by1 - ty2
    if gap < -0.4 * min(th, bh) or gap > 0.8 * max(th, bh):
        return None
    overlap = min(tx2, bx2) - max(tx1, bx1)
    if overlap < 0.5 * min(tx2 - tx1, bx2 - bx1):
        return None
    return gap


def _has_fraction_bar(
    image_bgr: np.ndarray, top: TextSpan, bot: TextSpan, *, dpi: int
) -> bool:
    """Look for a short horizontal ink line between numerator and denominator.

    The line must cover most of the narrower part and must not run far past
    both parts: a long line is a table rule or underline, not a fraction bar.
    """
    s = dpi / 72.0
    img_h, img_w = image_bgr.shape[:2]
    ux1 = min(top.bbox[0], bot.bbox[0])
    ux2 = max(top.bbox[2], bot.bbox[2])
    union_w = max(1.0, ux2 - ux1)
    narrow_w = max(1.0, min(top.bbox[2] - top.bbox[0], bot.bbox[2] - bot.bbox[0]))
    th = top.bbox[3] - top.bbox[1]
    bh = bot.bbox[3] - bot.bbox[1]
    y1 = min(top.bbox[3] - 0.25 * th, bot.bbox[1])
    y2 = max(bot.bbox[1] + 0.25 * bh, top.bbox[3])
    px1 = int(max(0.0, (ux1 - union_w) * s))
    px2 = int(min(float(img_w), (ux2 + union_w) * s))
    py1 = int(max(0.0, y1 * s))
    py2 = int(min(float(img_h), y2 * s))
    if px2 - px1 < 4 or py2 - py1 < 1:
        return False
    roi = image_bgr[py1:py2, px1:px2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    ink = gray < min(150.0, float(np.median(gray)) * 0.65)
    core_x1 = int(ux1 * s) - px1
    core_x2 = int(ux2 * s) - px1
    slack = int(0.5 * union_w * s)
    min_run = max(6, int(0.7 * narrow_w * s))
    for row in ink:
        padded = np.concatenate(([False], row, [False]))
        edges = np.flatnonzero(padded[1:] != padded[:-1])
        runs: list[list[int]] = []
        for start, stop in zip(edges[::2], edges[1::2]):
            # Scan noise breaks lines into pieces; rejoin tiny gaps so a
            # grid line is measured at its real length.
            if runs and start - runs[-1][1] <= 3:
                runs[-1][1] = stop
            else:
                runs.append([start, stop])
        for start, stop in runs:
            if stop - start < min_run or stop <= core_x1 or start >= core_x2:
                continue
            if start >= core_x1 - slack and stop <= core_x2 + slack:
                return True
    return False


def _fraction_edges(
    fragments: list[TextSpan],
    *,
    image_bgr: np.ndarray | None,
    dpi: int,
) -> dict[tuple[int, int], bool]:
    """Numerator->denominator pairs; value tells whether a bar was seen.

    Pairs are accepted greedily by gap, and a span may not be both a
    denominator and a numerator: that chain is two formulas (or a table
    column) stacked on top of each other, not one fraction.
    """
    cands: list[tuple[float, int, int]] = []
    for i, top in enumerate(fragments):
        for j, bot in enumerate(fragments):
            if i != j:
                gap = _is_stacked_pair(top, bot)
                if gap is not None:
                    cands.append((gap, i, j))
    tops: set[int] = set()
    bottoms: set[int] = set()
    edges: dict[tuple[int, int], bool] = {}
    for _gap, i, j in sorted(cands):
        if i in tops or j in bottoms or i in bottoms or j in tops:
            continue
        bar = (
            image_bgr is not None
            and _has_fraction_bar(image_bgr, fragments[i], fragments[j], dpi=dpi)
        )
        if image_bgr is not None and not bar:
            continue
        tops.add(i)
        bottoms.add(j)
        edges[(i, j)] = bar
    return edges


def _build_merge_groups(
    fragments: list[TextSpan],
    edges: dict[tuple[int, int], bool],
    *,
    page_w: float,
    page_h: float,
) -> list[list[int]]:
    """Group fragments on the same line, or linked through a fraction."""
    parent = list(range(len(fragments)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(fragments)):
        for j in range(i + 1, len(fragments)):
            a, b = fragments[i], fragments[j]
            if not _can_merge_fragments(a, b, page_w=page_w, page_h=page_h):
                continue
            if _y_overlap(a, b) or (i, j) in edges or (j, i) in edges:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(fragments)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _layout_markers(
    group: list[int],
    fragments: list[TextSpan],
    edges: dict[tuple[int, int], bool],
) -> int:
    """Marker weight contributed by fractions inside a group.

    A pixel-confirmed bar over/under a variable counts like a ``\\frac``.
    Without an image, stacking counts only between two variable parts.
    A digit fraction (1/2) counts only next to such a fraction.
    """
    members = set(group)
    strong = 0
    digit = 0
    for (i, j), bar in edges.items():
        if i not in members or j not in members:
            continue
        top_var = bool(_VARIABLE_RE.search(fragments[i].text))
        bot_var = bool(_VARIABLE_RE.search(fragments[j].text))
        if bar and (top_var or bot_var):
            strong += _BAR_FRACTION_WEIGHT
        elif top_var and bot_var:
            strong += _STACKED_FRACTION_WEIGHT
        else:
            digit += _STACKED_FRACTION_WEIGHT
    return strong + (digit if strong else 0)


def _is_text_heavy_page(spans: list[TextSpan]) -> bool:
    total_text = "".join(s.text for s in spans)
    # Plain hyphens are mostly word hyphenation in prose.
    total_math = sum(1 for ch in total_text if ch in _FORMULA_MARKERS and ch != "-")
    total_math += len(_MATH_SYMBOL_RE.findall(total_text))
    return len(total_text) > 2000 and total_math < 5


def _body_spans(spans: list[TextSpan], *, page_h: float) -> list[TextSpan]:
    return [
        s
        for s in spans
        if s.text.strip() and 40.0 < _span_center(s)[1] < page_h - 40.0
    ]


def _formula_merge_blocked(
    spans: list[TextSpan], *, page_w: float, page_h: float
) -> bool:
    """Pages where fragment recovery must not run at all."""
    body = _body_spans(spans, page_h=page_h)
    texts = [s.text for s in body]
    return (
        _is_toc_like_text(texts)
        or _is_bibliography_like_text(texts)
        or _is_two_column_prose(_cluster_span_rows(body), page_w=page_w)
        or _is_text_heavy_page(body)
    )


def merge_formula_spans(
    spans: list[TextSpan],
    *,
    page_w: float,
    page_h: float,
    image_bgr: np.ndarray | None = None,
    dpi: int = 200,
) -> list[TextSpan]:
    """Merge OCR fragments of the same formula into new synthetic spans.

    Returns only the merged spans; the input spans are not modified.  Spans
    that are formulas on their own are left to the single-span heuristic.
    With ``image_bgr`` a stacked pair only links as a fraction when a
    fraction bar is visible between the parts.
    """
    if page_w <= 0 or page_h <= 0:
        return []
    if _formula_merge_blocked(spans, page_w=page_w, page_h=page_h):
        return []
    fragments = [
        s for s in _body_spans(spans, page_h=page_h) if _is_formula_fragment(s)
    ]
    if len(fragments) < 2:
        return []

    edges = _fraction_edges(fragments, image_bgr=image_bgr, dpi=dpi)
    merged: list[TextSpan] = []
    for group in _build_merge_groups(fragments, edges, page_w=page_w, page_h=page_h):
        if len(group) < 2:
            continue
        parts = [fragments[i] for i in group]
        x1 = min(s.bbox[0] for s in parts)
        y1 = min(s.bbox[1] for s in parts)
        x2 = max(s.bbox[2] for s in parts)
        y2 = max(s.bbox[3] for s in parts)
        if x2 - x1 > 0.30 * page_w or y2 - y1 > 0.15 * page_h:
            continue
        text = " ".join(
            s.text.strip()
            for s in sorted(parts, key=lambda s: (_span_center(s)[1], s.bbox[0]))
        )
        layout = _layout_markers(group, fragments, edges)
        if not is_real_formula(text, layout_markers=layout):
            continue
        if _is_toc_like_text([text]):
            continue
        merged.append(TextSpan(text=text, bbox=(x1, y1, x2, y2), font_size=None))
    return merged


def _merged_formula_score(span: TextSpan) -> float:
    """Confidence for a merged span that already passed ``is_real_formula``."""
    non_space = sum(not ch.isspace() for ch in span.text) or 1
    ratio = count_formula_markers(span.text) / non_space
    return min(0.85, 0.60 + 0.25 * min(1.0, ratio))


def _formula_crop_ok(region: DetectedRegion, *, img_w: int, img_h: int) -> bool:
    """Reject crops that are empty or too thin for OpenCV/UniMERNet."""
    if region.bbox_px is None:
        return False
    x1, y1, x2, y2 = region.bbox_px
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w, x2), min(img_h, y2)
    if x2 - x1 < 10 or y2 - y1 < 10:
        return False
    if region.method == "span_fragment_merge" and (
        y2 - y1 > 0.15 * img_h or x2 - x1 > 0.30 * img_w
    ):
        return False
    return True


def _bbox_overlaps(a: BBox, b: BBox, *, min_share: float = 0.2) -> bool:
    """Intersection covers at least ``min_share`` of the smaller box.

    Touching padded boxes of two stacked formulas must not count.
    """
    iw = min(a.x2, b.x2) - max(a.x1, b.x1)
    ih = min(a.y2, b.y2) - max(a.y1, b.y1)
    if iw <= 0 or ih <= 0:
        return False
    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    return iw * ih >= min_share * max(1e-9, min(area_a, area_b))


def merged_formula_regions(
    merged: list[TextSpan],
    *,
    existing: list[DetectedRegion],
    dpi: int,
    page_w: float,
    page_h: float,
) -> list[DetectedRegion]:
    """Turn merged fragment spans into formula regions not covered by ``existing``."""
    out: list[DetectedRegion] = []
    pad = 4.0
    for sp in merged:
        bbox = BBox(
            x1=max(0.0, sp.bbox[0] - pad),
            y1=max(0.0, sp.bbox[1] - pad),
            x2=min(page_w, sp.bbox[2] + pad),
            y2=min(page_h, sp.bbox[3] + pad),
        )
        if any(_bbox_overlaps(bbox, r.bbox_pt) for r in [*existing, *out]):
            continue
        out.append(
            DetectedRegion(
                type=BlockType.FORMULA,
                bbox_pt=bbox,
                bbox_px=_pt_to_px(bbox, dpi),
                score=_merged_formula_score(sp),
                method="span_fragment_merge",
                notes=["merged_fragments", f"text={sp.text}"],
            )
        )
    return out


def detect_formula_regions_from_spans(
    spans: list[TextSpan],
    *,
    dpi: int,
    page_w: float,
    page_h: float,
    image_bgr: np.ndarray | None = None,
) -> list[DetectedRegion]:
    """Merge contiguous math-like text spans into formula boxes.

    ``image_bgr`` (the page render at ``dpi``) lets fragment recovery verify
    fraction bars; without it only text evidence is used.
    """
    scored: list[tuple[TextSpan, float]] = []
    for sp in spans:
        sc = _line_math_score(sp.text)
        if sc >= 0.55:
            scored.append((sp, sc))

    out: list[DetectedRegion] = []
    if scored:
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
                    notes=[
                        f"n_spans={len(cl)}",
                        "text=" + " | ".join(s.text.strip() for s, _ in cl),
                    ],
                )
            )

    out.extend(
        merged_formula_regions(
            merge_formula_spans(
                spans, page_w=page_w, page_h=page_h, image_bgr=image_bgr, dpi=dpi
            ),
            existing=out,
            dpi=dpi,
            page_w=page_w,
            page_h=page_h,
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


def _is_bibliography_like_text(texts: list[str]) -> bool:
    """Guard reference pages from fragment-based formula recovery."""
    joined = " ".join(t.strip() for t in texts if t.strip())
    if not joined:
        return False
    if re.search(
        r"\b(литература|библиограф|references|список\s+использованных)\b",
        joined,
        re.IGNORECASE,
    ):
        return True
    years = len(re.findall(r"\b(?:18|19|20)\d{2}\b", joined))
    citation_separators = len(re.findall(r"//|№|с\.\s*\d|pp?\.\s*\d", joined))
    return len(texts) >= 20 and years >= 3 and citation_separators >= 2


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
    spans_from_layer = True

    # Scanned pages often have empty text layer — build pseudo-spans via OCR
    if use_ocr_fallback and len(spans) < 5:
        spans = _ocr_pseudo_spans(rendered.image_bgr, dpi=dpi, doc_id=doc_id, page=page_number)
        spans_from_layer = False

    formula_regions = detect_formula_regions_from_spans(
        spans, dpi=dpi, page_w=page_w, page_h=page_h, image_bgr=rendered.image_bgr
    )
    # A text layer can miss an image-only formula band.  OCR the upper part and
    # add only validated merged fragments, never raw OCR lines: raw lines would
    # re-feed prose into the single-span heuristic.
    if (
        spans_from_layer
        and use_ocr_fallback
        and not any(r.bbox_pt.y1 < 0.45 * page_h for r in formula_regions)
        and not _formula_merge_blocked(spans, page_w=page_w, page_h=page_h)
    ):
        ocr_top = _ocr_pseudo_spans(
            rendered.image_bgr,
            dpi=dpi,
            doc_id=doc_id,
            page=page_number,
            y_range=(0.0, 0.45),
        )
        if ocr_top:
            formula_regions.extend(
                merged_formula_regions(
                    merge_formula_spans(
                        ocr_top,
                        page_w=page_w,
                        page_h=page_h,
                        image_bgr=rendered.image_bgr,
                        dpi=dpi,
                    ),
                    existing=formula_regions,
                    dpi=dpi,
                    page_w=page_w,
                    page_h=page_h,
                )
            )

    img_h, img_w = rendered.image_bgr.shape[:2]
    formula_regions = [
        r for r in formula_regions if _formula_crop_ok(r, img_w=img_w, img_h=img_h)
    ]

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
    # Recovered fragments inside a table or figure are cell text or axis
    # labels; ruled lines there imitate fraction bars.
    objects = [r for r in regions if r.type in (BlockType.TABLE, BlockType.FIGURE)]
    regions = [
        r
        for r in regions
        if r.method != "span_fragment_merge"
        or not any(_bbox_overlaps(r.bbox_pt, o.bbox_pt, min_share=0.05) for o in objects)
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
