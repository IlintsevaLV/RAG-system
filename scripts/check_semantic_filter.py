"""Checks for the formula semantic filter (issue 007).

1. Unit cases from the issue: LaTeX + context -> expected (ok | reason).
2. Replay over existing *_regions.json: recompute formula statuses offline from
   latex, bbox, page figures/tables and OCR source text, without UniMERNet.
   Regions of _1954.pdf page-tests have manual labels, so precision is reported.

Usage:
  python scripts/check_semantic_filter.py
  python scripts/check_semantic_filter.py --ir "data/ir/page_tests/*_final/_1954_regions.json"
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.ir import BBox
from ingestion.formula_pipeline import (
    _source_text,
    is_real_formula_semantic,
    normalize_and_validate,
)


def _bb(x1: float, y1: float, x2: float, y2: float) -> BBox:
    return BBox(x1=x1, y1=y1, x2=x2, y2=y2)


FIGURE = _bb(100, 100, 500, 400)
TABLE = _bb(80, 500, 500, 770)

# (name, latex, kwargs, expected reason or "" for a real formula)
UNIT_CASES: list[tuple[str, str, dict, str]] = [
    ("T3 p34 Tv/75N", r"\begin{array} { l } { \displaystyle { = \frac { T v } { 7 5 N } \, . } } \end{array}", {}, ""),
    ("T3 p40 dM_k", r"\stackrel { \smile } { d . M _ { \kappa } } = k \, \frac 1 { 2 } \, \rho \, ( \mathrm { o } r ) ^ { 2 } \, b \, ( c _ { x p } + \upbeta _ { * } c _ { y } ) \, r \, d r .", {}, ""),
    ("T4 p48 (Vy/2 + ...)", r"= ( \frac { V _ { y } } { 2 } + \frac { \sigma _ { - } a \upomega R } { 1 6 } ) \times", {}, ""),
    ("T6 p61 m_k", r"5 . \ \ \ m _ { \mathrm { \kappa } } \, = m _ { \mathrm { \kappa } \, p } + m _ { \mathrm { \kappa } \, i } .", {}, ""),
    ("T3 p36 q = G/N", r"q \! = \! \frac { G } { N } \! = \! \frac { 1 7 0 0 } { 1 7 0 } \! = \! 1 0", {}, ""),
    ("T4 p47 dT (III.2)", r"d \bar { T } { = } ( 2 \pi r \, d r \, _ { \theta } V _ { 1 } ) \, 2 v { = } 4 \pi \theta V _ { 1 } v r \, d r . \ ( \mathrm { I I I . } \, 2 )", {}, ""),
    ("const allowed", r"C = \mathrm { c o n s t }", {}, ""),
    ("T3 p34 TyIIdTe array", r"\begin{array} { l } { { \mathrm { T y I I d T e . i l b H z d . ~ c K o p o c T b ~ } \ V } } \end{array}", {}, "caption_or_header"),
    ("T3 p44 npu N = 200", r"\vert n p u \ N = 2 0 0", {}, "long_letter_run"),
    ("KT 160G heading", r"\mathbf{K H H T A \ I . \ O 6 m u e \Pi 0 . n o x e n H X}", {}, "caption_or_header"),
    ("sqrt phi next to figure", r"\sqrt { \varphi + 5 }", {"bbox_pt": _bb(150, 405, 180, 420), "page_figures": [FIGURE]}, ""),
    ("T5 p25 caption under figure", r"\begin{array} { r l } & { \qquad \qquad \qquad \quad \cdots ^ { m m \times n } } \end{array}", {"bbox_pt": _bb(150, 405, 450, 425), "page_figures": [FIGURE], "source_text": "а — потребная мощность"}, "figure_caption"),
    ("T6 6 = 0,06 in table", r"6 = 0 , 0 6", {"bbox_pt": _bb(200, 600, 260, 615), "page_tables": [TABLE]}, "table_value"),
    ("T6 m_k = 0,00052 in table", r"m _ { \mathrm { K } } { = } 0 , 0 0 0 5 2", {"bbox_pt": _bb(200, 600, 300, 615), "page_tables": [TABLE]}, "table_value"),
    ("formula number (I.7)", r"( \mathrm { I } . 7 )", {}, "formula_number"),
    ("T4 p48 empty numerator", r"\frac { \" } { { \texttt { e \_ { r } } } ^ { \alpha _ { \omega } } R }", {}, "truncated"),
    ("T3 p34 M = -", r"M \! = \! -", {}, "truncated"),
    ("T3 p34 eta = I/{}", r"\eta \! = \! \frac { \mathrm { ~ I ~ } } { }", {}, "truncated"),
    ("T6 p68 m_k values", r"m _ { \mathrm { { \scriptscriptstyle K } } } { = } 0 , 0 0 0 5 2 \qquad \biggl | \quad ^ { : } \ m _ { \mathrm { { \scriptscriptstyle K } } } { = } 0 , 0 0 0 8 8", {}, "numeric_value"),
    ("T3 p44 (N=200000, C=1225 KZ)", r"( N = 2 0 0 0 0 0 , \, C = 1 2 2 5 \, K Z )", {}, "numeric_value"),
    ("pure numbers", r"1 2 , 5 \times 3", {}, "pure_numbers"),
    ("russian words", r"\text{при этом} x = y", {}, "russian_words"),
    ("T3 p36 chart labels (OCR text)", r"M = O _ { 1 } \mathscr { U }", {"source_text": "μ = 0,75 (хороший несущий винт)"}, "prose_source_text"),
    ("no math markers", r"\mathrm { A } \, \mathrm { B }", {}, "no_math_markers"),
    ("tiny crop", r"x = y", {"raw_crop_wh": (15, 30)}, "tiny_crop"),
]

# Manual labels of _1954.pdf page-test formula regions from crops (2026-09-23):
# F real formula, G not a formula, T cut-off formula, M formula mixed with text,
# B real formula with unusable LaTeX. Key: (page, round(x1), round(y1)).
LABELS: dict[tuple[int, int, int], str] = {
    (34, 180, 509): "T", (34, 76, 568): "G", (34, 72, 715): "T", (34, 265, 742): "F",
    (36, 209, 73): "F", (36, 224, 232): "T", (36, 227, 303): "T", (36, 234, 474): "G",
    (37, 178, 112): "F", (37, 131, 254): "F", (37, 113, 428): "F",
    (40, 221, 101): "T", (40, 153, 278): "F", (40, 202, 427): "F", (40, 288, 652): "M",
    (40, 209, 711): "F", (40, 249, 745): "F",
    (41, 314, 111): "F", (41, 137, 184): "F", (41, 60, 252): "B", (41, 92, 261): "G",
    (41, 269, 447): "F", (41, 226, 576): "B", (41, 217, 664): "T",
    (44, 171, 226): "G", (44, 195, 309): "G", (44, 133, 466): "F", (44, 262, 570): "B",
    (47, 174, 78): "F", (47, 242, 271): "F", (47, 266, 320): "F", (47, 282, 615): "F",
    (47, 127, 779): "F",
    (48, 243, 80): "F", (48, 266, 180): "T", (48, 265, 260): "M", (48, 221, 514): "F",
    (51, 223, 113): "B", (51, 357, 299): "B",
    (25, 310, 437): "G",
    (61, 197, 33): "B", (61, 254, 90): "F", (61, 468, 446): "F", (61, 290, 494): "F",
    (68, 333, 146): "G", (68, 350, 445): "G", (68, 332, 733): "G",
}


def run_unit_cases() -> int:
    failed = 0
    for name, latex, kwargs, expected in UNIT_CASES:
        ok, reason = is_real_formula_semantic(latex, **kwargs)
        got = "" if ok else reason
        mark = "PASS" if got == expected else "FAIL"
        failed += got != expected
        print(f"  {mark}  {name:34s} expected={expected or 'ok':18s} got={got or 'ok'}")
    print(f"unit cases: {len(UNIT_CASES) - failed}/{len(UNIT_CASES)} passed")
    return failed


REPAIR_CASES: list[tuple[str, str, bool]] = [
    (r"\eta = \frac{I}{N}", False),
    (r"T = c_{\pi} \pi R^{2} \frac{\ell}{2} \left(\mathrm{{o} R\right)^{2}", True),
    (r"\begin{array}{c} { T = c _ { \pi } }", True),
]


def run_repair_cases() -> int:
    failed = 0
    for latex, expect_repair in REPAIR_CASES:
        fixed, ok, notes = normalize_and_validate(latex)
        repaired = "braces_repaired" in notes or "spacing_collapsed" in notes
        mark = "PASS" if repaired == expect_repair and (ok or not expect_repair) else "FAIL"
        failed += mark == "FAIL"
        print(f"  {mark}  repair {latex[:50]!r:52s} notes={notes} ok={ok}")
    print(f"repair cases: {len(REPAIR_CASES) - failed}/{len(REPAIR_CASES)} passed")
    return failed


def _page_boxes(page: dict, kind: str) -> list[BBox]:
    return [BBox(**b["bbox"]) for b in page["blocks"] if b["type"] == kind]


def replay_ir(pattern: str, dpi: int) -> None:
    scale = dpi / 72.0
    labelled: Counter[tuple[str, str]] = Counter()
    for path in sorted(glob.glob(pattern)):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        counts: Counter[str] = Counter()
        print(f"\n{Path(path).parent.name}")
        for page in data["pages"]:
            figures = _page_boxes(page, "figure")
            tables = _page_boxes(page, "table")
            for b in page["blocks"]:
                if b["type"] != "formula":
                    continue
                content = b.get("content") or {}
                old = content.get("status")
                bbox = BBox(**b["bbox"])
                new = old
                reason = ""
                if old == "ok":
                    ok, reason = is_real_formula_semantic(
                        content.get("latex") or "",
                        bbox_pt=bbox,
                        page_figures=figures,
                        page_tables=tables,
                        source_text=_source_text(b["quality"].get("notes") or []),
                        raw_crop_wh=(
                            int(round((bbox.x2 - bbox.x1) * scale)),
                            int(round((bbox.y2 - bbox.y1) * scale)),
                        ),
                    )
                    new = "ok" if ok else (
                        "caption_or_header" if reason in ("caption_or_header", "figure_caption") else "suspicious_semantic"
                    )
                counts[new] += 1
                key = (page["page"], round(bbox.x1), round(bbox.y1))
                label = LABELS.get(key, "?")
                labelled[(label, new)] += 1
                print(
                    f"  p{page['page']:03d} [{label}] {old:>10s} -> {new:19s} "
                    f"{reason:18s} {(content.get('latex') or '')[:60]!r}"
                )
        print(f"  total: {dict(counts)}")

    if not labelled:
        return
    ok_items = {lab: n for (lab, st), n in labelled.items() if st == "ok"}
    n_ok = sum(ok_items.values())
    n_ok_known = sum(n for lab, n in ok_items.items() if lab != "?")
    if n_ok_known:
        real = ok_items.get("F", 0)
        acceptable = real + ok_items.get("B", 0) + ok_items.get("M", 0)
        print(
            f"\nstatus=ok: {n_ok} regions, labelled {n_ok_known}; "
            f"real formula (F) {real} = {real / n_ok_known:.0%}, "
            f"formula region incl. bad LaTeX/mixed (F+B+M) {acceptable / n_ok_known:.0%}"
        )
    lost = [st for (lab, st), n in labelled.items() for _ in range(n) if lab == "F" and st != "ok"]
    print(f"real formulas (F) not ok: {len(lost)} ({Counter(lost)})")
    print("label x status:", dict(sorted(labelled.items())))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ir", default="data/ir/page_tests/*_final/_1954_regions.json")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args(argv)
    failed = run_unit_cases() + run_repair_cases()
    replay_ir(args.ir, args.dpi)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
