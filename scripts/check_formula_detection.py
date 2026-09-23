# -*- coding: utf-8 -*-
"""Detection-only regression check for the 10 page-tests of `_1954.pdf`.

Runs `detect_page_regions` (no UniMERNet / VLM) and prints, per test, how many
formula / table / figure regions were found plus the source text of every
formula region.  OCR results are cached on disk so several detector versions
can be compared quickly.

Examples:
  python scripts/check_formula_detection.py вычитка/_1954.pdf
  python scripts/check_formula_detection.py data/raw/_1954.pdf --tests T6,T7
  python scripts/check_formula_detection.py data/raw/_1954.pdf --detector-file old_region_detect.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fitz  # noqa: E402

PAGE_TESTS: dict[str, list[int]] = {
    "T1_title": [1, 2, 3],
    "T2_table_typo": [3],
    "T3_equations": [34, 36, 37, 40, 41, 44],
    "T4_greek": [47, 48, 51],
    "T5_figures": [9, 11, 15, 20, 25],
    "T6_real_tables": [61, 68, 69, 189],
    "T7_bibliography": [246, 247, 250],
    "T8_toc": [252, 253, 254],
    "T9_twocolumn": [23, 24],
    "T10_dense_text": [10, 30, 102, 190],
}

# Tests where any formula region is a false positive.
ZERO_FORMULA_TESTS = {"T7_bibliography", "T8_toc", "T9_twocolumn", "T10_dense_text"}


def _load_detector(path: Path | None):
    if path is None:
        from ingestion import region_detect

        return region_detect
    spec = importlib.util.spec_from_file_location("region_detect_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _install_ocr_cache(rd, cache_dir: Path, pdf: Path) -> None:
    original = rd._ocr_pseudo_spans
    cache_dir.mkdir(parents=True, exist_ok=True)

    def cached(image_bgr, *, dpi, doc_id, page, **kwargs):
        key_src = f"{pdf.resolve()}|{page}|{dpi}|{kwargs.get('y_range')}|{image_bgr.shape}"
        key = hashlib.sha1(key_src.encode("utf-8")).hexdigest()[:16]
        path = cache_dir / f"{doc_id}_p{page:04d}_{key}.pkl"
        if path.is_file():
            return pickle.loads(path.read_bytes())
        spans = original(image_bgr, dpi=dpi, doc_id=doc_id, page=page, **kwargs)
        path.write_bytes(pickle.dumps(spans))
        return spans

    rd._ocr_pseudo_spans = cached


def _formula_text(region, spans) -> str:
    for note in region.notes:
        if note.startswith("text="):
            return note[len("text="):]
    b = region.bbox_pt
    inside = [
        s.text.strip()
        for s in spans
        if s.bbox[0] >= b.x1 - 1
        and s.bbox[1] >= b.y1 - 1
        and s.bbox[2] <= b.x2 + 1
        and s.bbox[3] <= b.y2 + 1
    ]
    return " | ".join(t for t in inside if t)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--tests", default="", help="comma list of test-name prefixes")
    ap.add_argument("--dpi", type=int, default=None)
    ap.add_argument("--detector-file", type=Path, default=None)
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "cache" / "ocr_spans")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    from core.config import get_settings

    dpi = args.dpi or get_settings().render_dpi
    rd = _load_detector(args.detector_file)
    _install_ocr_cache(rd, args.cache_dir, args.pdf)

    wanted = [t.strip() for t in args.tests.split(",") if t.strip()]
    tests = {
        name: pages
        for name, pages in PAGE_TESTS.items()
        if not wanted or any(name.startswith(w) for w in wanted)
    }

    doc = fitz.open(args.pdf)
    doc_id = args.pdf.stem
    page_cache: dict[int, dict] = {}
    report: dict[str, list[dict]] = {}
    failures: list[str] = []
    try:
        for name, pages in tests.items():
            rows = []
            for pno in pages:
                if pno not in page_cache:
                    _img, regions, _blocks, spans = rd.detect_page_regions(
                        doc, pno, doc_id=doc_id, dpi=dpi
                    )
                    page_cache[pno] = {
                        "page": pno,
                        "formula": [
                            {
                                "method": r.method,
                                "score": round(r.score, 3),
                                "bbox": [
                                    round(r.bbox_pt.x1, 1),
                                    round(r.bbox_pt.y1, 1),
                                    round(r.bbox_pt.x2, 1),
                                    round(r.bbox_pt.y2, 1),
                                ],
                                "notes": [n for n in r.notes if not n.startswith("text=")],
                                "text": _formula_text(r, spans),
                            }
                            for r in regions
                            if r.type.value == "formula"
                        ],
                        "table": sum(1 for r in regions if r.type.value == "table"),
                        "figure": sum(1 for r in regions if r.type.value == "figure"),
                    }
                rows.append(page_cache[pno])
            report[name] = rows
            n_f = sum(len(r["formula"]) for r in rows)
            n_t = sum(r["table"] for r in rows)
            n_g = sum(r["figure"] for r in rows)
            print(f"{name:18s} formula={n_f:2d} table={n_t} figure={n_g}")
            for r in rows:
                for f in r["formula"]:
                    tag = "MERGED" if any(n.startswith("merged") for n in f["notes"]) else "span"
                    print(f"    p{r['page']:03d} [{tag}] {f['text'][:90]!r}")
            if name in ZERO_FORMULA_TESTS and n_f:
                failures.append(f"{name}: formula={n_f} (expected 0)")
    finally:
        doc.close()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if failures:
        print("\nFAIL:\n  " + "\n  ".join(failures))
        return 1
    print("\nOK: zero-formula tests are clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
