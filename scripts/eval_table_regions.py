# -*- coding: utf-8 -*-
"""Evaluate table-region detection on selected PDF pages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import fitz

from ingestion.preprocess import render_page
from ingestion.region_detect import (
    _ocr_pseudo_spans,
    detect_table_regions_cv,
    detect_table_regions_from_spans,
    merge_regions,
)
from ingestion.source_analysis import _extract_spans


def eval_page(doc: fitz.Document, page_number: int, *, dpi: int, doc_id: str) -> dict:
    page = doc[page_number - 1]
    page_w, page_h = float(page.rect.width), float(page.rect.height)
    img = render_page(doc, page_number, dpi=dpi).image_bgr
    _, spans = _extract_spans(page)
    if len(spans) < 5:
        spans = _ocr_pseudo_spans(img, dpi=dpi, doc_id=doc_id, page=page_number)
    cv = detect_table_regions_cv(
        img, dpi=dpi, page_w=page_w, page_h=page_h
    )
    sp = detect_table_regions_from_spans(
        spans, dpi=dpi, page_w=page_w, page_h=page_h
    )
    merged = merge_regions([*cv, *sp])
    tables = [r for r in merged if r.type.value == "table"]
    return {
        "page": page_number,
        "n_spans": len(spans),
        "n_cv": len(cv),
        "n_span": len(sp),
        "n_tables": len(tables),
        "tables": [
            {
                "method": r.method,
                "score": round(r.score, 3),
                "bbox_pt": [r.bbox_pt.x1, r.bbox_pt.y1, r.bbox_pt.x2, r.bbox_pt.y2],
                "notes": r.notes,
            }
            for r in tables
        ],
        "sample_spans": [s.text for s in spans[:6]],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--pages", default="5,10,41,61,81")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("data/ir/table_detect_eval.json"))
    args = ap.parse_args()
    pages = [int(x) for x in args.pages.replace(" ", "").split(",") if x]
    doc = fitz.open(args.pdf)
    doc_id = args.pdf.stem
    results = [eval_page(doc, p, dpi=args.dpi, doc_id=doc_id) for p in pages]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"pdf": str(args.pdf), "dpi": args.dpi, "pages": results}
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for r in results:
        print(
            f"p{r['page']}: spans={r['n_spans']} cv={r['n_cv']} "
            f"span={r['n_span']} tables={r['n_tables']} "
            f"{[t['method']+':'+str(t['score']) for t in r['tables']]}"
        )
    print("wrote", args.out)


if __name__ == "__main__":
    main()
