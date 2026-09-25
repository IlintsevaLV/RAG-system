"""Unit checks + optional IoU vs data/ir/gold/figures_v1.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.ir import BBox
from ingestion.models import TextSpan
from ingestion.region_detect import (
    FIGURE_CAPTION_RE,
    DetectedRegion,
    detect_caption_figure_regions,
    is_valid_figure_bbox,
    merge_figure_candidates,
    tighten_bbox_to_ink,
)
from core.ir import BlockType


def _iou(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    aa = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    bb = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    return inter / (aa + bb - inter + 1e-9)


def test_units() -> int:
    failed = 0
    img = np.full((400, 400, 3), 240, dtype=np.uint8)
    img[80:260, 90:310] = 20
    loose = BBox(x1=10, y1=10, x2=280, y2=280)
    tight = tighten_bbox_to_ink(img, loose, dpi=72)
    if 70 < tight.x1 < 100 and 65 < tight.y1 < 95 and 250 < tight.x2 < 325 and 245 < tight.y2 < 280:
        print("  PASS  tighten_bbox_to_ink")
    else:
        print(f"  FAIL  tighten got {tight}")
        failed += 1

    page_w, page_h = 378.0, 807.0
    ok, reason = is_valid_figure_bbox(
        BBox(x1=0, y1=0, x2=371, y2=791), page_w=page_w, page_h=page_h
    )
    if (not ok) and reason == "too_large":
        print("  PASS  reject full-page")
    else:
        print(f"  FAIL  full-page ok={ok} {reason}")
        failed += 1

    if FIGURE_CAPTION_RE.search("Фиг. I.1. Одновинтовой"):
        print("  PASS  caption regex")
    else:
        print("  FAIL  caption regex")
        failed += 1

    spans = [
        TextSpan(
            text="Фиг. I.1. схема",
            bbox=(90.0, 300.0, 280.0, 318.0),
            font_size=10,
        )
    ]
    found = detect_caption_figure_regions(spans, img, dpi=72, page_w=300, page_h=400)
    if found and found[0].method == "caption_anchor":
        print("  PASS  caption_anchor")
    else:
        print(f"  FAIL  caption_anchor {found}")
        failed += 1

    a = DetectedRegion(
        type=BlockType.FIGURE,
        bbox_pt=BBox(x1=10, y1=10, x2=100, y2=100),
        score=0.5,
        method="nontext_ink_cc",
    )
    b = DetectedRegion(
        type=BlockType.FIGURE,
        bbox_pt=BBox(x1=12, y1=12, x2=98, y2=96),
        score=0.8,
        method="pdf_image",
    )
    merged = merge_figure_candidates([a, b])
    if len(merged) == 1 and merged[0].method == "pdf_image":
        print("  PASS  merge prefers pdf_image")
    else:
        print(f"  FAIL  merge {[(m.method) for m in merged]}")
        failed += 1
    return failed


def replay_gold(gold_path: Path, pdf: Path | None, dpi: int, iou_thr: float) -> int:
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    pages = gold.get("pages") or []
    n_review = sum(1 for p in pages if p.get("needs_review"))
    print(f"gold pages={len(pages)} needs_review={n_review}")
    if pdf is None or not pdf.is_file():
        print("  skip IoU replay (no --pdf). Confirm boxes, then re-run with the file.")
        return 0

    import fitz

    from ingestion.region_detect import detect_page_regions

    doc = fitz.open(str(pdf))
    doc_id = pdf.stem
    tp = fp = fn = 0
    ious: list[float] = []
    for item in pages:
        if item.get("doc_id") not in {doc_id, pdf.stem.replace(" ", "_")}:
            # allow _1954 vs file stem
            if item.get("source") and Path(item["source"]).stem != pdf.stem:
                continue
        page_no = int(item["page"])
        _img, dets, _blocks, _spans = detect_page_regions(
            doc, page_no, doc_id=doc_id, dpi=dpi
        )
        pred = [d.bbox_pt for d in dets if d.type == BlockType.FIGURE]
        gold_bb = [
            BBox(**g["bbox"]) if isinstance(g["bbox"], dict) else BBox(*g["bbox"])
            for g in item.get("figures") or []
        ]
        used = set()
        for g in gold_bb:
            best_i, best = -1, 0.0
            for i, p in enumerate(pred):
                if i in used:
                    continue
                v = _iou(g, p)
                if v > best:
                    best, best_i = v, i
            if best >= iou_thr and best_i >= 0:
                tp += 1
                used.add(best_i)
                ious.append(best)
            else:
                fn += 1
                ious.append(best)
        fp += max(0, len(pred) - len(used))
        print(
            f"  p{page_no:04d} gold={len(gold_bb)} pred={len(pred)} "
            f"matched={len(used)}"
        )
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    mean_iou = sum(ious) / max(len(ious), 1)
    print(f"IoU>={iou_thr}: P={prec:.2f} R={rec:.2f} mean_best_iou={mean_iou:.2f} tp={tp} fp={fp} fn={fn}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="data/ir/gold/figures_v1.json")
    parser.add_argument("--pdf", default="")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    failed = test_units()
    print(f"unit: {5 - failed}/5")
    gold = Path(args.gold)
    if gold.is_file():
        iou_thr = float(json.loads(gold.read_text(encoding="utf-8")).get("iou_hit", 0.5))
        replay_gold(gold, Path(args.pdf) if args.pdf else None, args.dpi, iou_thr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
