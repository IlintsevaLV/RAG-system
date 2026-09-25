"""Помощник разметки golden set.

Пишет JSON-запись страницы либо в stdout, либо сразу в golden-файл
через --append-to. Второй режим не зависит от pipe и redirect, надёжно
работает в PowerShell 5.

Использование (сразу в golden, рекомендуемый режим):
    python scripts/gold_mark_helper.py \
      --doc-id _1954 --source "data/raw/_1954.pdf" --page 9 \
      --fig "171,830,1050,1190" \
      --caption "Фиг. I.1. Одновинтовой вертолет Белл Н-13." \
      --kind photo \
      --append-to data/ir/gold/figures_v2.json

Печать в stdout (старый режим):
    python scripts/gold_mark_helper.py ... (без --append-to)
"""
import argparse
import json
from pathlib import Path

import pymupdf as fitz  # PyMuPDF; новый namespace без warning

DPI = 150
SCALE = DPI / 72.0  # ≈ 2.0833


def px_to_pt(x1, y1, x2, y2):
    return {
        "x1": round(x1 / SCALE, 1),
        "y1": round(y1 / SCALE, 1),
        "x2": round(x2 / SCALE, 1),
        "y2": round(y2 / SCALE, 1),
    }


def parse_fig(spec: str):
    parts = spec.split(",")
    if len(parts) != 4:
        raise SystemExit(
            f"bad --fig spec: {spec!r} (expected 'x1,y1,x2,y2' in px)"
        )
    try:
        x1, y1, x2, y2 = (float(v) for v in parts)
    except ValueError as e:
        raise SystemExit(f"bad --fig numbers: {spec!r} ({e})")
    if not (x2 > x1 and y2 > y1):
        raise SystemExit(f"--fig: x2<=x1 or y2<=y1 in {spec!r}")
    return px_to_pt(x1, y1, x2, y2)


def get_page_size(pdf_path: str, page_no: int):
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_no - 1]
        return float(page.rect.width), float(page.rect.height)
    finally:
        doc.close()


def load_gold(path: Path) -> dict:
    if not path.exists():
        return {
            "version": 2,
            "iou_hit": 0.5,
            "note": "Extended golden set for figure detector. "
                    "v1 remains regression baseline.",
            "pages": [],
        }
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_gold(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def append_record(gold_path: Path, record: dict) -> str:
    gold = load_gold(gold_path)
    pages = gold.setdefault("pages", [])
    key = (record["doc_id"], record["page"])
    for i, p in enumerate(pages):
        if (p.get("doc_id"), p.get("page")) == key:
            pages[i] = record
            pages.sort(key=lambda p: (p.get("doc_id", ""), p.get("page", 0)))
            save_gold(gold_path, gold)
            return f"REPLACED page {key} in {gold_path} (total {len(pages)})"
    pages.append(record)
    pages.sort(key=lambda p: (p.get("doc_id", ""), p.get("page", 0)))
    save_gold(gold_path, gold)
    return f"ADDED page {key} to {gold_path} (total {len(pages)})"


def main():
    ap = argparse.ArgumentParser(
        description="Build one golden-set page record (bbox in 150 dpi px -> pt).",
    )
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--source", required=True,
                    help="PDF path relative to repo root")
    ap.add_argument("--page", type=int, required=True,
                    help="1-based page number")
    ap.add_argument("--fig", action="append", default=[],
                    help="x1,y1,x2,y2 in 150 dpi px; repeatable")
    ap.add_argument("--caption", action="append", default=[],
                    help="caption text; repeatable, paired with --fig by index")
    ap.add_argument("--kind", action="append", default=[],
                    help="scheme|chart|photo|drawing; repeatable, paired with --fig")
    ap.add_argument("--note", default="")
    ap.add_argument("--needs-review", action="store_true")
    ap.add_argument("--no-figures", action="store_true",
                    help="explicit negative page; fails if --fig is also given")
    ap.add_argument("--append-to", type=Path, default=None,
                    help="write result directly into this golden JSON file")
    args = ap.parse_args()

    if args.no_figures and args.fig:
        raise SystemExit("--no-figures cannot be combined with --fig")

    if len(args.caption) > len(args.fig):
        raise SystemExit("more --caption than --fig")
    if len(args.kind) > len(args.fig):
        raise SystemExit("more --kind than --fig")

    page_w, page_h = get_page_size(args.source, args.page)

    figures = []
    for i, fig_spec in enumerate(args.fig):
        bbox = parse_fig(fig_spec)
        caption = args.caption[i] if i < len(args.caption) else ""
        kind = args.kind[i] if i < len(args.kind) else "photo"
        figures.append({"bbox": bbox, "caption": caption, "kind": kind})

    record = {
        "doc_id": args.doc_id,
        "source": args.source,
        "page": args.page,
        "page_w": round(page_w, 2),
        "page_h": round(page_h, 2),
        "needs_review": bool(args.needs_review),
        "note": args.note,
        "figures": figures,
    }

    if args.append_to is not None:
        msg = append_record(args.append_to, record)
        print(msg)
    else:
        print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()