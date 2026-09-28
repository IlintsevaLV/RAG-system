"""Помощник разметки golden формул.

Пишет JSON-запись страницы в golden-файл формул.
Работает так же, как gold_mark_helper.py, но для формул.

Использование:
    python scripts/gold_mark_formulas.py \
      --doc-id _1954 --source "data/raw/_1954.pdf" --page 7 \
      --formula "80.6,582.4,280.0,603.8|\lambda = \frac{V \sin \alpha + v}{\omega R}|inline|ok" \
      --append-to data/ir/gold/formulas_v1.json

Негатив:
    python scripts/gold_mark_formulas.py \
      --doc-id _1954 --source "data/raw/_1954.pdf" --page 1 \
      --no-formulas --note "title page" \
      --append-to data/ir/gold/formulas_v1.json

Формат --formula:
    x1,y1,x2,y2|latex|kind|status
    - координаты в PDF points (не пиксели!)
    - latex — эталонный LaTeX (можно пустой)
    - kind: inline|display|fraction|symbol
    - status: ok|suspicious|wrong
"""
import argparse
import json
from pathlib import Path

import pymupdf as fitz


def parse_formula(spec: str):
    parts = spec.split("|")
    if len(parts) < 1:
        raise SystemExit(f"bad --formula spec: {spec!r}")
    coords = parts[0].split(",")
    if len(coords) != 4:
        raise SystemExit(f"bad coords: {parts[0]!r} (expected 'x1,y1,x2,y2')")
    x1, y1, x2, y2 = (float(v) for v in coords)
    latex = parts[1] if len(parts) > 1 else ""
    kind = parts[2] if len(parts) > 2 else "inline"
    status = parts[3] if len(parts) > 3 else "ok"
    if x2 <= x1 or y2 <= y1:
        raise SystemExit(f"bad bbox: x2<=x1 or y2<=y1 in {spec!r}")
    return {
        "bbox": {"x1": round(x1,1), "y1": round(y1,1),
                 "x2": round(x2,1), "y2": round(y2,1)},
        "latex": latex,
        "kind": kind,
        "status": status,
    }


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
            "version": 1,
            "iou_hit": 0.5,
            "note": "Formula golden set. Reviewed 2026-09-28.",
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--page", type=int, required=True)
    ap.add_argument("--formula", action="append", default=[],
                    help="x1,y1,x2,y2|latex|kind|status; repeatable")
    ap.add_argument("--note", default="")
    ap.add_argument("--no-formulas", action="store_true")
    ap.add_argument("--append-to", type=Path, required=True)
    args = ap.parse_args()

    if args.no_formulas and args.formula:
        raise SystemExit("--no-formulas cannot be combined with --formula")

    page_w, page_h = get_page_size(args.source, args.page)

    formulas = [parse_formula(s) for s in args.formula]

    record = {
        "doc_id": args.doc_id,
        "source": args.source,
        "page": args.page,
        "page_w": round(page_w, 2),
        "page_h": round(page_h, 2),
        "needs_review": False,
        "note": args.note,
        "formulas": formulas,
    }

    msg = append_record(args.append_to, record)
    print(msg)


if __name__ == "__main__":
    main()