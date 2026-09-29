"""Draw golden (red) vs detector (blue) formula boxes on rdk89 pages.

Coordinates stay in PDF points — PyMuPDF scales them when rendering.
"""
from __future__ import annotations

import json
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[3]
GOLD = ROOT / "data/ir/gold/formulas_v1.json"
REGIONS = ROOT / "data/ir/regions/РДК_молниезащита 89г_regions.json"
PDF = ROOT / "data/raw/РДК_молниезащита 89г.pdf"
OUT_DIR = Path(__file__).resolve().parent


def main() -> None:
    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    gold_by_page = {
        p["page"]: p.get("formulas", [])
        for p in gold["pages"]
        if "89" in (p.get("doc_id") or "") or "РДК" in (p.get("source") or "")
    }
    regions = json.loads(REGIONS.read_text(encoding="utf-8"))
    reg_by_page = {
        p["page"]: [d for d in p.get("detections", []) if d.get("type") == "formula"]
        for p in regions.get("pages", [])
    }

    doc = fitz.open(str(PDF))
    for pg in (262, 274):
        page = doc[pg - 1]
        shape = page.new_shape()
        for f in gold_by_page.get(pg, []):
            b = f.get("bbox", {})
            shape.draw_rect(fitz.Rect(b["x1"], b["y1"], b["x2"], b["y2"]))
            shape.finish(color=(1, 0, 0), width=1.5)
        for d in reg_by_page.get(pg, []):
            b = d.get("bbox") or {}
            if not all(k in b for k in ("x1", "y1", "x2", "y2")):
                continue
            shape.draw_rect(fitz.Rect(b["x1"], b["y1"], b["x2"], b["y2"]))
            shape.finish(color=(0, 0, 1), width=1.5)
        shape.commit()
        pix = page.get_pixmap(dpi=150)
        out = OUT_DIR / f"_rdk89_p{pg}_overlay.png"
        pix.save(str(out))
        print(f"saved {out}")


if __name__ == "__main__":
    main()
