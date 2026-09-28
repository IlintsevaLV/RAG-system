"""Метрики formula detector: сравнение golden с data/ir/regions/*.json.

Загружает golden формул и готовые regions (посчитанные run_regions),
считает P/R/IoU по bbox'ам.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def iou(a, b):
    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def find_regions_file(doc_id: str, regions_dir: Path):
    """Найти regions-файл по doc_id (имя может отличаться от doc_id)."""
    # прямое совпадение
    candidates = list(regions_dir.glob("*.json"))
    for c in candidates:
        if c.stem.startswith(doc_id):
            return c
    # fuzzy: по первому слову doc_id
    first = doc_id.split()[0]
    for c in candidates:
        if first in c.stem:
            return c
    return None


def load_regions_for_doc(doc_id: str, regions_dir: Path, source_path: str | None):
    """Загрузить regions-файл, соответствующий doc_id."""
    path = find_regions_file(doc_id, regions_dir)
    if path is None and source_path:
        # fallback: по имени PDF
        stem = Path(source_path).stem
        for c in regions_dir.glob("*.json"):
            if stem in c.stem or c.stem in stem:
                path = c
                break
    if path is None:
        return None, None
    with path.open(encoding="utf-8") as f:
        return json.load(f), path


def get_regions_formulas(regions_data, page_no):
    """Взять из regions все формулы на указанной странице."""
    out = []
    for p in regions_data.get("pages", []):
        if p.get("page") != page_no:
            continue
        for b in p.get("blocks", []):
            if b.get("type") == "formula":
                bb = b.get("bbox", {})
                if "x1" not in bb:
                    continue
                out.append({
                    "x1": float(bb["x1"]), "y1": float(bb["y1"]),
                    "x2": float(bb["x2"]), "y2": float(bb["y2"]),
                    "latex": (b.get("content", {}) or {}).get("latex", ""),
                    "status": (b.get("content", {}) or {}).get("status", ""),
                })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="data/ir/gold/formulas_v1.json")
    ap.add_argument("--regions-dir", default="data/ir/regions")
    ap.add_argument("--doc-id", default=None, help="фильтр по doc_id")
    ap.add_argument("--iou-hit", type=float, default=0.5)
    args = ap.parse_args()

    gold_path = Path(args.gold)
    if not gold_path.exists():
        raise SystemExit(f"golden not found: {gold_path}")

    with gold_path.open(encoding="utf-8") as f:
        gold = json.load(f)

    pages = gold.get("pages", [])
    if args.doc_id:
        pages = [p for p in pages if p.get("doc_id") == args.doc_id]

    regions_dir = Path(args.regions_dir)

    print(f"gold pages = {len(pages)}  iou_hit={args.iou_hit}")
    print()

    tp = fp = fn = 0
    iou_sum = 0.0
    matched_count = 0

    for p in pages:
        doc_id = p.get("doc_id")
        page_no = p.get("page")
        gold_formulas = p.get("formulas", [])
        source = p.get("source")

        regions_data, regions_path = load_regions_for_doc(doc_id, regions_dir, source)
        if regions_data is None:
            print(f"  {doc_id[:30]} p{page_no}: SKIP (regions not found)")
            continue

        pred = get_regions_formulas(regions_data, page_no)

        # матчинг: для каждой golden ищем лучший pred (жадно, без повторов)
        matched_gold = set()
        matched_pred = set()
        for gi, g in enumerate(gold_formulas):
            best_iou = 0.0
            best_pi = -1
            for pi, pr in enumerate(pred):
                if pi in matched_pred:
                    continue
                v = iou(g["bbox"], pr)
                if v > best_iou:
                    best_iou = v
                    best_pi = pi
            if best_iou >= args.iou_hit:
                matched_gold.add(gi)
                matched_pred.add(best_pi)
                iou_sum += best_iou
                matched_count += 1

        page_tp = len(matched_gold)
        page_fp = len(pred) - len(matched_pred)
        page_fn = len(gold_formulas) - len(matched_gold)
        tp += page_tp
        fp += page_fp
        fn += page_fn

        print(f"  {doc_id[:30]:32s} p{page_no:4d}  gold={len(gold_formulas)} "
              f"pred={len(pred)} matched={page_tp}  "
              f"tp={page_tp} fp={page_fp} fn={page_fn}")

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    mean_iou = iou_sum / matched_count if matched_count else 0.0

    print()
    print(f"IoU>=0.5: P={prec:.2f} R={rec:.2f} mean_best_iou={mean_iou:.2f} "
          f"tp={tp} fp={fp} fn={fn}")


if __name__ == "__main__":
    main()