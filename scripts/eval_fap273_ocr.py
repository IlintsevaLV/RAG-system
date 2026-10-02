"""Word accuracy of OCR engines on data/ir/gold/fap273_ocr_v1.json.

  python -m scripts.eval_fap273_ocr
  python -m scripts.eval_fap273_ocr --engines rpd,rpd_server

Prints accuracy, CER, WER and seconds per page. Does not change the default engine.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fitz

from core.config import get_settings
from ingestion.ocr_rapid import RapidOCRConfig, rec_variant_for_engine, recognize_page_image
from ingestion.preprocess import process_page_image
from ingestion.models import PageRoute


def word_accuracy(reference: str, ocr: str) -> float:
    ref_words = set(reference.lower().split())
    if not ref_words:
        return 0.0
    ocr_words = set(ocr.lower().split())
    return len(ref_words & ocr_words) / len(ref_words)


def _edit_distance(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def wer(reference: str, ocr: str) -> float:
    ref = reference.lower().split()
    if not ref:
        return 0.0
    return _edit_distance(ref, ocr.lower().split()) / len(ref)


def cer(reference: str, ocr: str) -> float:
    ref = list(reference.lower())
    if not ref:
        return 0.0
    return _edit_distance(ref, list(ocr.lower())) / len(ref)


def _ocr_page(doc, page: int, engine: str, settings) -> tuple[str, float]:
    prep = process_page_image(
        doc,
        page,
        route=PageRoute.OCR,
        dpi_good=settings.render_dpi,
        dpi_bad=settings.render_dpi_bad,
        enable_binarize=False,
        force=True,
    )
    image = prep.get("image_bgr")
    if image is None:
        return "", 0.0
    cfg = RapidOCRConfig(
        use_gpu=False,
        max_side_len=settings.ocr_max_side_len,
        model_dir=str(settings.ocr_model_dir),
        enable_latin=settings.ocr_enable_latin,
        enable_greek=settings.ocr_enable_greek,
        rec_variant=rec_variant_for_engine(engine),
    )
    started = time.perf_counter()
    result = recognize_page_image(
        image,
        dpi=int(prep.get("dpi") or settings.render_dpi),
        page=page,
        doc_id="fap273",
        cfg=cfg,
    )
    elapsed = time.perf_counter() - started
    return result.text, elapsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score FAP 273 OCR against the golden pages")
    parser.add_argument("--gold", default="data/ir/gold/fap273_ocr_v1.json")
    parser.add_argument("--pdf", default=None)
    parser.add_argument("--engines", default="rpd,rpd_server")
    args = parser.parse_args(argv)

    gold_path = Path(args.gold)
    if not gold_path.is_file():
        print(f"Golden нет: {gold_path}")
        return 2
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    pdf = Path(args.pdf or gold.get("source_pdf") or "")
    if not pdf.is_file():
        print(f"PDF нет: {pdf}")
        return 2

    settings = get_settings()
    engines = [part.strip() for part in args.engines.split(",") if part.strip()]
    doc = fitz.open(pdf)
    try:
        print(f"{'engine':<12} {'page':>5} {'accuracy':>9} {'CER':>8} {'WER':>8} {'sec':>8}")
        totals: dict[str, list[float]] = {name: [] for name in engines}
        times: dict[str, list[float]] = {name: [] for name in engines}
        for engine in engines:
            for item in gold["pages"]:
                page = int(item["page"])
                reference = item["reference_text"]
                try:
                    text, elapsed = _ocr_page(doc, page, engine, settings)
                except FileNotFoundError as exc:
                    print(f"{engine:<12} {page:>5}  model missing: {exc}")
                    continue
                acc = word_accuracy(reference, text)
                totals[engine].append(acc)
                times[engine].append(elapsed)
                print(
                    f"{engine:<12} {page:>5} {acc:9.3f} {cer(reference, text):8.3f} "
                    f"{wer(reference, text):8.3f} {elapsed:8.1f}"
                )
        print()
        for engine in engines:
            if not totals[engine]:
                continue
            mean_acc = sum(totals[engine]) / len(totals[engine])
            mean_sec = sum(times[engine]) / len(times[engine])
            print(f"{engine}: mean accuracy {mean_acc:.3f}, {mean_sec:.1f} sec/page")
    finally:
        doc.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
