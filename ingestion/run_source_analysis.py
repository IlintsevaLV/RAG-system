"""CLI: TLQ → preprocess (OCR pages) → RapidOCR.

Examples:
  python -m ingestion.run_source_analysis path/to.pdf --pages 1-5 --ocr
  python -m ingestion.run_source_analysis path/to.pdf --pages 10 --ocr --force-ocr
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import get_settings
from core.logger import get_logger, setup_logging
from ingestion.models import PageRoute
from ingestion.ocr_rapid import RapidOCRConfig, recognize_page_image
from ingestion.preprocess import process_page_image, save_preview
from ingestion.source_analysis import analyze_pdf


def parse_pages(spec: str | None) -> list[int] | None:
    if not spec:
        return None
    pages: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            pages.extend(range(int(a), int(b) + 1))
        else:
            pages.append(int(part))
    return sorted(set(pages))


def iter_pdfs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.pdf")) + sorted(path.rglob("*.PDF"))


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    setup_logging(settings.log_level)
    log = get_logger("source_analysis")

    parser = argparse.ArgumentParser(description="TLQ + IQS preprocess + RapidOCR")
    parser.add_argument(
        "path",
        nargs="?",
        default=str(settings.normativka_dir),
        help="PDF file or directory (default: NORMATIVKA_DIR)",
    )
    parser.add_argument("--pages", default=None, help="e.g. 1-3,7")
    parser.add_argument("--doc-id", default=None)
    parser.add_argument("--threshold", type=float, default=None, help="TLQ threshold")
    parser.add_argument("--no-visual", action="store_true", help="Skip visual_agreement")
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Render+IQS+preprocess for OCR-routed pages",
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Run RapidOCR on OCR-routed pages (implies --preprocess)",
    )
    parser.add_argument(
        "--force-ocr",
        action="store_true",
        help="OCR even if TLQ accepted text_layer (debug / mixed pages)",
    )
    parser.add_argument("--resume", action="store_true", help="Skip if IR json already exists")
    parser.add_argument("--out-dir", default=None, help="Override IR_DIR")
    args = parser.parse_args(argv)

    if args.ocr:
        args.preprocess = True

    threshold = args.threshold if args.threshold is not None else settings.tlq_threshold
    enable_visual = settings.tlq_enable_visual_agreement and not args.no_visual
    out_dir = Path(args.out_dir) if args.out_dir else settings.ir_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = parse_pages(args.pages)

    targets = iter_pdfs(Path(args.path))
    if not targets:
        log.error("no_pdfs_found", path=args.path)
        return 1

    ocr_cfg = RapidOCRConfig(
        use_gpu=settings.enable_gpu_ocr,
        max_side_len=settings.ocr_max_side_len,
        band_trigger_px=settings.ocr_band_trigger_px,
        band_height=settings.ocr_band_height,
        model_dir=str(settings.ocr_model_dir),
    )

    for pdf in targets:
        doc_id = args.doc_id or pdf.stem
        out_json = out_dir / f"{doc_id}_source.json"
        if args.resume and out_json.exists():
            log.info("skip_resume", doc_id=doc_id, path=str(out_json))
            continue

        log.info("analyze_start", path=str(pdf), doc_id=doc_id)
        analysis = analyze_pdf(
            pdf,
            doc_id=doc_id,
            pages=pages,
            tlq_threshold=threshold,
            enable_visual=enable_visual,
            lexical_veto_threshold=settings.lexical_veto_threshold,
            lexical_min_tokens=settings.lexical_min_tokens,
        )

        preprocess_meta: list[dict] = []
        ocr_pages = 0
        if args.preprocess:
            doc = fitz.open(pdf)
            try:
                preview_dir = settings.cache_dir / "preprocess" / doc_id
                for page_res in analysis.pages:
                    need_ocr = page_res.tlq.route == PageRoute.OCR or args.force_ocr
                    result = process_page_image(
                        doc,
                        page_res.page,
                        route=page_res.tlq.route,
                        dpi_good=settings.render_dpi,
                        dpi_bad=settings.render_dpi_bad,
                        enable_binarize=settings.preprocess_enable_binarize,
                        force=args.force_ocr,
                    )
                    if result.get("iqs") is not None:
                        page_res.iqs = result["iqs"]
                        page_res.preprocess_plan = result.get("steps") or []

                    meta: dict = {
                        "page": page_res.page,
                        "route": page_res.tlq.route.value,
                        "skipped": result.get("skipped_preprocess"),
                        "steps": result.get("steps") or [],
                        "iqs_score": page_res.iqs.score if page_res.iqs else None,
                        "iqs_level": page_res.iqs.level.value if page_res.iqs else None,
                        "band_count": result.get("band_count", 0),
                    }

                    if result.get("image_bgr") is not None:
                        preview_path = preview_dir / f"page_{page_res.page:04d}.png"
                        save_preview(result["image_bgr"], preview_path)
                        meta["preview"] = str(preview_path)

                    if args.ocr and need_ocr and result.get("image_bgr") is not None:
                        ocr_res = recognize_page_image(
                            result["image_bgr"],
                            dpi=int(result.get("dpi") or settings.render_dpi),
                            page=page_res.page,
                            doc_id=doc_id,
                            cfg=ocr_cfg,
                            bands=result.get("bands"),
                        )
                        page_res.ocr = ocr_res
                        # Prefer OCR text when this page was OCR-routed / forced
                        page_res.extracted_text = ocr_res.text
                        ocr_pages += 1
                        meta["ocr_lines"] = ocr_res.line_count
                        meta["ocr_mean_conf"] = round(ocr_res.mean_confidence, 4)
                        meta["ocr_used_bands"] = ocr_res.used_bands

                    preprocess_meta.append(meta)
                    log.info(
                        "page_done",
                        page=page_res.page,
                        tlq=round(page_res.tlq.score, 3),
                        route=page_res.tlq.route.value,
                        iqs=meta["iqs_level"],
                        ocr_lines=meta.get("ocr_lines"),
                    )
            finally:
                doc.close()

        analysis.summary["ocr_pages"] = ocr_pages
        payload = analysis.model_dump()
        payload["preprocess_meta"] = preprocess_meta
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("analyze_done", out=str(out_json), summary=analysis.summary)

        print(f"\n=== {doc_id} ===")
        print(f"pages: {analysis.summary}")
        for p in analysis.pages:
            c = p.tlq.components
            print(
                f"  p{p.page:03d}  TLQ={p.tlq.score:.3f}  route={p.tlq.route.value}"
                f"  print={c.printable_ratio:.2f} lang={c.language_score:.2f}"
                f"  geom={c.geometry_score:.2f} dens={c.text_density_score:.2f}"
                f"  vis={c.visual_agreement if c.visual_agreement is not None else '-'}"
                f"  chars={p.tlq.char_count}"
            )
            if p.iqs:
                print(
                    f"         IQS={p.iqs.score:.3f} level={p.iqs.level.value}"
                    f" skew={p.iqs.estimated_skew_deg:.2f}° steps={p.preprocess_plan}"
                )
            if p.ocr:
                preview = (p.ocr.text[:120] + "…") if len(p.ocr.text) > 120 else p.ocr.text
                print(
                    f"         OCR lines={p.ocr.line_count} "
                    f"conf={p.ocr.mean_confidence:.3f} bands={p.ocr.used_bands}"
                )
                if preview:
                    print(f"         OCR text: {preview!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
