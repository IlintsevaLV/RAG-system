"""CLI: extract RAG-ready text by page class A/B/C/D.

Examples:
  python -m ingestion.run_extract_pages data/raw --pages 1-3
  python -m ingestion.run_extract_pages data/raw/doc.pdf --pages 83,246 --enable-vlm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import get_settings
from core.logger import get_logger, setup_logging
from ingestion.extract_page_text import extract_pdf


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
    return sorted({*path.rglob("*.pdf"), *path.rglob("*.PDF")})


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    setup_logging(settings.log_level)
    log = get_logger("run_extract")

    parser = argparse.ArgumentParser(description="Extract RAG text (classes A/B/C/D)")
    parser.add_argument("path", nargs="?", default=str(settings.normativka_dir))
    parser.add_argument("--pages", default=None)
    parser.add_argument("--doc-id", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--enable-vlm",
        action="store_true",
        help="Use VLM for C/D extract and optional A/B layout check",
    )
    parser.add_argument(
        "--no-ocr-fallback",
        action="store_true",
        help="For C/D: do not fall back to RapidOCR if VLM unavailable",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    if args.enable_vlm:
        settings.enable_vlm = True
    if args.no_ocr_fallback:
        settings.cd_ocr_fallback = False

    out_dir = Path(args.out_dir) if args.out_dir else settings.ir_dir / "pages"
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = parse_pages(args.pages)
    targets = iter_pdfs(Path(args.path))
    if not targets:
        log.error("no_pdfs", path=args.path)
        return 1

    for pdf in targets:
        doc_id = args.doc_id or pdf.stem
        out_json = out_dir / f"{doc_id}_pages.json"
        if args.resume and out_json.exists():
            log.info("skip_resume", path=str(out_json))
            continue

        result = extract_pdf(pdf, doc_id=doc_id, pages=pages, settings=settings)
        out_json.write_text(
            json.dumps(result.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # also write plain text dumps for RAG ingestion later
        txt_dir = out_dir / doc_id
        txt_dir.mkdir(parents=True, exist_ok=True)
        for p in result.pages:
            (txt_dir / f"page_{p.page:04d}.txt").write_text(p.text, encoding="utf-8")

        log.info("extract_done", out=str(out_json), summary=result.summary)
        print(f"\n=== {doc_id} === {result.summary}")
        for p in result.pages:
            preview = (p.text[:100] + "…") if len(p.text) > 100 else p.text
            print(
                f"  p{p.page:03d} class={p.page_class.value} status={p.status.value} "
                f"src={p.text_source} cols={p.layout.n_columns} "
                f"table={p.layout.likely_table} chars={len(p.text)}"
            )
            if preview:
                print(f"       {preview!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
