"""CLI: detect + process formula/table/figure regions.

Examples:
  python -m ingestion.run_regions data/raw/doc.pdf --pages 41
  python -m ingestion.run_regions data/raw --pages 1-3 --enable-vlm
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
from ingestion.process_regions import process_pdf_regions


def _safe_print(msg: str) -> None:
    try:
        print(msg)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(msg.encode(enc, errors="replace").decode(enc, errors="replace"))


def parse_pages(spec: str | None) -> list[int] | None:
    if not spec:
        return None
    pages: list[int] = []
    for part in spec.split(","):
        part = part.strip()
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
    log = get_logger("run_regions")

    parser = argparse.ArgumentParser(description="Formula/table/figure region pipelines")
    parser.add_argument("path", nargs="?", default=str(settings.normativka_dir))
    parser.add_argument("--pages", default=None)
    parser.add_argument("--doc-id", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--enable-vlm", action="store_true")
    parser.add_argument("--enable-unimernet", action="store_true")
    args = parser.parse_args(argv)

    if args.enable_vlm:
        settings.enable_vlm = True
    if args.enable_unimernet:
        settings.enable_unimernet = True

    out_dir = Path(args.out_dir) if args.out_dir else settings.ir_dir / "regions"
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = parse_pages(args.pages)
    targets = iter_pdfs(Path(args.path))
    if not targets:
        log.error("no_pdfs", path=args.path)
        return 1

    for pdf in targets:
        doc_id = args.doc_id or pdf.stem
        result = process_pdf_regions(pdf, doc_id=doc_id, pages=pages, settings=settings)
        out_path = out_dir / f"{doc_id}_regions.json"
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("regions_done", out=str(out_path))
        _safe_print(f"\n=== {doc_id} === -> {out_path}")
        for pg in result["pages"]:
            s = pg["summary"]
            _safe_print(
                f"  p{s['page']:03d} det={s['n_detections']} "
                f"formula={s['n_formulas']}(ok={s['formula_ok']}) "
                f"table={s['n_tables']}(ok={s['table_ok']}) "
                f"figure={s['n_figures']}"
            )
            for b in pg["blocks"]:
                ctype = b["type"]
                content = b.get("content") or {}
                if ctype == "formula":
                    preview = (content.get("latex") or "")[:80]
                elif ctype == "table":
                    preview = (content.get("markdown") or "").replace("\n", " ")[:80]
                else:
                    preview = (content.get("caption") or content.get("crop_path") or "")[:80]
                _safe_print(f"       [{ctype}] {b['region_id']} {preview!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
