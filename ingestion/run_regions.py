"""CLI: detect + process formula/table/figure regions.

Examples:
  python -m ingestion.run_regions data/raw/doc.pdf --pages 41
  python -m ingestion.run_regions data/raw --pages 1-3 --enable-vlm
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import sys
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import get_settings
from core.logger import get_logger, setup_logging
from ingestion.process_regions import process_pdf_regions


def _force_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure") and (getattr(stream, "encoding", "") or "").lower() != "utf-8":
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _safe_print(msg: str) -> None:
    try:
        print(msg)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(msg.encode(enc, errors="replace").decode(enc, errors="replace"))


def _load_regions(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _write_regions(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _available_ram_bytes() -> int | None:
    """Free physical RAM. Used to shrink a batch before a long PDF runs out of memory."""
    if sys.platform != "win32":
        return None
    class _Mem(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    stat = _Mem()
    stat.dwLength = ctypes.sizeof(stat)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
        return None
    return int(stat.ullAvailPhys)


def _pdf_fingerprint(path: Path) -> dict:
    st = path.stat()
    return {"mtime_ns": st.st_mtime_ns, "size": st.st_size}


def _load_hashes(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


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
    _force_utf8_stdio()
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip a PDF whose regions JSON already covers every page",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Keep pages already stored in the regions JSON and fill the rest",
    )
    parser.add_argument(
        "--batch-size",
        "--max-pages-per-batch",
        type=int,
        default=200,
        dest="batch_size",
        help="Save regions JSON after this many pages (limits RAM on long PDFs)",
    )
    parser.add_argument(
        "--skip-unchanged",
        action="store_true",
        help="Skip a PDF whose mtime and size match data/cache/pdf_hashes.json",
    )
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

    batch_size = max(1, args.batch_size)
    free = _available_ram_bytes()
    if free is not None and free < 1 << 30:
        batch_size = 1
        log.warning("low_ram_batch", free_mb=free // (1 << 20), batch_size=1)
    hash_path = Path(settings.cache_dir) / "pdf_hashes.json"
    hashes = _load_hashes(hash_path) if args.skip_unchanged else {}
    for pdf in targets:
        doc_id = (args.doc_id or pdf.stem).strip().rstrip(".")
        fingerprint = _pdf_fingerprint(pdf)
        if args.skip_unchanged and hashes.get(doc_id) == fingerprint:
            _safe_print(f"\n=== {doc_id} === unchanged, skip")
            continue
        out_path = out_dir / f"{doc_id}_regions.json"
        existing = _load_regions(out_path) if (args.resume or args.merge) else None
        with fitz.open(pdf) as opened:
            total = opened.page_count
        wanted = pages or list(range(1, total + 1))
        wanted = [p for p in wanted if 1 <= p <= total]
        have = {int(p["page"]) for p in (existing or {}).get("pages", [])}
        if args.resume and not args.merge and have and set(wanted) <= have:
            log.info("skip_resume", path=str(out_path))
            _safe_print(f"\n=== {doc_id} === resume skip {out_path}")
            continue
        if args.merge:
            wanted = [p for p in wanted if p not in have]
            if not wanted:
                log.info("skip_merge_complete", path=str(out_path))
                _safe_print(f"\n=== {doc_id} === merge complete {out_path}")
                continue
        acc = existing if args.merge and existing else {
            "doc_id": doc_id,
            "source_path": str(pdf.resolve()),
            "page_count": total,
            "pages": [],
        }
        by_page = {int(p["page"]): p for p in acc.get("pages", [])}
        for start in range(0, len(wanted), batch_size):
            chunk = wanted[start : start + batch_size]
            part = process_pdf_regions(pdf, doc_id=doc_id, pages=chunk, settings=settings)
            for page in part["pages"]:
                by_page[int(page["page"])] = page
            acc["doc_id"] = doc_id
            acc["source_path"] = str(pdf.resolve())
            acc["page_count"] = total
            acc["pages"] = [by_page[k] for k in sorted(by_page)]
            _write_regions(out_path, acc)
            gc.collect()
            log.info("regions_batch", out=str(out_path), pages=f"{chunk[0]}-{chunk[-1]}")
        if args.skip_unchanged:
            hashes[doc_id] = fingerprint
            hash_path.parent.mkdir(parents=True, exist_ok=True)
            hash_path.write_text(
                json.dumps(hashes, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        result = acc
        log.info("regions_done", out=str(out_path))
        _safe_print(f"\n=== {doc_id} === -> {out_path}")
        for pg in result["pages"]:
            if pages and int(pg["page"]) not in pages:
                continue
            s = pg["summary"]
            _safe_print(
                f"  p{s['page']:03d} det={s['n_detections']} "
                f"formula={s['n_formulas']}(ok={s['formula_ok']}, "
                f"sem={s.get('formula_suspicious_semantic', 0)}, "
                f"susp={s.get('formula_suspicious', 0)}, "
                f"caption={s.get('formula_caption_or_header', 0)}) "
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
