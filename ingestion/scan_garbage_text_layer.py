"""Scan PDFs for garbage / OCR-like embedded text layers.

On the work PC (after git pull):

  pip install -r requirements.txt
  python -m ingestion.scan_garbage_text_layer data\\raw
  python -m ingestion.scan_garbage_text_layer data\\raw --pages-per-doc 8
  python -m ingestion.scan_garbage_text_layer data\\raw --full --no-visual

Outputs:
  data/ir/garbage_text_layer_report.json
  data/ir/garbage_text_layer_candidates.csv

Candidates = pages with a non-empty text layer that fails lexical quality
(garbage_veto) OR has low lexical_quality — prioritized for proofreading.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import get_settings
from core.logger import get_logger, setup_logging
from ingestion.source_analysis import analyze_pdf


def iter_pdfs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted({*path.rglob("*.pdf"), *path.rglob("*.PDF")})


def sample_pages(page_count: int, n: int) -> list[int]:
    """Evenly sample up to n 1-based page numbers."""
    if page_count <= 0:
        return []
    if n <= 0 or n >= page_count:
        return list(range(1, page_count + 1))
    if n == 1:
        return [1]
    # include first, last, and evenly spaced middle pages
    idxs = sorted(
        {
            1,
            page_count,
            *[1 + round(i * (page_count - 1) / (n - 1)) for i in range(n)],
        }
    )
    return [int(i) for i in idxs if 1 <= i <= page_count][:n]


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    setup_logging(settings.log_level)
    log = get_logger("scan_garbage")

    parser = argparse.ArgumentParser(
        description="Find PDFs/pages with garbage embedded text layer"
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=str(settings.normativka_dir),
        help="PDF file or directory (default NORMATIVKA_DIR)",
    )
    parser.add_argument(
        "--pages-per-doc",
        type=int,
        default=8,
        help="Sample this many pages per PDF (default 8). Ignored with --full",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Analyze every page (slow on large corpora)",
    )
    parser.add_argument(
        "--no-visual",
        action="store_true",
        help="Skip visual_agreement (faster; lexical veto still runs)",
    )
    parser.add_argument(
        "--suspect-threshold",
        type=float,
        default=None,
        help="Also list pages with lexical_quality below this (default=veto threshold)",
    )
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir) if args.out_dir else settings.ir_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    suspect_thr = (
        args.suspect_threshold
        if args.suspect_threshold is not None
        else settings.lexical_veto_threshold
    )

    pdfs = iter_pdfs(Path(args.path))
    if not pdfs:
        log.error("no_pdfs_found", path=args.path)
        return 1

    candidates: list[dict] = []
    doc_summaries: list[dict] = []

    for pdf in pdfs:
        import fitz

        with fitz.open(pdf) as doc:
            total = doc.page_count
        pages = None if args.full else sample_pages(total, args.pages_per_doc)
        log.info("scan_doc", path=str(pdf), pages=pages or "all", page_count=total)

        analysis = analyze_pdf(
            pdf,
            pages=pages,
            tlq_threshold=settings.tlq_threshold,
            enable_visual=settings.tlq_enable_visual_agreement and not args.no_visual,
            lexical_veto_threshold=settings.lexical_veto_threshold,
            lexical_min_tokens=settings.lexical_min_tokens,
        )

        veto_pages = []
        suspect_pages = []
        for p in analysis.pages:
            lex = p.tlq.components.lexical_quality
            row = {
                "doc_id": analysis.doc_id,
                "source_path": analysis.source_path,
                "page": p.page,
                "page_count": analysis.page_count,
                "has_text_layer": p.tlq.has_text_layer,
                "char_count": p.tlq.char_count,
                "tlq": round(p.tlq.score, 4),
                "lexical_quality": round(lex, 4) if lex is not None else None,
                "garbage_score": p.tlq.components.garbage_score,
                "dict_hit": p.tlq.components.dict_hit,
                "typo_ratio": p.tlq.components.typo_ratio,
                "oov_hard": p.tlq.components.oov_hard,
                "garbage_veto": p.tlq.garbage_veto,
                "route": p.tlq.route.value,
                "lang_hint": p.tlq.lang_hint,
                "sample_bad_tokens": "|".join(p.tlq.sample_bad_tokens[:8]),
                "priority": None,
            }
            is_candidate = False
            if p.tlq.has_text_layer and p.tlq.char_count >= 40:
                if p.tlq.garbage_veto:
                    row["priority"] = "veto"
                    veto_pages.append(p.page)
                    is_candidate = True
                elif lex is not None and lex < suspect_thr:
                    row["priority"] = "suspect"
                    suspect_pages.append(p.page)
                    is_candidate = True
            if is_candidate:
                candidates.append(row)

        doc_summaries.append(
            {
                "doc_id": analysis.doc_id,
                "source_path": analysis.source_path,
                "page_count": analysis.page_count,
                "pages_scanned": analysis.summary.get("pages_analyzed"),
                "garbage_veto_pages": veto_pages,
                "suspect_pages": suspect_pages,
                "summary": analysis.summary,
            }
        )
        log.info(
            "scan_doc_done",
            doc_id=analysis.doc_id,
            veto=len(veto_pages),
            suspect=len(suspect_pages),
        )

    # Sort: veto first, then lowest lexical quality
    def sort_key(r: dict):
        pri = 0 if r["priority"] == "veto" else 1
        lq = r["lexical_quality"] if r["lexical_quality"] is not None else 1.0
        return (pri, lq, r["doc_id"], r["page"])

    candidates.sort(key=sort_key)

    report = {
        "docs_scanned": len(doc_summaries),
        "candidate_pages": len(candidates),
        "veto_pages": sum(1 for c in candidates if c["priority"] == "veto"),
        "suspect_pages": sum(1 for c in candidates if c["priority"] == "suspect"),
        "lexical_veto_threshold": settings.lexical_veto_threshold,
        "suspect_threshold": suspect_thr,
        "docs": doc_summaries,
        "candidates": candidates,
    }

    json_path = out_dir / "garbage_text_layer_report.json"
    csv_path = out_dir / "garbage_text_layer_candidates.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "priority",
        "doc_id",
        "page",
        "page_count",
        "tlq",
        "lexical_quality",
        "garbage_score",
        "dict_hit",
        "typo_ratio",
        "oov_hard",
        "route",
        "lang_hint",
        "char_count",
        "sample_bad_tokens",
        "source_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in candidates:
            w.writerow(row)

    print(f"\nDocs scanned: {len(doc_summaries)}")
    print(f"Candidate pages: {len(candidates)} (veto={report['veto_pages']}, suspect={report['suspect_pages']})")
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    print("\nTop candidates for proofreading:")
    for row in candidates[:25]:
        print(
            f"  [{row['priority']:7}] {row['doc_id'][:40]:40} p{row['page']:4}  "
            f"TLQ={row['tlq']:.2f} lex={row['lexical_quality']}  "
            f"bad={row['sample_bad_tokens'][:60]}"
        )
    if len(candidates) > 25:
        print(f"  ... and {len(candidates) - 25} more (see CSV)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
