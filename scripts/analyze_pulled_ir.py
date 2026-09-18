"""Summarize pulled *_pages.json and regions JSON for quality review."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def main() -> None:
    pages_dir = Path("data/ir/pages")
    regions_dir = Path("data/ir/regions")
    out: dict = {"pages_files": 0, "page_rows": 0}

    class_c = Counter()
    status_c = Counter()
    source_c = Counter()
    notes_hits = Counter()
    per_doc = []
    interesting = []

    for jf in sorted(pages_dir.glob("*_pages.json")):
        data = json.loads(jf.read_text(encoding="utf-8"))
        out["pages_files"] += 1
        summary = data.get("summary") or {}
        pages = data.get("pages") or []
        out["page_rows"] += len(pages)
        doc_sources = Counter()
        for p in pages:
            class_c[p.get("page_class", "?")] += 1
            status_c[p.get("status", "?")] += 1
            src = p.get("text_source") or "(empty)"
            source_c[src] += 1
            doc_sources[src] += 1
            for n in p.get("notes") or []:
                key = n.split(":")[0].split("=")[0]
                if any(
                    k in n
                    for k in (
                        "vlm",
                        "ocr_fallback",
                        "escalate",
                        "switched_to_ocr",
                        "garbage",
                        "unreadable",
                    )
                ):
                    notes_hits[n[:80]] += 1
            if p.get("page_class") in ("C", "D") or "vlm" in src or src == "ocr_fallback":
                interesting.append(
                    {
                        "doc": data.get("doc_id", jf.stem)[:50],
                        "page": p.get("page"),
                        "class": p.get("page_class"),
                        "status": p.get("status"),
                        "src": src,
                        "chars": len(p.get("text") or ""),
                        "notes": (p.get("notes") or [])[:4],
                    }
                )
        per_doc.append(
            {
                "doc": (data.get("doc_id") or jf.stem)[:60],
                "summary": summary,
                "sources": dict(doc_sources),
                "n_pages": len(pages),
            }
        )

    regions = []
    for jf in sorted(regions_dir.glob("*_regions.json")):
        # skip local-only ignored file if empty-ish; still include all present
        data = json.loads(jf.read_text(encoding="utf-8"))
        for pg in data.get("pages") or []:
            s = pg.get("summary") or {}
            blocks = pg.get("blocks") or []
            methods = Counter()
            previews = []
            for b in blocks:
                content = b.get("content") or {}
                prov = b.get("provenance") or {}
                method = (
                    content.get("method")
                    or prov.get("method")
                    or (b.get("quality") or {}).get("method")
                    or "?"
                )
                # try nested
                if method == "?":
                    method = (b.get("notes") or [None])[0] or "?"
                methods[b.get("type"), method] += 1
                if b.get("type") == "table":
                    md = (content.get("markdown") or "")[:120]
                    previews.append(
                        {
                            "type": "table",
                            "id": b.get("region_id"),
                            "method": content.get("method")
                            or (b.get("provenance") or {}).get("extractor"),
                            "conf": (b.get("quality") or {}).get("confidence")
                            or content.get("confidence"),
                            "preview": md.replace("\n", " "),
                            "notes": (b.get("notes") or content.get("notes") or [])[:5],
                        }
                    )
                elif b.get("type") == "formula":
                    previews.append(
                        {
                            "type": "formula",
                            "id": b.get("region_id"),
                            "latex": (content.get("latex") or "")[:100],
                            "method": content.get("method"),
                            "conf": (b.get("quality") or {}).get("confidence"),
                            "notes": (b.get("notes") or [])[:5],
                        }
                    )
                elif b.get("type") == "figure":
                    previews.append(
                        {
                            "type": "figure",
                            "id": b.get("region_id"),
                            "caption": (content.get("caption") or "")[:80],
                            "method": content.get("method"),
                        }
                    )
            # flatten methods keys
            methods_flat = {f"{t}:{m}": c for (t, m), c in methods.items()}
            regions.append(
                {
                    "file": jf.name,
                    "doc": data.get("doc_id"),
                    "page": s.get("page"),
                    "summary": s,
                    "methods": methods_flat,
                    "previews": previews[:12],
                    "raw_block0_keys": list(blocks[0].keys()) if blocks else [],
                    "raw_block0": blocks[0] if blocks else None,
                }
            )

    report = {
        "pages_files": out["pages_files"],
        "page_rows": out["page_rows"],
        "class_counts": dict(class_c),
        "status_counts": dict(status_c),
        "source_counts": dict(source_c),
        "note_hits_top": notes_hits.most_common(25),
        "interesting_sample": interesting[:40],
        "per_doc": per_doc,
        "regions": [
            {k: v for k, v in r.items() if k != "raw_block0"} | {"raw_block0": r["raw_block0"]}
            for r in regions
        ],
    }
    path = Path("data/ir/pulled_ir_analysis.json")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {path}")
    print("pages_files", out["pages_files"], "page_rows", out["page_rows"])
    print("classes", dict(class_c))
    print("status", dict(status_c))
    print("sources", dict(source_c))
    print("regions files/pages:", len(regions))
    for r in regions:
        print(
            f"  {r['file']} p{r['page']} summary={r['summary']} methods={r['methods']}"
        )


if __name__ == "__main__":
    main()
