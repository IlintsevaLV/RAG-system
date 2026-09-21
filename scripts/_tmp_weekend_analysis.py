from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

out = Path("data/ir/weekend_local_analysis.json")

# log tail/head
log_path = Path("data/ir/logs/weekend_20260918_163909.log")
log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
log_lines = log_text.splitlines()

pages_root = Path("data/ir/pages")
txt_counts = {}
for d in pages_root.iterdir():
    if d.is_dir():
        txt_counts[d.name] = len(list(d.glob("page_*.txt")))

json_info = []
all_src = Counter()
all_cls = Counter()
all_st = Counter()
note_flags = Counter()
for jf in sorted(pages_root.glob("*_pages.json")):
    data = json.loads(jf.read_text(encoding="utf-8"))
    pages = data.get("pages") or []
    src = Counter(p.get("text_source") or "(empty)" for p in pages)
    cls = Counter(p.get("page_class") for p in pages)
    st = Counter(p.get("status") for p in pages)
    all_src.update(src)
    all_cls.update(cls)
    all_st.update(st)
    for p in pages:
        for n in p.get("notes") or []:
            low = n.lower()
            if "vlm_disabled" in low:
                note_flags["vlm_disabled_or_unavailable"] += 1
            elif "vlm_extract_ok" in low:
                note_flags["vlm_extract_ok"] += 1
            elif "ocr_fallback_used" in low:
                note_flags["ocr_fallback_used"] += 1
            elif "escalate_b_to_c" in low:
                note_flags["escalate_B_to_C"] += 1
            elif low.startswith("classified="):
                note_flags[n] += 1
    json_info.append(
        {
            "doc": data.get("doc_id") or jf.stem,
            "json_pages": len(pages),
            "page_nums_sample": [p.get("page") for p in pages[:5]]
            + ([p.get("page") for p in pages[-2:]] if len(pages) > 5 else []),
            "summary": data.get("summary"),
            "sources": dict(src),
            "classes": dict(cls),
            "txt_count": txt_counts.get(data.get("doc_id") or jf.stem.replace("_pages", ""), None),
        }
    )

# match txt_count by stem
for item in json_info:
    doc = item["doc"]
    item["txt_count"] = txt_counts.get(doc, 0)

regions = []
for jf in Path("data/ir/regions").glob("*_regions.json"):
    data = json.loads(jf.read_text(encoding="utf-8"))
    for pg in data.get("pages") or []:
        s = pg.get("summary") or {}
        methods = Counter()
        for b in pg.get("blocks") or []:
            m = (b.get("provenance") or {}).get("method") or (b.get("content") or {}).get("method") or "?"
            methods[f"{b.get('type')}:{m}"] += 1
        regions.append({"file": jf.name, "summary": s, "methods": dict(methods)})

backup = Path("data/ir/pages_backup_20260918_163909")
report = {
    "txt_docs": len(txt_counts),
    "txt_pages_total": sum(txt_counts.values()),
    "txt_per_doc_sorted": sorted(
        [{"doc": k, "n": v} for k, v in txt_counts.items()], key=lambda x: -x["n"]
    ),
    "json_files": len(json_info),
    "json_page_rows": sum(i["json_pages"] for i in json_info),
    "global_classes": dict(all_cls),
    "global_status": dict(all_st),
    "global_sources": dict(all_src),
    "note_flags": dict(note_flags),
    "json_per_doc": sorted(json_info, key=lambda x: -x["json_pages"]),
    "regions": regions,
    "backup_json_count": len(list(backup.glob("*_pages.json"))) if backup.exists() else 0,
    "log_exists": log_path.exists(),
    "log_lines": len(log_lines),
    "log_head": log_lines[:40],
    "log_tail": log_lines[-60:],
    "garbage_csv_lines": (
        len(Path("data/ir/garbage_text_layer_candidates.csv").read_text(encoding="utf-8", errors="replace").splitlines())
        if Path("data/ir/garbage_text_layer_candidates.csv").exists()
        else 0
    ),
}
out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print("wrote", out)
print("txt_total", report["txt_pages_total"], "json_rows", report["json_page_rows"])
print("sources", report["global_sources"])
print("classes", report["global_classes"])
print("notes", report["note_flags"])
print("top txt", report["txt_per_doc_sorted"][:5])
print("top json", [(i["doc"][:40], i["json_pages"], i["txt_count"]) for i in report["json_per_doc"][:5]])
print("log_lines", report["log_lines"])
