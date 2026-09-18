"""Compare data/ir/pages extracts vs data/reference; write quality report."""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path


def cer(ref: str, hyp: str) -> float:
    a = re.sub(r"\s+", " ", ref).strip()
    b = re.sub(r"\s+", " ", hyp).strip()
    if not a:
        return 1.0 if b else 0.0
    a, b = a[:10000], b[:10000]
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        ca = a[i - 1]
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if ca == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[m] / n


def parse_ref(path: Path) -> str:
    lines = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.startswith("#") or ln.startswith("~"):
            continue
        lines.append(ln)
    return "\n".join(lines).strip()


def match_doc(stem: str, docs: dict[str, dict[int, str]], pno: int):
    tokens = re.findall(r"[A-Za-zА-Яа-яЁё0-9]{4,}", stem)
    best = None
    for name, pages in docs.items():
        if pno not in pages:
            continue
        score = sum(1 for tok in tokens if tok.lower() in name.lower())
        if stem[:30] in name or name[:30] in stem:
            score += 2
        if best is None or score > best[2]:
            best = (name, pages[pno], score)
    if best and best[2] >= 1:
        return best
    return None


def main() -> None:
    root = Path("data/ir/pages")
    ref_dir = Path("data/reference")
    docs: dict[str, dict[int, str]] = {}
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        docs[d.name] = {
            int(f.stem.split("_")[1]): f.read_text(encoding="utf-8", errors="replace")
            for f in d.glob("page_*.txt")
        }

    all_lens = [len(t) for pages in docs.values() for t in pages.values()]
    empties = [
        (n, p) for n, pages in docs.items() for p, t in pages.items() if not t.strip()
    ]
    short = sum(
        1
        for pages in docs.values()
        for t in pages.values()
        if 0 < len(t.strip()) < 80
    )
    markers = [
        "airframc",
        "plicaled",
        "subsysicm",
        "helicopiers",
        "arrngcment",
        "cxposcd",
        "rcquircmcnts",
        "maintcnancc",
        "navigalion",
    ]
    garbage_hits = []
    for n, pages in docs.items():
        for p, t in pages.items():
            low = t.lower()
            hits = sum(1 for w in markers if w in low)
            if hits:
                garbage_hits.append((n, p, hits))

    # Cyrillic density for Russian-looking docs
    cyr_docs = []
    for n, pages in docs.items():
        sample = " ".join(list(pages.values())[:3])
        if not sample.strip():
            continue
        cyr = sum(1 for ch in sample if "А" <= ch <= "я" or ch in "ёЁ")
        lat = sum(1 for ch in sample if "A" <= ch <= "z")
        cyr_docs.append(
            {
                "doc": n[:70],
                "cyr_ratio": round(cyr / max(cyr + lat, 1), 3),
                "chars": len(sample),
            }
        )

    ref_results = []
    for rf in sorted(ref_dir.glob("*.txt")):
        m = re.search(r"_p(\d+)\.txt$", rf.name)
        if not m:
            continue
        pno = int(m.group(1))
        stem = re.sub(r"_p\d+\.txt$", "", rf.name)
        matched = match_doc(stem, docs, pno)
        gt = parse_ref(rf)
        if not matched:
            ref_results.append({"ref": rf.name, "page": pno, "matched": False})
            continue
        name, hyp, score = matched
        c = cer(gt, hyp)
        ref_results.append(
            {
                "ref": rf.name,
                "page": pno,
                "matched": True,
                "match_score": score,
                "doc": name[:80],
                "cer": round(c, 3),
                "gt_chars": len(gt),
                "hyp_chars": len(hyp),
                "hyp_empty": not hyp.strip(),
                "preview_hyp": re.sub(r"\s+", " ", hyp)[:160],
                "preview_gt": re.sub(r"\s+", " ", gt)[:160],
            }
        )

    # Per-doc summary
    per_doc = []
    for n, pages in sorted(docs.items(), key=lambda x: -len(x[1])):
        lens = [len(t) for t in pages.values()]
        empty_n = sum(1 for t in pages.values() if not t.strip())
        per_doc.append(
            {
                "doc": n[:80],
                "pages": len(pages),
                "empty": empty_n,
                "median_chars": int(statistics.median(lens)) if lens else 0,
                "min_chars": min(lens) if lens else 0,
                "max_chars": max(lens) if lens else 0,
            }
        )

    out = {
        "docs": len(docs),
        "pages": sum(len(p) for p in docs.values()),
        "empty_count": len(empties),
        "empty_pages": [{"doc": n[:70], "page": p} for n, p in empties],
        "short_count": short,
        "median_chars": int(statistics.median(all_lens)) if all_lens else 0,
        "mean_chars": int(statistics.mean(all_lens)) if all_lens else 0,
        "garbage_marker_pages": len(garbage_hits),
        "garbage_examples": [
            {"doc": n[:70], "page": p, "hits": h} for n, p, h in garbage_hits[:15]
        ],
        "cyrillic_sample": sorted(cyr_docs, key=lambda x: x["cyr_ratio"])[:8]
        + sorted(cyr_docs, key=lambda x: -x["cyr_ratio"])[:8],
        "per_doc": per_doc,
        "ref_eval": ref_results,
    }
    report_path = Path("data/ir/extract_quality_report.json")
    report_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {report_path}")
    print(f"docs={out['docs']} pages={out['pages']} empty={out['empty_count']} short={out['short_count']}")
    print(f"median_chars={out['median_chars']} mean_chars={out['mean_chars']}")
    print(f"garbage_marker_pages={out['garbage_marker_pages']}")
    print("REF CER:")
    for r in ref_results:
        if r.get("matched"):
            print(
                f"  p{r['page']:4d} CER={r['cer']:.3f} hyp={r['hyp_chars']:5d} "
                f"empty={r['hyp_empty']} | {r['ref'][:60]}"
            )
        else:
            print(f"  p{r['page']:4d} NO_MATCH | {r['ref'][:60]}")


if __name__ == "__main__":
    main()
