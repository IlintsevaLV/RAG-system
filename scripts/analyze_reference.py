"""Analyze data/reference etalons vs garbage-scan CSV and local PDF text layers."""

from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ingestion.source_analysis import analyze_pdf, _extract_spans
from ingestion.text_layer_quality import lexical_quality
import fitz


def parse_ref(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    is_draft = "СТАТУС: ЗАГОТОВКА" in raw
    page = None
    m = re.search(r"_p(\d+)\.txt$", path.name, re.I)
    if m:
        page = int(m.group(1))
    # also from header
    for line in raw.splitlines()[:15]:
        mm = re.search(r"страница\s+(\d+)", line, re.I)
        if mm:
            page = int(mm.group(1))
        if "движком:" in line:
            engine = line.split("движком:")[-1].strip()
        else:
            engine = None

    engine = None
    for line in raw.splitlines()[:20]:
        if "движком:" in line:
            engine = line.split("движком:")[-1].strip()

    text_lines, formula_lines = [], []
    for line in raw.splitlines():
        if line.startswith("#"):
            continue
        if line.startswith("~"):
            formula_lines.append(line[1:].strip())
        else:
            text_lines.append(line)
    text = "\n".join(text_lines).strip()
    cyr = sum(1 for c in text if "\u0400" <= c <= "\u04FF")
    lat = sum(1 for c in text if c.isalpha() and c.isascii())
    return {
        "name": path.name,
        "path": path,
        "page": page,
        "is_draft": is_draft,
        "engine": engine,
        "chars": len(text),
        "lines": len([x for x in text_lines if x.strip()]),
        "formulas": len(formula_lines),
        "cyr": cyr,
        "lat": lat,
        "lang": "ru" if cyr > lat else ("en" if lat > cyr else "mixed/empty"),
        "text": text,
        "preview": re.sub(r"\s+", " ", text)[:140],
    }


def stem_from_ref_name(name: str) -> str:
    # DOC_pNNN.txt -> DOC
    return re.sub(r"_p\d+\.txt$", "", name, flags=re.I)


def find_pdf(stem: str) -> Path | None:
    candidates = []
    for folder in [ROOT / "data" / "raw", ROOT / "Нормативка", ROOT / "вычитка"]:
        if not folder.exists():
            continue
        for p in folder.rglob("*.pdf"):
            candidates.append(p)
        for p in folder.rglob("*.PDF"):
            candidates.append(p)
    # normalize compare
    stem_n = re.sub(r"\.pdf$", "", stem, flags=re.I)
    stem_n = stem_n.replace(".pdf", "")
    for p in candidates:
        if p.stem == stem_n or stem_n in p.stem or p.stem in stem_n:
            return p
        # AS file named with .pdf in stem
        if stem_n.replace(".pdf", "") == p.stem:
            return p
    # fuzzy: significant token overlap
    best = None
    best_score = 0
    tokens = set(re.findall(r"[A-Za-zА-Яа-яЁё0-9]{4,}", stem_n.lower()))
    for p in candidates:
        pt = set(re.findall(r"[A-Za-zА-Яа-яЁё0-9]{4,}", p.stem.lower()))
        score = len(tokens & pt)
        if score > best_score:
            best_score = score
            best = p
    return best if best_score >= 2 else None


def cer(ref: str, hyp: str) -> float:
    a = re.sub(r"\s+", " ", ref).strip()
    b = re.sub(r"\s+", " ", hyp).strip()
    if not a:
        return 1.0 if b else 0.0
    # limit size
    a, b = a[:12000], b[:12000]
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


def main() -> None:
    refs = [parse_ref(p) for p in sorted((ROOT / "data" / "reference").glob("*.txt"))]
    print("=" * 70)
    print(f"REFERENCE FILES: {len(refs)}")
    print("=" * 70)

    # CSV
    csv_rows = []
    csv_path = ROOT / "data" / "ir" / "garbage_text_layer_candidates.csv"
    if csv_path.exists():
        with csv_path.open(encoding="utf-8") as f:
            csv_rows = list(csv.DictReader(f))
    print(f"CSV candidates: {len(csv_rows)}")

    # Match CSV to refs by page + doc fragment
    matched = 0
    for r in refs:
        stem = stem_from_ref_name(r["name"])
        print("-" * 70)
        print(f"FILE: {r['name']}")
        print(
            f"  page={r['page']} draft={r['is_draft']} engine={r['engine']} "
            f"lang={r['lang']} chars={r['chars']} lines={r['lines']} formulas={r['formulas']}"
        )
        print(f"  preview: {r['preview']!r}")

        # lexical on REFERENCE (should be good if proofread)
        lex = lexical_quality(r["text"])
        print(
            f"  REF lexical_quality={lex.lexical_quality:.3f} oov={lex.oov_hard:.3f} "
            f"veto={lex.garbage_veto} bad_sample={lex.sample_bad_tokens[:5]}"
        )

        # CSV match
        hits = []
        for row in csv_rows:
            try:
                if int(row["page"]) != r["page"]:
                    continue
            except Exception:
                continue
            doc = row.get("doc_id") or ""
            if any(tok for tok in re.findall(r"[A-Za-zА-Яа-яЁё0-9]{5,}", stem) if tok.lower() in doc.lower() or tok in doc):
                hits.append(row)
            elif stem[:20] in doc or doc[:20] in stem:
                hits.append(row)
        if hits:
            matched += 1
            row = hits[0]
            print(
                f"  CSV: priority={row.get('priority')} tlq={row.get('tlq')} "
                f"lex={row.get('lexical_quality')} route={row.get('route')} "
                f"bad={row.get('sample_bad_tokens','')[:80]}"
            )
        else:
            print("  CSV: NO MATCH")

        # PDF layer compare if available
        pdf = find_pdf(stem)
        if pdf and r["page"]:
            try:
                analysis = analyze_pdf(pdf, pages=[r["page"]], enable_visual=False)
                p = analysis.pages[0]
                layer = p.extracted_text if p.tlq.has_text_layer else ""
                # always get full layer for CER even if routed to ocr
                doc = fitz.open(pdf)
                layer_full, spans = _extract_spans(doc[r["page"] - 1])
                doc.close()
                c = cer(r["text"], layer_full) if layer_full.strip() else 1.0
                print(
                    f"  PDF: {pdf.name} | TLQ={p.tlq.score:.3f} lex={p.tlq.components.lexical_quality} "
                    f"veto={p.tlq.garbage_veto} route={p.tlq.route.value} "
                    f"layer_chars={len(layer_full)} CER(layer vs REF)~{c:.3f}"
                )
            except Exception as exc:
                print(f"  PDF analyze error: {exc}")
        else:
            print(f"  PDF: not found locally for stem={stem[:60]!r}")

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    drafts = sum(1 for r in refs if r["is_draft"])
    emptyish = sum(1 for r in refs if r["chars"] < 200)
    with_formulas = sum(1 for r in refs if r["formulas"] > 0)
    by_lang = Counter(r["lang"] for r in refs)
    print(f"refs={len(refs)} csv={len(csv_rows)} matched_csv≈{matched}")
    print(f"still_DRAFT_flag={drafts} short(<200chars)={emptyish} with_formula_marks={with_formulas}")
    print(f"languages={dict(by_lang)}")

    # CSV pages without ref
    ref_pages = {(stem_from_ref_name(r["name"]), r["page"]) for r in refs}
    print("\nCSV rows without obvious local ref file:")
    for row in csv_rows:
        doc = row.get("doc_id", "")
        page = int(row["page"])
        found = False
        for r in refs:
            if r["page"] != page:
                continue
            stem = stem_from_ref_name(r["name"])
            if any(tok for tok in re.findall(r"[A-Za-zА-Яа-яЁё0-9]{5,}", stem) if tok.lower() in doc.lower() or tok in doc):
                found = True
                break
            if stem[:25] in doc or doc[:25] in stem:
                found = True
                break
        if not found:
            print(f"  - {doc} p{page} priority={row.get('priority')} lex={row.get('lexical_quality')}")


if __name__ == "__main__":
    main()
