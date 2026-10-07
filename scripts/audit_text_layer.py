"""Page-text diagnosis: XML, Greek lookalikes, broken words, control characters.

Prints one row per document: share of pages with each problem, for every page
and for pages the text index would admit (class A/B, status ok).

  python -X utf8 -m scripts.audit_text_layer
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import get_settings
from core.text_norm import normalize_greek_lookalikes
from retrieval.engine import _low_confidence, is_toc_page

_XML_TAG = re.compile(r"</?[A-Za-z][\w:.-]*\b[^>]*>")
_SPLIT_HYPHEN = re.compile(r"[A-Za-zА-Яа-яЁё]-\n[A-Za-zА-Яа-яЁё]")
_REPLACEMENT = "\ufffd"


def _problems(text: str) -> dict[str, bool]:
    return {
        "xml": bool(_XML_TAG.search(text)),
        "greek": normalize_greek_lookalikes(text) != text,
        "control": any(
            unicodedata.category(ch) == "Cc" and ch not in "\n\r\t" for ch in text
        ),
        "broken": _REPLACEMENT in text or bool(_SPLIT_HYPHEN.search(text)),
    }


def _admitted(page: dict) -> bool:
    if page.get("page_class") not in ("A", "B"):
        return False
    if page.get("status") != "ok":
        return False
    if _low_confidence(page):
        return False
    text = (page.get("text") or "").strip()
    if len(text) < 40 or is_toc_page(text):
        return False
    return True


def audit(pages_dir: Path) -> list[dict]:
    rows: list[dict] = []
    keys = ("xml", "greek", "control", "broken")
    for path in sorted(pages_dir.glob("*_pages.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        pages = data.get("pages") if isinstance(data, dict) else data
        if not isinstance(pages, list):
            continue
        doc_id = data.get("doc_id") if isinstance(data, dict) else path.stem
        counts = {key: 0 for key in keys}
        admitted_n = 0
        admitted_counts = {key: 0 for key in keys}
        nonempty = 0
        for page in pages:
            text = page.get("text") or ""
            if not text.strip():
                continue
            nonempty += 1
            flags = _problems(text)
            for key in keys:
                counts[key] += int(flags[key])
            if _admitted(page):
                admitted_n += 1
                for key in keys:
                    admitted_counts[key] += int(flags[key])
        rows.append(
            {
                "doc_id": doc_id or path.stem,
                "pages": nonempty,
                "admitted": admitted_n,
                **{key: counts[key] for key in keys},
                **{f"adm_{key}": admitted_counts[key] for key in keys},
            }
        )
    return rows


def _pct(num: int, den: int) -> str:
    if not den:
        return "—"
    return f"{100.0 * num / den:.0f}%"


def main() -> int:
    pages_dir = get_settings().ir_dir / "pages"
    rows = audit(pages_dir)
    keys = ("xml", "greek", "control", "broken")
    print(f"Документов: {len(rows)}")
    header = "документ | страниц | A/B | xml | greek | control | обрыв | xml A/B | greek A/B | control A/B | обрыв A/B"
    print(header)
    dirty = []
    for row in rows:
        line = " | ".join(
            [
                str(row["doc_id"]),
                str(row["pages"]),
                str(row["admitted"]),
                _pct(row["xml"], row["pages"]),
                _pct(row["greek"], row["pages"]),
                _pct(row["control"], row["pages"]),
                _pct(row["broken"], row["pages"]),
                _pct(row["adm_xml"], row["admitted"]),
                _pct(row["adm_greek"], row["admitted"]),
                _pct(row["adm_control"], row["admitted"]),
                _pct(row["adm_broken"], row["admitted"]),
            ]
        )
        print(line)
        if not row["admitted"]:
            continue
        xml_share = row["adm_xml"] / row["admitted"]
        broken_share = row["adm_broken"] / row["admitted"]
        if xml_share >= 0.05 or broken_share >= 0.50:
            dirty.append((max(xml_share, broken_share), row))
    print("--- смотреть в первую очередь: XML ≥ 5% A/B или обрыв строки ≥ 50% A/B ---")
    print("Греческие двойники в сыром слое нормализуются при нарезке и сами по себе переснятие не требуют.")
    for _share, row in sorted(dirty, reverse=True):
        print(
            f"{row['doc_id']}  A/B={row['admitted']}  "
            f"xml={_pct(row['adm_xml'], row['admitted'])}  "
            f"обрыв={_pct(row['adm_broken'], row['admitted'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
