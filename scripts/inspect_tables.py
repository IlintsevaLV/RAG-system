"""Показать все таблицы из JSON с Markdown-содержимым."""
import json
import sys
from pathlib import Path


def main(paths):
    for p in paths:
        f = Path(p)
        if not f.is_file():
            continue
        d = json.load(open(f, encoding="utf-8"))
        for page in d.get("pages", []):
            page_no = page.get("page")
            for b in page.get("blocks", []):
                if b.get("type") != "table":
                    continue
                rid = b.get("region_id", "?")
                status = b.get("content", {}).get("status")
                method = b.get("provenance", {}).get("method", "?")
                notes = b.get("quality", {}).get("notes", [])
                md = b.get("content", {}).get("markdown", "")
                print(f"--- p{page_no} {rid} status={status} method={method}")
                print(f"    notes={notes}")
                print(f"    markdown (first 800):")
                print("    " + "\n    ".join(md[:800].splitlines()))
        print()


if __name__ == "__main__":
    main(sys.argv[1:])