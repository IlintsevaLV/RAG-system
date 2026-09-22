"""Показать все формулы из JSON с LaTeX и статусом."""
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
                if b.get("type") != "formula":
                    continue
                rid = b.get("region_id", "?")
                status = b.get("content", {}).get("status")
                method = b.get("provenance", {}).get("method", "?")
                latex = b.get("content", {}).get("latex", "")
                print(f"--- p{page_no} {rid} status={status} method={method}")
                print(f"    {latex[:300]}")
        print()


if __name__ == "__main__":
    main(sys.argv[1:])