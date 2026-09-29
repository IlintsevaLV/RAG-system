"""Отбор страниц-кандидатов для golden формул.

Проходит по data/ir/regions/*.json, собирает страницы с формулами,
ранжирует по количеству и разнообразию статусов, выдаёт список.
"""
import json
from pathlib import Path
from collections import defaultdict

REGIONS = Path("data/ir/regions")

# сколько страниц на каждый PDF хотим
PER_DOC_LIMIT = 8

candidates = []  # (doc_id, page, n_formulas, statuses_dict, path)

for jf in sorted(REGIONS.glob("*.json")):
    with jf.open(encoding="utf-8") as f:
        data = json.load(f)
    doc_id = data.get("doc_id", jf.stem)
    for p in data.get("pages", []):
        formulas = [b for b in p.get("blocks", []) if b.get("type") == "formula"]
        if not formulas:
            continue
        n = len(formulas)
        statuses = defaultdict(int)
        for b in formulas:
            statuses[b.get("content", {}).get("status", "?")] += 1
        candidates.append((doc_id, p["page"], n, dict(statuses), jf.name))

# группируем по doc_id
by_doc = defaultdict(list)
for c in candidates:
    by_doc[c[0]].append(c)

# сортируем внутри каждого PDF: сначала где много формул, потом где разные статусы
print("=== кандидаты по PDF ===")
print()
selected = []
for doc_id in sorted(by_doc):
    items = by_doc[doc_id]
    # ранжирование: больше формул + больше разных статусов
    items.sort(key=lambda x: (-x[2], -len(x[3])))
    print(f"--- {doc_id} ({len(items)} страниц с формулами) ---")
    for doc, page, n, statuses, jf_name in items[:PER_DOC_LIMIT]:
        st = ", ".join(f"{k}={v}" for k, v in sorted(statuses.items()))
        print(f"  p{page:4d}  n_formulas={n:3d}  {st}")
        selected.append((doc, page, n, statuses))
    print()

print()
print(f"=== итого отобрано: {len(selected)} страниц ===")
print()
print("Формат для копирования в разметку (doc_id page):")
for doc, page, n, st in selected:
    print(f"  {doc} p{page}")