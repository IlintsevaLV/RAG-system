import json
from pathlib import Path

docs = [
    "МР-21.002 Процедура квалификации комплектующих изделий",
    "МР-21.004 Процедура сертификации АТ",
    "НЛГ 33 Двигатели",
    "ПОЛ_ о проведении сертификации по п.п.603,613_2008",
    "РМ-178B оценка соответствия ПО бортовой аппаратуры",
    "РМ-254 оценка соответствия бортовой аппаратуцры требованиям КТ-254",
    "Руководство 23-29-М порядок ОС материалов (по 603 п. АП-29)",
    "Руководство 23-29.605 по МОС образцов ВС",
    "РЦ-АП-ВД 6.2 анализ безопасности",
    "РЦ-АП-ВД 6.3 пожарная безопасность",
    "РЦ-АП25 ТС МОС топливных систем самолетов",
    "ФАП-118 положение о порядке допуска к эксплуатации единичных экземпляров ВС",
]

print(f'{"doc_id":<55} {"pages":>6} {"A/B":>6} {"C/D":>6} {"size":>10}')
print("-" * 100)
for d in docs:
    f = Path(f"data/ir/pages/{d}_pages.json")
    if f.exists():
        data = json.load(open(f, encoding="utf-8-sig"))
        pages = data.get("pages", [])
        A = sum(1 for p in pages if p.get("page_class") == "A")
        B = sum(1 for p in pages if p.get("page_class") == "B")
        C = sum(1 for p in pages if p.get("page_class") == "C")
        D = sum(1 for p in pages if p.get("page_class") == "D")
        size = f.stat().st_size
        print(f"{d[:53]:<55} {len(pages):>6} {A+B:>6} {C+D:>6} {size:>10}")
    else:
        print(f"{d[:53]:<55} MISSING")
