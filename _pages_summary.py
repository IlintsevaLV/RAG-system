import json, sys
from pathlib import Path

pages_dir = Path('data/ir/pages')
files = sorted(pages_dir.glob('*_pages.json'))

print(f'{"doc_id":<60} {"pages":>6} {"A":>5} {"B":>5} {"C":>5} {"D":>5} {"ok":>5} {"susp":>5} {"ocr":>5} {"vlm":>5}')
print('-' * 120)

total = {'pages': 0, 'A': 0, 'B': 0, 'C': 0, 'D': 0, 'ok': 0, 'susp': 0, 'ocr': 0, 'vlm': 0}

for f in files:
    try:
        d = json.load(open(f, encoding='utf-8-sig'))
        s = d.get('summary', {})
        doc_id = d.get('doc_id', f.stem)[:58]
        pages = s.get('pages', 0)
        A = s.get('class_A', 0)
        B = s.get('class_B', 0)
        C = s.get('class_C', 0)
        D = s.get('class_D', 0)
        ok = s.get('ok', 0)
        susp = s.get('suspicious', 0)
        ocr = s.get('ocr_fallback_used', 0)
        vlm = s.get('vlm_used', 0)
        print(f'{doc_id:<60} {pages:>6} {A:>5} {B:>5} {C:>5} {D:>5} {ok:>5} {susp:>5} {ocr:>5} {vlm:>5}')
        total['pages'] += pages; total['A'] += A; total['B'] += B; total['C'] += C; total['D'] += D
        total['ok'] += ok; total['susp'] += susp; total['ocr'] += ocr; total['vlm'] += vlm
    except Exception as e:
        print(f'ERROR {f.name}: {e}')

print('-' * 120)
print(f'{"TOTAL":<60} {total["pages"]:>6} {total["A"]:>5} {total["B"]:>5} {total["C"]:>5} {total["D"]:>5} {total["ok"]:>5} {total["susp"]:>5} {total["ocr"]:>5} {total["vlm"]:>5}')
print()
print(f'Класс A+B (годятся для RAG без OCR): {total["A"] + total["B"]}')
print(f'Класс C+D (требуют OCR/VLM):        {total["C"] + total["D"]}')
