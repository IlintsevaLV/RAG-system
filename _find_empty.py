import json
from pathlib import Path

print(f'{"file":<70} {"pages":>6}')
print('-' * 80)
empty = []
for f in sorted(Path('data/ir/pages').glob('*_pages.json')):
    d = json.load(open(f, encoding='utf-8-sig'))
    n = len(d.get('pages', []))
    if n == 0:
        empty.append(f)
        print(f'{f.name[:68]:<70} {n:>6}')

print()
print(f'Пустых JSON: {len(empty)}')
for f in empty:
    # Попробуем найти соответствующий PDF
    doc_id = json.load(open(f, encoding='utf-8-sig')).get('doc_id', f.stem)
    raw_pdfs = list(Path('data/raw').glob(f'*{doc_id[:30]}*')) if doc_id else []
    print(f'  {doc_id[:50]:<52} -> PDFs: {[p.name[:40] for p in raw_pdfs[:2]]}')
