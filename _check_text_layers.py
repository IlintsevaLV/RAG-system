import fitz
from pathlib import Path

raw = Path('data/raw')
print(f'{"file":<60} {"pages":>6} {"text_chars_avg":>14}')
print('-' * 90)
for pdf in sorted(raw.glob('*.pdf')):
    try:
        doc = fitz.open(pdf)
        n = doc.page_count
        sample = [0, n//4, n//2, 3*n//4, n-1] if n > 5 else list(range(n))
        total = sum(len(doc[i].get_text()) for i in sample if i < n)
        avg = total // max(len(sample), 1)
        flag = '<<< LAYER' if avg > 200 else '   scan' if avg < 50 else '   ?'
        print(f'{pdf.name[:58]:<60} {n:>6} {avg:>14}  {flag}')
    except Exception as e:
        print(f'{pdf.name[:58]:<60} ERROR: {e}')
