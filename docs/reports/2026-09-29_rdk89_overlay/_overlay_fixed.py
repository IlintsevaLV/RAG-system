import fitz, json

g = json.load(open('data/ir/gold/formulas_v1.json', encoding='utf-8'))
gold_by_page = {}
for p in g['pages']:
    if '89' in p.get('doc_id', ''):
        gold_by_page[p['page']] = p.get('formulas', [])

r = json.load(open('data/ir/regions/РДК_молниезащита 89г_regions.json', encoding='utf-8'))
reg_by_page = {}
for p in r['pages']:
    reg_by_page[p['page']] = [d for d in p.get('detections', []) if d.get('type') == 'formula']

doc = fitz.open('data/raw/РДК_молниезащита 89г.pdf')

for pg in [262, 274]:
    page = doc[pg-1]

    # Golden — красным. Без scale, PyMuPDF в pt.
    for i, f in enumerate(gold_by_page.get(pg, [])):
        b = f['bbox']
        rect = fitz.Rect(b['x1'], b['y1'], b['x2'], b['y2'])
        page.draw_rect(rect, color=(1,0,0), width=1.5)
        page.insert_text(fitz.Point(rect.x0+2, rect.y0+10), f'G{i}', fontsize=11, color=(1,0,0))

    # Detector — синим. Без scale.
    for i, d in enumerate(reg_by_page.get(pg, [])):
        b = d.get('bbox') or d.get('box') or {}
        if all(k in b for k in ['x1','y1','x2','y2']):
            rect = fitz.Rect(b['x1'], b['y1'], b['x2'], b['y2'])
            page.draw_rect(rect, color=(0,0,1), width=1.5)
            page.insert_text(fitz.Point(rect.x0+2, rect.y0+10), f'D{i}', fontsize=11, color=(0,0,1))

    pix = page.get_pixmap(dpi=150)
    pix.save(f'_rdk89_p{pg}_fixed.png')
    print(f'saved _rdk89_p{pg}_fixed.png')
