import fitz, json

# Golden
g = json.load(open('data/ir/gold/formulas_v1.json', encoding='utf-8'))
gold_by_page = {}
for p in g['pages']:
    if '89' in p.get('doc_id', ''):
        gold_by_page[p['page']] = p.get('formulas', [])

# Regions
r = json.load(open('data/ir/regions/РДК_молниезащита 89г_regions.json', encoding='utf-8'))
reg_by_page = {}
for p in r['pages']:
    reg_by_page[p['page']] = [d for d in p.get('detections', []) if d.get('type') == 'formula']

doc = fitz.open('data/raw/РДК_молниезащита 89г.pdf')

for pg in [262, 274]:
    page = doc[pg-1]
    scale = 150/72.0  # PDF points -> pixels at 150 dpi

    # Рисуем golden — красным
    for f in gold_by_page.get(pg, []):
        b = f.get('bbox', {})
        rect = fitz.Rect(b['x1']*scale, b['y1']*scale, b['x2']*scale, b['y2']*scale)
        page.draw_rect(rect, color=(1,0,0), width=2)

    # Рисуем regions — синим
    for d in reg_by_page.get(pg, []):
        b = d.get('bbox') or d.get('box') or {}
        if all(k in b for k in ['x1','y1','x2','y2']):
            rect = fitz.Rect(b['x1']*scale, b['y1']*scale, b['x2']*scale, b['y2']*scale)
            page.draw_rect(rect, color=(0,0,1), width=2)

    pix = page.get_pixmap(dpi=150)
    out = f'_rdk89_p{pg}_overlay.png'
    pix.save(out)
    print(f'saved {out}')
