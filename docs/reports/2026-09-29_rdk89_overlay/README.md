# Overlay: golden vs detector для rdk89 p262/p274

Overlay-картинки для визуального сравнения:
- **Красные рамки (G0-G10)** — golden (`data/ir/gold/formulas_v1.json`)
- **Синие рамки (D0-D11)** — detector (`data/ir/regions/РДК_молниезащита 89г_regions.json`)

## Файлы

- `_rdk89_p262_fixed.png` — PDF index 262 = книжная стр. 246, «9.3.1 Complex Numbers»
- `_rdk89_p274_fixed.png` — PDF index 274 = книжная стр. 258, «9.7.5 Magnetic Induction»
- `_overlay_fixed.py` — скрипт генерации

## Важно: координатная система

PyMuPDF `page.draw_rect` работает в **pt** в системе координат страницы,
НЕ в пикселях pixmap. НЕ применять `scale = dpi/72` при рисовании —
pixmap сам масштабирует. Иначе bbox окажется вдвое дальше по x и y
(баг предыдущей версии overlay).

## Воспроизведение

Из корня репозитория:

    & $Py docs\reports\2026-09-29_rdk89_overlay\_overlay_fixed.py

## Что показывают картинки

**p274:** golden размечает 11 формул точно. Detector находит 9,
но bbox расширяются вправо через границу колонки (golden x2=260 pt,
detector x2=552 pt). Колоночный фикс `2c01c7c` не срабатывает.

**p262:** golden размечает 5 формул комплексных чисел отдельно
(x=143-233 pt, y=94-170 pt). Detector сливает их в один блок D1
(230×140 pt). Плюс detector находит формулы из правой колонки
(9.1, 9.4, 9.5, 9.8, 9.9, 9.10), которых нет в golden p262 —
это не FP, а недостающая разметка.
