# Issue 20: Figure detector не работает на сканах

**Дата:** 2026-09-25  
**Приоритет:** высокий  
**Связан с:** [#016](./016_figure_crop_cyrillic.md), [#017](./017_figure_caption_fallback.md), [#018](./018_figure_bbox_filter.md)

## Проблема

Detector figure в `region_detect.py` **полностью проваливает** golden set
(`data/ir/gold/figures_v1.json`, 12 страниц, 13 figures).

## Метрики на golden set

### `_1954.pdf`

```
IoU>=0.5: P=0.00 R=0.00 mean_best_iou=0.33 tp=0 fp=12 fn=8
```

### `РДК_молниезащита 89г.pdf`

```
IoU>=0.5: P=0.00 R=0.00 mean_best_iou=0.01 tp=0 fp=1 fn=5
```

**Recall = 0% на обоих документах.**

## Три причины

### A. Figures внутри сканов не детектируются

`_1954` и `rdk89` — **сканы**. Каждая страница — **один image XObject
(весь скан)**. Detector работает с image XObjects и видит всю страницу
как один объект → отсеивает как `too_large`.

**Figures внутри скана (фото, схемы, графики) полностью невидимы.**

Примеры:
- `_1954 p9`: фото Bell H-13 внутри скана. Detector: **0 figure**.
- `_1954 p15`: 5 схем фиг. I.9 внутри скана. Detector: **3 ложные**.
- `_1954 p20`: 3 figures (I.14, I.15, I.16). Detector: **1 ложная**.
- `_1954 p25`: 2 chart (I.18, I.19). Detector: **3 ложные**.
- `rdk89 p25`: Fig. 1.9 Positive corona. Detector: **0**.
- `rdk89 p26`: Fig. 1.10 (a, b). Detector: **0**.
- `rdk89 p39`: Fig. 2.1 (a, b, c). Detector: **0**.

### B. Ложные срабатывания на текст

- `_1954 p11`: golden=1, pred=3. Detector нашёл текст как figure.
- `_1954 p15`: golden=1, pred=3. То же.
- `_1954 p25`: golden=2, pred=3. То же.
- `rdk89 p1`: golden=1, pred=1, но matched=0. Detector нашёл всю страницу, golden — только фото здания.

### C. Bbox не совпадает с golden (mean_best_iou=0.33)

Даже когда detector что-то нашёл — bbox **не совпадает** с golden
на 50% площади. Возможно, ошибка в единицах (pt vs px), или detector
склеивает несколько figures в один bbox.

Пример `p0020`: golden=3 (I.14, I.15, I.16), pred=1, matched=0.
Detector нашёл 1 общий bbox на все 3 figures? Или один из них неправильно?

## Фикс

### A. Pixel-based detection для сканов

Если у страницы **1 image XObject = вся страница** (скан):

1. **Использовать OpenCV** на rendered image_bgr:
   - threshold (чернила vs фон),
   - morphological closing,
   - connected components,
   - фильтр по размеру (>= 80×80 pt, <= 60% page),
   - фильтр по aspect ratio.

2. **Или Docling layout model** (уже установлен):
   - `docling.datamodel.pipeline_options.PdfPipelineOptions`
   - использовать только layout detection, без OCR.

3. **Или LayoutParser / Detectron2** (если Docling не подходит).

### B. Ужесточить фильтры для false positives

В `is_valid_figure_bbox`:
- Требовать **контраст** относительно фона (фото/схема — плотный блок).
- Отсеивать **узкие полосы** и **тонкие линии**.
- Отсеивать **текстовые блоки** (проверить, что внутри — не строки текста).
- Отсеивать **эмблемы / логотипы** (маленькие круглые/квадратные объекты).

### C. Проверить единицы bbox

Убедиться, что `bbox_pt` (points) vs `bbox_px` (pixels) правильно
конвертируются. Сейчас `mean_best_iou=0.33` — слишком высоко для «не нашли»,
но слишком низко для «нашли». Возможно, bbox **смещён на 20-30%**.

## Тесты

```powershell
& $PythonMain scripts\check_figure_detect.py --pdf "data/raw/_1954.pdf"
& $PythonMain scripts\check_figure_detect.py --pdf "data/raw/РДК_молниезащита 89г.pdf"
```

**Цель после фикса:**
```
_1954:    P >= 0.7, R >= 0.7, mean_best_iou >= 0.6
rdk89:    P >= 0.7, R >= 0.7, mean_best_iou >= 0.6
```

## Файлы для изменения

- `ingestion/region_detect.py` — `detect_figure_regions`, `is_valid_figure_bbox`.
- Возможно, добавить `ingestion/layout_detect.py` — модуль для layout detection.
- `scripts/check_figure_detect.py` — расширить тесты.

## Приложение: golden set

`data/ir/gold/figures_v1.json` — 12 pages, 13 figures, needs_review=false.