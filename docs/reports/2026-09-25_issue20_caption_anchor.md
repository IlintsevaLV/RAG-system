# Issue 20: фикс caption-anchor детекции рисунков на сканах

Дата: 2026-09-25. ТЗ: [`docs/issues/020_figure_detector_fails_on_scans.md`](../issues/020_figure_detector_fails_on_scans.md).

## Контекст

На golden set (`data/ir/gold/figures_v1.json`, 12 страниц, 13 figures)
детектор давал **P=0 R=0**. Caption находилась, но bbox был неверный
(`_1954` p9: верх уезжал ~212 pt, IoU ≈ 0.28). На сканах путь `pdf_image`
бесполезен (один XObject = вся страница → `too_large`). Ink-CC на сканах
сливает текст и графику → лишние false positives.

## Что сделано

### 1. Density-aware `tighten_bbox_to_ink`

Было: min/max по всем тёмным пикселям — одиночный пиксель / линия /
колонтитул не давали сжать окно.

Стало:

- строка / столбец считаются ink-несущими при плотности ≥ 2%;
- берётся самый длинный плотный прогон (короткие дыры внутри рисунка
  склеиваются);
- порог чернил от **фона** (85-й перцентиль серого), не от медианы окна —
  иначе окно, уже почти целиком рисунок, само себя не видело.

### 2. Окно над / под подписью

- Верх `above`: `max(низ предыдущего абзаца + 6 pt, caption − 400 pt)`.
- Низ `below`: упирается в следующий абзац.
- Короткие подписи осей (< 15 букв и уже 40% страницы) абзацем не
  считаются — иначе график обрезается изнутри.

### 3. Выбор above vs below

Оба кандидата дожимаются и проходят `is_valid_figure_bbox`. Выигрывает
максимальный **относительный** shrink `(area_before − area_after) /
area_before`, а не первое окно в кортеже.

### 4. Дополнения (без них счётчик фигур не падает)

- **Склейка разорванной подписи** в одну строку (`Fig.` + `1.10`), чтобы
  `FIGURE_CAPTION_RE` ловил rdk-подписи, разрезанные PyMuPDF.
- Если `caption_anchor` уже дал фигуру на странице — **ink-CC не
  добавляется** (на скане он давал 2–3 лишних бокса на p11/p15/p25).

Docling layout fallback **не** подключался: по плану Issue 20 он нужен
только если после этих правок recall на rdk89 останется < 0.7.

## Проверка здесь

```powershell
python scripts\check_figure_detect.py
```

Юнит-тесты: all PASS. Синтетика «как `_1954` p9»: **IoU ≈ 0.73**.
Реальных PDF в этом клоне нет — golden IoU нужно снять на рабочем ПК.

## На рабочем ПК

```powershell
git pull
python scripts\check_figure_detect.py --pdf "data/raw/_1954.pdf"
python scripts\check_figure_detect.py --pdf "data/raw/РДК_молниезащита 89г.pdf"
```

Целевые метрики Issue 20: **P ≥ 0.7, R ≥ 0.7, mean_best_iou ≥ 0.6** на
обоих документах.

Ожидание по страницам:

| Страница | Было | Цель |
|---|---|---|
| `_1954` p9 | IoU ≈ 0.28 | IoU ≥ 0.6, 1 figure |
| `_1954` p11 / p15 / p25 | pred=3 | pred=1 (или 2 на p25 по двум caption) |
| `_1954` p20 | 1 склеенный | до 3 по подписям |
| rdk89 p25 / p26 / p39 | 0 | ≥1 по `Fig. …` |

Если rdk всё ещё R < 0.7 — следующий шаг: Docling Picture layout fallback
с приоритетом `pdf_image > caption_anchor > docling_layout`.

## Файлы

| Файл | Изменение |
|---|---|
| `ingestion/region_detect.py` | density tighten, caption windows, split-caption join, ink-CC gate |
| `scripts/check_figure_detect.py` | тесты hairline / p9-like / split Fig / below-wins |
