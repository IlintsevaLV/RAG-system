# Отчёт: фиксы по задачам page-tests `_1954.pdf` (2026-09-22)

Источник задач: [docs/reports/2026-09-22_tasks.md](./2026-09-22_tasks.md)  
Источник тестов: [docs/reports/2026-09-22_page_tests_1954.md](./2026-09-22_page_tests_1954.md)

**Статус:** 4 из 5 задач закрыты в коде. Issue 3 (`max_seq_len`) оставлен на ручную проверку на рабочем ПК.

---

## Что сделано

| # | Задача | Статус | Файлы |
|---|---|---|---|
| 1 | Ложные таблицы на тексте | ✅ закрыта | `ingestion/region_detect.py` |
| 2 | OCR путает кириллицу и греческий | ✅ закрыта (частично) | `ingestion/table_pipeline.py` |
| 3 | `max_seq_len` мал | ⏸ не меняли | `unimernet_tiny.yaml` только на рабочем ПК |
| 4 | Дублирование formula+table | ✅ закрыта | `ingestion/region_detect.py` |
| 5 | Ложные figure на формулах | ✅ закрыта | `ingestion/region_detect.py` |

---

## Issue 1 — ложные таблицы (`span_cell_grid`)

### Что было не так

Детектор считал «numeric» почти любое короткое слово (широкий charset regex).
Из-за этого двустолбцовый/плотный текст получал высокий `numeric_ratio` и проходил как таблица.

### Фикс

1. `_is_numericish()` теперь требует **реальные цифры**, а не «короткое слово».
2. Span-таблица без числового каркаса (`numeric_ratio < 0.30`) отбрасывается.
3. Ruled-grid (`cv_line_grid`) для настоящих линованных таблиц не трогали.
4. Сохранены фильтры TOC / two-column prose.

### Проверка (локально, `_1954.pdf`)

| Страница | До | После |
|---|---|---|
| 24 (двухколоночный) | 1 ложная table | **0** |
| 37 (текст+формулы) | 3 ложные table | **0** |
| 190 (плотный текст) | 3 ложные table | **0** |
| 61 (настоящая таблица) | 1 table ok | **1 table ok** (сохранена) |

---

## Issue 2 — кириллица vs греческая OCR-голова в ячейках

### Что было не так

Для каждой ячейки сразу шли все три головы. `el` на кириллице давал мусор вида `$Β Γραπγcαx$`.

### Фикс

1. По умолчанию ячейка OCR-ится **только cyrillic-головой**.
2. Latin/Greek включаются только если есть math-маркеры, явный English или уже греческие символы.
3. В `span_cells`: греческий текст **без** math-контекста → `UNREADABLE` (не индексируем галлюцинацию).

### Что ещё остаётся

`Суженне` → `Сужение` — это ошибка самой cyrillic-головы, не греческой.
Нужен отдельный словарь/контекстный корректор OCR (не в этом коммите).

---

## Issue 4 — formula внутри table

### Фикс

В `merge_regions`: formula, чей bbox **полностью внутри** bbox table, подавляется
даже если IoU < 0.20 (из-за большой площади таблицы IoU часто маленький).

---

## Issue 5 — figure на формуле

### Фикс

Перед merge удаляются figure-кандидаты, которые:

- полностью содержат formula с `span_math_heuristic` score ≥ 0.70;
- и занимают < 15% площади страницы.

На стр. 25 после фикса: `figure=0`, formula остаётся.

---

## Issue 3 — `max_seq_len` (не сделано здесь)

Файл `data/models/unimernet/unimernet_tiny.yaml` **не в git** (веса/конфиг только на рабочем ПК).

Рекомендация для рабочего ПК:

```yaml
model:
  model_config:
    max_seq_len: 2048   # было 1536
```

Затем повторить T3 (`34,36,37,40,41,44`). Если качество/скорость упадут — вернуть `1536`.

---

## Изменённые файлы

```text
ingestion/region_detect.py   — numeric gate, containment formula⊂table, figure vs formula
ingestion/table_pipeline.py  — cyrillic-first cell OCR, safe greek span rule
docs/reports/2026-09-22_fixes_page_tests.md  — этот отчёт
```

---

## Как перепроверить на рабочем ПК

```powershell
cd C:\Users\l.ilyintseva\Desktop\RAG-system
git pull
$PythonMain = ".\.venv\Scripts\python.exe"
$Pdf = "data\raw\_1954.pdf"

$env:ENABLE_VLM = "false"
$env:ENABLE_UNIMERNET = "true"
$env:UNIMERNET_CONFIG_PATH = ".\data\models\unimernet\unimernet_tiny.yaml"
$env:OCR_ENABLE_LATIN = "true"
$env:OCR_ENABLE_GREEK = "true"

# Issue 1
& $PythonMain -m ingestion.run_regions $Pdf --pages 23,24 --out-dir "data\ir\page_tests\T9_twocolumn"
& $PythonMain -m ingestion.run_regions $Pdf --pages 10,30,102,190 --out-dir "data\ir\page_tests\T10_dense_text"
& $PythonMain -m ingestion.run_regions $Pdf --pages 34,36,37,40,41,44 --out-dir "data\ir\page_tests\T3_equations" --enable-unimernet

# Issue 2
& $PythonMain -m ingestion.run_regions $Pdf --pages 61,68,69,189 --out-dir "data\ir\page_tests\T6_real_tables"

# Issue 4 / 5
& $PythonMain -m ingestion.run_regions $Pdf --pages 1,2,3 --out-dir "data\ir\page_tests\T1_title"
& $PythonMain -m ingestion.run_regions $Pdf --pages 9,11,15,20,25 --out-dir "data\ir\page_tests\T5_figures"
```

**Ожидаемо:**

| Метрика | Было (отчёт) | Цель |
|---|---|---|
| Ложные таблицы (24/37/190) | 7 | **0** |
| Настоящая таблица (61) | 1 ok | 1 ok |
| formula внутри table (стр. 3) | дубль | formula подавлена |
| figure на формуле (стр. 25) | 1 ложный | **0** |

---

## Итог

Пайплайн закрывает главные регрессии из page-tests: ложные таблицы на тексте убраны,
пересечения formula/table/figure разрулены, OCR ячеек больше не гоняет греческую голову
на каждый русский фрагмент. Осталось вручную поднять `max_seq_len` на рабочем ПК и
повторить полный набор T1–T10.
