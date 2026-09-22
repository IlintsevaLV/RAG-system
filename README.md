# rag_system — технический RAG по нормативке

Ответы **строго по документам** с цитатой: документ + страница + bbox/зона.

## Где что крутится

| Место | Что |
|--------|-----|
| Этот ПК (Cursor) | код, конфиги, IR-схемы, eval |
| Рабочий ПК | Docker (Qdrant/Neo4j), OCR/VLM, индексация, API |
| GitHub | код; **не** PDF, **не** `.env`, **не** веса моделей |

## Быстрый старт (рабочий ПК)

```bash
git clone <repo>
cd rag_system
cp .env.example .env
# положите PDF в data/raw/ (или укажите NORMATIVKA_DIR)
docker compose up -d
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/smoke_work_pc.py
```

Если PowerShell блокирует `Activate.ps1`, активация не нужна:

```powershell
& .\.venv\Scripts\python.exe -m ingestion.run_extract_pages data\raw
```

Для длительных запусков это предпочтительный вариант на корпоративном
Windows-ПК: не требуется менять `ExecutionPolicy`.

## Source analysis + preprocess + OCR (этап 3)

```powershell
pip install -r requirements.txt

# TLQ только
python -m ingestion.run_source_analysis data\raw --pages 1-5

# TLQ → preprocess → RapidOCR для страниц с низким TLQ
python -m ingestion.run_source_analysis data\raw --pages 1-5 --ocr

# Принудительно OCR даже при хорошем text layer (отладка)
python -m ingestion.run_source_analysis data\raw --pages 10 --ocr --force-ocr
```

Результат: `data/ir/<doc_id>_source.json` (поля `tlq`, `iqs`, `ocr.lines` с bbox).
Порог `TLQ_THRESHOLD` (0.65) позже калибруется на golden set.
Мусорный text layer: `lexical_quality` + `garbage_veto` (даже при высоком TLQ → OCR).
Модель: RapidOCR PP-OCRv5 Cyrillic (`ENABLE_GPU_OCR=false` по умолчанию).

## Поиск мусорного text layer (рабочий ПК)

```powershell
git pull
pip install -r requirements.txt

# Быстрый обзор: ~8 страниц на документ
python -m ingestion.scan_garbage_text_layer data\raw --pages-per-doc 8 --no-visual

# Полный прогон всех страниц (дольше)
python -m ingestion.scan_garbage_text_layer data\raw --full --no-visual
```

Результаты для вычитки:
- `data/ir/garbage_text_layer_candidates.csv` — приоритетный список страниц
- `data/ir/garbage_text_layer_report.json` — полный отчёт


## OCR: кириллица + латиница + греческий

Три recognition-головы (если модели лежат в `OCR_MODEL_DIR`):
- `cyrillic_PP-OCRv5_rec_mobile.onnx` — длинный русский текст;
- `latin_PP-OCRv5_rec_mobile.onnx` — длинный английский / латинский текст;
- `el_PP-OCRv5_rec_mobile.onnx` — греческие буквы в формулах и обозначениях (α β γ θ λ μ π σ ω …).

Перекрывающиеся боксы склеиваются по script-affinity: греческий для формул,
латиница для English prose, кириллица для русского.

```powershell
python -m scripts.prepare_ocr_models
# в data\models\ocr должны быть cyrillic + latin + el onnx
```

Пограничные случаи, которые пайплайн учитывает:
- греческие буквы в тексте и в ячейках таблиц (dual OCR + soft recovery);
- формулы внутри ячеек → `$LaTeX$` (UniMERNet, если включён, иначе formula-text);
- мелкие ячейки / индексы — upsample перед OCR;
- многострочные ячейки — склеиваются в одну Markdown-ячейку;
- оглавления и двустолбцовый текст — не таблицы;
- пустые/разреженные ложные сетки — отвергаются, а не пишутся в RAG.

Ещё не полностью закрыто (осознанный backlog):
- таблицы, разрезанные разрывом страницы (нужна склейка across pages);
- вертикальный текст в шапке;
- сложные merged cells без линий.

## Регионы: формулы / таблицы / рисунки

```powershell
python -m ingestion.run_regions data\raw\doc.pdf --pages 41

# VLM recovery для плохих таблиц/формул
python -m ingestion.run_regions data\raw\doc.pdf --pages 41 --enable-vlm

# UniMERNet (когда установите пакет)
python -m ingestion.run_regions data\raw\doc.pdf --pages 41 --enable-unimernet
```

Опционально: `pip install docling paddleocr img2table pylatexenc`  
Результат: `data/ir/regions/<doc>_regions.json` (latex / markdown / caption + bbox).

Для сканов нормативки таблица обрабатывается в таком порядке:
`span_cells` (для таблиц без линий) → `cell_ocr` (сетка → OCR ячеек) → Docling → PP-Structure → img2table
→ VLM recovery.  VLM не является основным источником чисел в таблицах.
Результат с низкой структурной оценкой помечается `suspicious`.

Детекция регионов таблиц специально отсекает:
- двустолбцовый основной текст;
- оглавления / содержание с точечными лидерами и номерами страниц;
- облака коротких OCR-фрагментов на всю страницу.
Принимаются либо настоящие линейные сетки (`cv_line_grid`), либо
локальные компактные cell-grid из выровненных коротких ячеек
(`span_cell_grid`). Итоговый формат для RAG — Markdown-таблица.

## Извлечение текста по классам A/B/C/D


На рабочем ПК часто нет доступа к `modelscope.cn`. Модели кладём локально:

```powershell
# На ПК с интернетом / где rapidocr уже скачал модели:
python -m scripts.prepare_ocr_models --zip

# Скопируйте папку data\models\ocr  (или ocr_models.zip) на рабочий ПК
# в тот же путь относительно репозитория, затем:
python -m ingestion.run_extract_pages data\raw --pages 1-5
```

В `.env`: `OCR_MODEL_DIR=./data/models/ocr`

```powershell
# A/B: text layer + нормализация (+ OCR-проверка для B)
# C/D: preprocess + VLM (или OCR fallback, если VLM выключен)
python -m ingestion.run_extract_pages data\raw --pages 1-5

# С VLM на рабочем ПК (llama-server должен слушать MODEL_HOST:PORT)
# в .env: ENABLE_VLM=true
python -m ingestion.run_extract_pages data\raw --pages 83,246 --enable-vlm
```

Результат: `data/ir/pages/<doc>_pages.json` и `data/ir/pages/<doc>/page_XXXX.txt`.
Для страниц класса C JSON содержит отдельные поля `needs_vlm`,
`vlm_attempted`, `vlm_used`, `vlm_quality` и `ocr_fallback_used`.
Класс D = нечитаемая страница (как US Army p572) — в RAG как факт не кладём.
`needs_vlm` означает «страница требует визуального маршрута», а не
«VLM уже обработал страницу». Для контроля смотрите одновременно
`vlm_attempted`, `vlm_used`, `vlm_quality` и `text_source`.
В summary `needs_vlm` — число визуальных кандидатов, а
`needs_vlm_pending` — кандидаты без успешного VLM-результата.
`ocr_fallback` всегда имеет `status=suspicious`, даже если VLM был включён:
это запасной результат, требующий проверки.

## Формулы (UniMERNet)

UniMERNet требует отдельные веса и официальный конфигурационный файл:

```powershell
pip install -U "unimernet[full]"
git clone https://github.com/opendatalab/UniMERNet.git data\models\unimernet_src
git clone https://huggingface.co/wanderkid/unimernet_tiny data\models\unimernet\unimernet_tiny
$env:UNIMERNET_CONFIG_PATH="data\models\unimernet_src\configs\demo.yaml"
$env:ENABLE_UNIMERNET="true"
python -m ingestion.run_regions data\raw\_1954.pdf --pages 41 --enable-unimernet --enable-vlm
```

Если `UNIMERNET_CONFIG_PATH` не задан, формульный блок остаётся
`suspicious` и используется VLM fallback, если он включён.

## Текущий полный pipeline страницы

Одна PDF-страница **не** отправляется целиком в одну модель. Сначала
фиксируется provenance (doc_id, номер страницы, размеры, bbox), затем
идут **два параллельных контура**:

1. **Текст страницы** (маршрут A/B/C/D) → `data/ir/pages/…`
2. **Объекты на листе** (таблица / рисунок / формула) → `data/ir/regions/…`

```text
                         ┌─────────────────────────┐
                         │     PDF-страница        │
                         │  (provenance: doc/page) │
                         └───────────┬─────────────┘
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
        КОНТУР 1: ТЕКСТ                         КОНТУР 2: РЕГИОНЫ
   (run_extract_pages)                         (run_regions)
                 │                                       │
                 ▼                                       ▼
      1. Извлечь text layer                    1. Render 200–300 dpi
      2. TLQ (качество слоя)                   2. Spans: text layer
         • объём / слова                         или OCR pseudo-spans
         • lexical garbage                     3. Детекция кандидатов
         • visual agreement                       (таблица / рисунок /
      3. Класс страницы A/B/C/D                    формула)
      4. Нормализация → page.txt               4. Приоритет пересечений:
                                                  FIGURE/TABLE > FORMULA
                                               5. Распознавание содержимого
                                               6. IR: markdown / latex /
                                                  caption + bbox + quality
```

### Контур 1 — текст страницы (A/B/C/D)

| Класс | Когда | Что делаем | Куда в RAG |
|-------|--------|------------|------------|
| **A** | Text layer чистый | Только слой + нормализация | Как факт |
| **B** | Слой сомнительный | Слой + OCR-проверка (три головы), берём лучший / эскалируем | Как факт, если согласование ок |
| **C** | Скан / плохой слой | Preprocess → VLM (если есть) → иначе RapidOCR + reading order | С телеметрией `needs_vlm` / `vlm_*`; OCR-fallback = `suspicious` |
| **D** | Страница нечитаема | Recovery или отказ | **Не** кладём как факт |

Пояснения:
- **TLQ** решает класс **до** OCR/VLM. OCR не заменяет text layer на каждой странице.
- **OCR** (RapidOCR): три головы — `cyrillic` (русский), `latin` (английский), `el` (греческий для формул). Перекрытия склеиваются по script-affinity.
- **Нормализация**: whitespace, homoglyphs (с сохранением греческих), spaced letters, boilerplate скана.
- **VLM** (`llama-server`) включается только при `ENABLE_VLM=true`. Поля `needs_vlm` ≠ «VLM уже отработал».

### Контур 2 — регионы (объекты кроме «просто текста»)

Порядок приоритета при пересечении bbox:

`рисунок / таблица → формула → остаточный текст`

**Таблицы** (результат для RAG = Markdown):
1. Детекция: линейная сетка (`cv_line_grid`) или локальный cell-grid (`span_cell_grid`).
2. Явно **не** таблицы: оглавления (лидеры `....`), двустолбцовый текст.
3. Содержимое: `span_cells` → `cell_ocr` (+ формулы в ячейках как `$LaTeX$`) → Docling → PP-Structure → img2table → VLM recovery.
4. Пустые/разреженные сетки отвергаются, а не пишутся в IR.

**Формулы**:
1. Строгий math-heuristic по spans (без библиографии и колонтитулов).
2. Crop → UniMERNet → validate LaTeX (нужен math-сигнал, не только pylatexenc) → VLM fallback.

**Рисунки / графики**:
1. Non-text connected components; пометка `chart_candidate`.
2. Crop + опциональный VLM-caption.

### CLI

```powershell
# Текст
python -m ingestion.run_extract_pages data\raw --pages 1-5
python -m ingestion.run_extract_pages data\raw --pages 83,246 --enable-vlm

# Регионы
python -m ingestion.run_regions data\raw\doc.pdf --pages 41
python -m ingestion.run_regions data\raw\doc.pdf --pages 41 --enable-unimernet --enable-vlm
```

Артефакты: `data/ir/pages/<doc>_pages.json`, `page_XXXX.txt`,
`data/ir/regions/<doc>_regions.json`.

## Структура

```text
core/          конфиг, логгер, интерфейсы, IR
ingestion/     OCR / layout / extract (этапы 3+)
storage/       Qdrant / Neo4j
retrieval/     поиск
agent/         API / UI
eval/          golden set
scripts/       smoke / утилиты
data/raw/      PDF (gitignored)
data/ir/       промежуточный IR (можно коммитить jsonl выборочно)
```
