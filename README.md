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

## OCR-модели без ModelScope (рабочий ПК)

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
Класс D = нечитаемая страница (как US Army p572) — в RAG как факт не кладём.

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
