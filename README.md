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

## Этапы

Активный план: [docs_ROADMAP.md](docs_ROADMAP.md). Исходники идей: `roadmap.md`, `roadmap_new.md`, chart pipeline.

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
