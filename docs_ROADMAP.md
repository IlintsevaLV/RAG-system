# Roadmap: RAG по нормативке (рабочий ПК + GitHub)

## Цель

Отвечать на технические вопросы **строго по документам** из `Нормативка/` и всегда отдавать
**ссылку на место** (`doc_id` + страница + `bbox` / zone / chart_id).

Разработка кода — на этом ПК (Cursor). Тяжёлый прогон (OCR/VLM/GPU) — на рабочем ПК.
Обмен: GitHub (код + конфиги + IR/манифесты; PDF в git не кладём).

## Принципы (из `roadmap_new` + `roadmap` + chart pipeline)

1. **Evidence first** — сначала геометрия/текст/OCR, потом смысл.
2. **VLM = recovery**, не основной OCR всего листа.
3. **Unified IR** с provenance у каждого блока.
4. **Embeddings + graph** вместе (Qdrant + Neo4j), не вместо друг друга.
5. **Один этап — одна сессия**: отладка → прогон на реальных файлах → коммит → следующий этап.
6. **Скромные ресурсы**: модели по очереди, освобождение VRAM, `--resume`, CPU-fallback.

## Схема

```text
PDF (локально на рабочем ПК, не в git)
  → SOURCE ANALYSIS (TLQ / vector / raster)
  → PREPROCESS (IQS: good/medium/bad)
  → LAYOUT (TEXT | FORMULA | TABLE | CHART | DRAWING)
  → specialized extractors
  → UNIFIED IR + quality routing
  → normalize → entities → Qdrant + Neo4j
  → retrieval (hybrid + grounding) → ответ + цитата
```

---

## Этап 0. Репозиторий и перенос (сейчас)

- [ ] `git init`, структура проекта, `.gitignore` (PDF, модели, `.env`, cache)
- [ ] `README` с инструкцией: clone на рабочем → `.env` → docker → pipeline
- [ ] Каталоги: `data/raw` (symlink/копия нормативки вне git), `data/ir`, `data/eval`
- [ ] Скрипт `scripts/smoke_work_pc.py` — проверка CUDA / VRAM / docker

**Готово:** репозиторий пушится на GitHub без секретов и без PDF.

---

## Этап 1. Скелет проекта

Из `roadmap.md`, без лишнего.

- [ ] `docker-compose.yml`: Qdrant + Neo4j (GDS позже, если хватит RAM)
- [ ] `core/config.py` (pydantic-settings), `.env.example`
- [ ] `core/logger.py`
- [ ] `core/interfaces.py`: LLM / Vision / OCR / Embedding / Reranker / VectorStore / GraphStore
- [ ] `core/ir.py`: envelope блока (Unified IR)
- [ ] `requirements.txt` (минимальный набор)

**Готово:** контейнеры поднимаются, конфиг читается, интерфейсы согласованы.

---

## Этап 2. Железо и модели (только рабочий ПК)

Раньше бизнес-логики — иначе рискуем неделю зря.

- [ ] `core/model_server.py`: start / ready / stop, контекстный менеджер
- [ ] GGUF по очереди (не держать две модели в VRAM): LLM 7–8B Q4/Q5, VL отдельно
- [ ] Проверка: ответ + **VRAM освобождается** после stop
- [ ] GBNF на простом примере; один multimodal smoke

**Fallback при слабом GPU:** только RapidOCR + text layer; VLM выключен флагом.

**Готово:** две модели последовательно в одном скрипте без OOM.

---

## Этап 3. Source analysis + OCR baseline

Слияние этапа OCR из `roadmap` и §1–2 из `roadmap_new`.

- [ ] TLQ по text layer → route: text | OCR
- [ ] IQS → good / medium / bad preprocess (бинаризация **не** по умолчанию)
- [ ] RapidOCR + нормализация омоглифов + structural check обозначений
- [ ] Horizontal full-width bands (не квадратные tiles как основной путь)
- [ ] Манифест + `--resume` + `--pages`
- [ ] Eval: CER/WER + line coverage + канонизация

**Готово:** стабильный прогон по корпусным PDF; метрика на 1–2 вычитанных страницах.

---

## Этап 3а. Layout + specialized regions

- [ ] Детекция блоков: TEXT / TABLE / CHART / DRAWING / FORMULA
- [ ] DRAWING: зоны по ГОСТ (title block, ТТ, боковые, поле) — не весь A1 в VLM
- [ ] TABLE: структура → OCR по ячейкам (Docling + RapidOCR backend)
- [ ] FORMULA: позже / optional (UniMERNet) — не блокирует MVP
- [ ] CHART: каркас по `chart_extraction…` (detection → evidence → graph model → validate → VLM recovery)

**Готово:** IR с `type` + `bbox` + provenance; чарт/чертёж не ломают текстовый путь.

---

## Этап 4. Парсинг → чанки с происхождением

- [ ] IR → chunks: `doc_id`, page, bbox, region_id
- [ ] XLSX-спецификации — детерминированный парсер (когда появятся)
- [ ] Извлечение изображений / crops зон для UI-цитат

**Готово:** таблицы из реальных docs; у каждого чанка есть страница+bbox.

---

## Этап 5–6. Schema + entities

- [ ] Компактная схема узлов/рёбер под 4 типа вопросов (GBNF)
- [ ] Детерминированный extract → LLM на остаток
- [ ] Entity resolution + омоглифы в канонизации (обязательно)
- [ ] Выгрузка спорных кейсов

**Готово:** номенклатура без дублей на реальных данных; 8B стабильно заполняет схему.

---

## Этап 7. Хранилища + оркестратор

- [ ] Qdrant: dense + sparse, payload = provenance + reserved ACL fields
- [ ] Neo4j за абстракцией GraphStore
- [ ] `orchestrator`: subprocess stages, manifest, `--resume`, `--only-stage`

**Готово:** полный прогон корпуса + прерывание с resume.

---

## Этап 8. Золотой набор (до тюнинга поиска)

- [ ] 20–30 Q в `eval/golden_set.yaml`
- [ ] Сценарии + «ответа нет» + нечёткие формулировки
- [ ] Метрика: correctness + **provenance accuracy** (страница/bbox)

---

## Этап 9. Retrieval + grounding

- [ ] Query analyzer (+ уточняющий вопрос)
- [ ] Hybrid Qdrant (RRF) + graph walk
- [ ] Rerank + пороги отказа («не найдено в корпусе»)
- [ ] Ответ только с цитатой; иначе refuse

---

## Этап 10. API + UI

- [ ] FastAPI + SSE
- [ ] Лёгкий UI: ответ + документ + страница + подсветка bbox

---

## Отложено (как в исходном roadmap)

- ACL (поля зарезервированы)
- Louvain/Leiden обобщения
- Инкрементальный reindex
- Полная телеметрия
- UniMERNet / тяжёлый chart reconstruction — после текстового MVP

## Порядок внедрения сейчас

**Этап 0+1** → push на GitHub → на рабочем ПК smoke железа → этап 2 → 3.
Chart/formula не блокируют MVP: текстовый RAG с цитатами важнее.
