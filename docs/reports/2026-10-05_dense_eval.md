# Dense-поиск: результаты eval

**Дата:** 2026-10-05
**Контекст:** после коммита `672bfbb`, dense-индекс построен на `e5-small`

## Конфигурация

- Модель: `intfloat/multilingual-e5-small` (384 dim, локально в `data/models/multilingual-e5-small`)
- Чанков: 30937
- `dense.npy` — 47.5 MB
- `dense_ids.json` — 4.7 MB (30937 чанков, sha256 кэш)
- RRF: k=60, w_bm25=1.0, w_dense=1.0
- Индекс: `data/cache/text_index/`

## Результаты eval (recall@1)

| Датасет | lexical | dense | hybrid | ТЗ |
|---|---|---|---|---|
| retrieval_v1 (дословные) | **0.9231** | 0.3846 | 0.4615 | ≥ 0.90 |
| retrieval_v1_para (перефраз) | 0.2308 | **0.4615** | 0.3077 | ≥ 0.50 |
| retrieval_v1_paraphrase | 0.0000 | **0.2000** | 0.2000 | ≥ 0.40 |

## Результаты eval (recall@5)

| Датасет | lexical | dense | hybrid |
|---|---|---|---|
| retrieval_v1 | **1.0000** | 0.6154 | 0.7692 |
| retrieval_v1_para | 0.2308 | **0.6154** | 0.6154 |
| retrieval_v1_paraphrase | 0.0000 | **0.5000** | 0.5000 |

## Результаты eval (MRR)

| Датасет | lexical | dense | hybrid |
|---|---|---|---|
| retrieval_v1 | **0.9615** | 0.4744 | 0.5833 |
| retrieval_v1_para | 0.2308 | **0.5154** | 0.4026 |
| retrieval_v1_paraphrase | 0.0000 | **0.2833** | 0.2917 |

## Выводы

### 1. Dense работает на перефразах

- **para:** 0.23 → **0.46** (×2)
- **paraphrase:** 0.00 → **0.20** (с нуля)
- **recall@5 paraphrase = 0.50** — половина находится в top-5

### 2. Dense проигрывает на дословных

- **v1:** 0.92 → **0.38** (падение ×2.4)
- Причина: dense ищет семантику, а не точные слова. Для `27.1435` он не знает, что это важно.

### 3. Hybrid с равными весами не оптимален

- **v1:** 0.46 (между lexical и dense, но ближе к dense)
- **para:** **0.31** — **хуже dense** (0.46), потому что BM25-результаты (плохие для para) вытесняют dense
- **paraphrase:** 0.20 (как dense)

Проблема: RRF с равными весами усредняет, а не берёт лучшее.

## Технические проблемы

### 1. `eval` не принимает флаги RRF

`query` поддерживает `--w-bm25`, `--w-dense`, `--rrf-k`, но `eval` — нет.
**Нельзя подобрать веса при оценке.**

### 2. `eval`/`query` не принимают `--embedding-model`

Зависимость от `meta.json`. Нельзя явно указать модель.

### 3. Баг смены модели при cache hit

`build --dense --embedding-model intfloat/multilingual-e5-base` при cache hit:
- Векторы остались от `e5-small` (правильно)
- Но `meta.json` записал `e5-base`
- Следующий `query`/`eval` начал качать `e5-base` (1.11 GB)

Временно исправлено: `meta.json` → `data/models/multilingual-e5-small` (локальный путь).

## Задачи разработчику

Подробно — в `docs/reports/2026-10-05_message_to_dev.txt`.

- Добавить `--w-bm25`, `--w-dense`, `--rrf-k` в `eval`
- Добавить `--embedding-model` в `eval` и `query`
- Не менять `meta.json` при cache hit
- Адаптивный hybrid (по score BM25)

## Артефакты

- `data/cache/text_index/dense.npy` (47.5 MB, локально, не в git)
- `data/cache/text_index/dense_ids.json` (4.7 MB, локально)
- `data/models/multilingual-e5-small/` (~2.5 GB, все форматы)
- Датасеты: `data/ir/gold/retrieval_v1*.json`
