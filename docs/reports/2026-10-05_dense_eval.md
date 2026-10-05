# Dense-поиск и hybrid: полные результаты eval

**Дата:** 2026-10-05
**Контекст:** после коммита `672bfbb`, dense-индекс построен на `e5-small`, оценка на 3 датасетах

## Конфигурация

- Модель: `intfloat/multilingual-e5-small` (384 dim, локально в `data/models/multilingual-e5-small`)
- Чанков: 30937
- `dense.npy` — 47.5 MB
- `dense_ids.json` — 4.7 MB (30937 чанков, sha256 кэш)
- RRF: k=60, w_bm25=1.0, w_dense=1.0
- Индекс: `data/cache/text_index/`

## Eval: recall@1

| Датасет | lexical | dense | hybrid | ТЗ |
|---|---|---|---|---|
| retrieval_v1 (дословные) | **0.9231** | 0.3846 | 0.4615 | ≥ 0.90 |
| retrieval_v1_para (перефраз) | 0.2308 | **0.4615** | 0.3077 | ≥ 0.50 |
| retrieval_v1_paraphrase | 0.0000 | **0.2000** | 0.2000 | ≥ 0.40 |

## Eval: recall@5

| Датасет | lexical | dense | hybrid |
|---|---|---|---|
| retrieval_v1 | **1.0000** | 0.6154 | 0.7692 |
| retrieval_v1_para | 0.2308 | **0.6154** | 0.6154 |
| retrieval_v1_paraphrase | 0.0000 | **0.5000** | 0.5000 |

## Eval: MRR

| Датасет | lexical | dense | hybrid |
|---|---|---|---|
| retrieval_v1 | **0.9615** | 0.4744 | 0.5833 |
| retrieval_v1_para | 0.2308 | **0.5154** | 0.4026 |
| retrieval_v1_paraphrase | 0.0000 | **0.2833** | 0.2917 |

## Выводы

### 1. Dense побеждает на перефразах

- para: 0.23 → **0.46** (×2)
- paraphrase: 0.00 → **0.20** (с нуля)
- recall@5 paraphrase = 0.50 — половина находится в top-5

### 2. Dense проигрывает на дословных

- v1: 0.92 → **0.38** (падение ×2.4)
- Причина: dense ищет семантику, а не точные слова. Для `27.1435` он не знает, что это важно.

### 3. Hybrid с равными весами не оптимален

- v1: 0.46 (между lexical и dense, но ближе к dense)
- para: **0.31** — хуже dense (0.46), потому что BM25-результаты (плохие для para) вытесняют dense
- paraphrase: 0.20 (как dense)

Проблема: RRF с равными весами усредняет, а не берёт лучшее.

## Дополнительные тесты

### Query с разными rrf-k (запрос «требования к прочности гидросистем», hybrid)

| rrf-k | top-1 | score |
|---|---|---|
| **10** | **ФАП-118 п.3, стр.10** | **0.091** ← лучший |
| 20 | ФАП-118 п.3, стр.7 | 0.059 |
| 60 (default) | НЛГ-27 п.27.251 | 0.027 |
| 100 | НЛГ-27 п.27.251 | 0.017 |

**Вывод:** `rrf-k = 10-20` — значительно лучше, чем default 60.

### Query с разными весами

| Параметры | top-1 |
|---|---|
| `--w-bm25 3.0 --w-dense 1.0` | НЛГ-27 п.27.251 |
| `--w-bm25 1.0 --w-dense 3.0` | ФАП-118 п.3 |
| `--rrf-k 10` | **ФАП-118 п.3, стр.10** |

### `rrf-k=10` на разных запросах — универсально

- `EDTO` → ИКАО ×2 + Doc 9760
- `шасси` → НЛГ-27 ×2 + АС
- `пожарная безопасность` → РЦ-АП-ВД 6.3 ×3

Работает корректно, top-1 = релевантный.

### Probe (99 проб, recall@5)

| Режим | Recall@5 |
|---|---|
| lexical | **0.9899** |
| dense | 0.7576 ⚠️ |
| hybrid | 0.9091 |

Dense сильно ухудшает probe (точные фразы). Hybrid — компромисс.

## 🔥 Negative examples — КРИТИЧЕСКИЕ ПРОБЛЕМЫ

### 1. Dense даёт высокие score на мусорных XML-чанках

"рецепт пирога" --mode dense:

АС п.3.9.5.2.13.5, XML <lcMultipleSelect> — score 0.848 ⚠️

АС п.3.9.5.2.13.5, XML <lcAnswerOption> — score 0.845

АС п.3, "Установить пиропатроны" — score 0.842

Нет ничего общего с пирогом! Dense не различает мусор от текста.

### 2. Hybrid не уважает `--min-score`
"требования к прочности" --mode hybrid --min-score 5:

ФАП-118 п.3 (score 0.033) ← меньше 5, но возвращается

РЦ AC 29-2C п.29.307 (score 0.030)

`--min-score` действует только в lexical.

### 3. В индексе много мусора

- XML S1000D (`<para>`, `<lcMultipleSelect>`)
- Оглавления с точечными лидерами
- Таблицы чисел (US_Army)
- Технические URL (NASA)

Для dense — все близки к любому запросу. Портит top-k.

## Технические проблемы

1. `eval` не принимает `--w-bm25`, `--w-dense`, `--rrf-k` (в `query` есть)
2. `eval`/`query` не принимают `--embedding-model` (только `meta.json`)
3. `build --dense --embedding-model intfloat/multilingual-e5-base` при cache hit записал `e5-base` в `meta.json`, при этом векторы остались от `e5-small`. Следующий `query` начал качать `e5-base` (1.11 GB)

## Задачи разработчику

Подробно — в `docs/reports/2026-10-05_message_to_dev.txt`.

## Артефакты

- `data/cache/text_index/dense.npy` (47.5 MB, локально)
- `data/cache/text_index/dense_ids.json` (4.7 MB, локально)
- `data/cache/text_index/meta.json` (модель: `data/models/multilingual-e5-small`)
- `data/models/multilingual-e5-small/` (~2.5 GB, все форматы)
- Датасеты: `data/ir/gold/retrieval_v1*.json`
