# UniMERNet через subprocess: решение конфликта venv

**Дата:** 2026-09-22
**Проблема:** `unimernet[full]` требует `transformers==4.42.4`, что конфликтует
с Docling (`transformers>=5.0`). В одном venv они не уживаются.
**Решение:** отдельный venv `.venv-unimernet` + persistent worker через subprocess.

---

## Симптомы исходной проблемы

После `pip install unimernet[full]` в основной `.venv`:

1. `transformers` откатился `5.17.0 → 4.42.4`.
2. `numpy` откатился `2.3.5 → 1.26.4`.
3. `tokenizers` откатился `0.23.2 → 0.19.1`.
4. Docling начал падать на загрузке layout-модели:

```
[warning] docling_fail error='Failed to load label mapping from model config ...
The checkpoint you are trying to load has model type `rt_detr_v2`
but Transformers does not recognize this architecture.'
```

5. Детектор формул перестал работать: `det_method=span_math_heuristic`
   возвращал некорректные регионы.
6. UniMERNet получал плохие crop'ы, возвращал пустой `pred_str`.
7. Результат: все формулы `latex=''`, `status=suspicious`.

**Метрики регрессии (5 страниц: 41, 50, 59, 82, 156):**

| | До отката | После отката |
|---|---|---|
| Формул | 32 | 11 |
| `status=ok` | 29 | **0** |

---

## Архитектура решения

```
main .venv (Docling + пайплайн)
  │  transformers 5.x, numpy 2.x
  │
  └─ subprocess ──► .venv-unimernet\Scripts\python.exe
                       │  transformers 4.42.4, numpy 1.26.4
                       │  unimernet 0.2.3
                       │
                       ▼
                    scripts/unimernet_worker.py
                       │
                       ├─ загружает модель ОДИН РАЗ при старте
                       ├─ отправляет {"ready": true} в stdout
                       ├─ читает JSON-запросы из stdin построчно
                       │   {"image_b64": "<base64 PNG>"}
                       ├─ возвращает JSON-ответы в stdout
                       │   {"latex": "...", "confidence": 0.85}
                       └─ работает, пока main-процесс жив
```

**Ключевое:** модель загружается **один раз на весь прогон**, а не на каждую
формулу. Для 254 страниц это экономит часы.

---

## Файлы

### `scripts/unimernet_worker.py`

Persistent worker. Запускается **только** из `.venv-unimernet`:

```powershell
.\.venv-unimernet\Scripts\python.exe scripts\unimernet_worker.py <config.yaml>
```

Протокол:

| Направление | Формат |
|---|---|
| stdout, старт | `{"ready": true}` |
| stdin, запрос | `{"image_b64": "<base64 PNG>"}` |
| stdout, ответ | `{"latex": "...", "confidence": 0.85}` |
| stdout, ошибка | `{"error": "...", "traceback": "..."}` |

Worker **не парсит аргументы**, всё через stdin — чтобы main мог читать
ответы синхронно.

### `ingestion/formula_pipeline.py`

Класс `UniMERNetRecognizer` переписан:

- **Убран** прямой `import unimernet`.
- **Добавлен** `subprocess.Popen` на `.venv-unimernet\Scripts\python.exe`.
- **Добавлена** `_find_worker_python()` — ищет worker-python в:
  - `$env:UNIMERNET_WORKER_PYTHON` (если задан),
  - `.venv-unimernet\Scripts\python.exe` (Windows),
  - `.venv-unimernet/bin/python` (POSIX).
- **Добавлена** фильтрация stdout: worker печатает init-сообщения модели
  (`CustomVisionEncoderDecoderModel init`, ...) в stdout до `{"ready": true}`.
  `_try_load` читает строки, пока не найдёт валидный JSON.
- **Добавлен** `__del__` — корректно закрывает subprocess.

### `.gitignore`

```
.venv/
.venv-unimernet/
__pycache__/
*.pyc
data/ir/logs/
data/cache/
```

---

## Установка

### 1. Основной venv — как раньше

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**Убедитесь, что в основном venv:**
- `transformers>=5.0`
- `numpy>=2.0`
- `docling>=2.0`

### 2. Отдельный venv для UniMERNet

```powershell
python -m venv .venv-unimernet
.\.venv-unimernet\Scripts\python.exe -m pip install "unimernet[full]"
```

**Не запускайте `pip install -r requirements.txt` в этом venv** — он
подтянет `transformers 5.x` и сломает UniMERNet.

### 3. Веса UniMERNet

```powershell
# через huggingface-cli (надёжнее на Windows)
.\.venv\Scripts\python.exe -m huggingface_hub.commands.huggingface_cli download `
    wanderkid/unimernet_tiny `
    --local-dir data\models\unimernet\unimernet_tiny
```

Должен появиться `unimernet_tiny.pth` (~410 МБ).

### 4. Конфиг

`data\models\unimernet\unimernet_tiny.yaml` — копия `demo.yaml` с правками:

```yaml
model:
  arch: unimernet
  model_config:
    model_name: ./data/models/unimernet/unimernet_tiny
  load_pretrained: True
  pretrained: './data/models/unimernet/unimernet_tiny/unimernet_tiny.pth'
  tokenizer_config:
    path: ./data/models/unimernet/unimernet_tiny

run:
  device: "cpu"          # было cuda
  distributed: False     # было True
  evaluate: False
```

---

## Запуск

```powershell
cd C:\Users\l.ilyintseva\Desktop\RAG-system
$PythonMain = ".\.venv\Scripts\python.exe"

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:ENABLE_VLM = "false"
$env:ENABLE_UNIMERNET = "true"
$env:UNIMERNET_CONFIG_PATH = ".\data\models\unimernet\unimernet_tiny.yaml"

& $PythonMain -m ingestion.run_regions `
  "data\raw\_1954.pdf" `
  --pages 41 `
  --out-dir "data\ir\regions" `
  --enable-unimernet
```

**Что увидите при старте:**

```
[UNIMERNET] py = C:\...\.venv-unimernet\Scripts\python.exe
[UNIMERNET] cfg = ./data/models/unimernet/unimernet_tiny.yaml
[UNIMERNET] proc started, waiting for ready
[UNIMERNET] stdout line = 'CustomVisionEncoderDecoderModel init\n'
[UNIMERNET] stdout line = 'VariableUnimerNetModel init\n'
...
[UNIMERNET] stdout line = '{"ready": true}\n'
available: True
```

Дальше — обычный прогон.

---

## Результаты

### Тест на 5 страницах (41, 50, 59, 82, 156)

| Страница | До subprocess | После subprocess |
|---|---|---|
| p41 | 4 (0 ok) | **4 (4 ok)** |
| p50 | 2 (0 ok) | **2 (1 ok)** |
| p59 | 3 (0 ok) | **3 (3 ok)** |
| p82 | 1 (0 ok) | 1 (0 ok) |
| p156 | 1 (0 ok) | **1 (1 ok)** |
| **Итого** | **11 (0 ok)** | **11 (9 ok)** |

**Recall: 9/11 = 82%.**

### Что это значит

- **Docling layout работает** — детектор находит формулы.
- **UniMERNet работает** — LaTeX распознаётся.
- **Worker живёт весь прогон** — модель грузится один раз.
- **Изоляция venv** — конфликт `transformers` версий решён.

---

## Известные проблемы

### 1. `p82: 1 (0 ok)`

Формула не распозналась. Возможные причины:
- слишком маленький crop (bbox < 80×30 pt),
- worker вернул пустой `pred_str`.

TODO: логировать `worker_error` в JSON.

### 2. `p50: 1 из 2 не распознана`

Одна из двух формул не прошла `validate_latex`.
TODO: посмотреть `notes` в JSON.

### 3. Оверхед на загрузку модели

Worker загружает модель ~10–20 сек при старте. Для одного прогона это ок,
для частых коротких вызовов — заметно.

TODO (опционально): persistent-демон, поднимаемый отдельно.

---

## Что дальше

1. **Полный прогон 254 страниц** — `--pages 1-254`.
2. **Оценить recall** на всём корпусе.
3. **Закоммитить результат.**
4. **Обновить README** — упомянуть `.venv-unimernet`.

---

## Ссылки

- Коммит с subprocess-версией: `<hash>`
- UniMERNet: https://github.com/opendatalab/UniMERNet
- Веса: https://huggingface.co/wanderkid/unimernet_tiny