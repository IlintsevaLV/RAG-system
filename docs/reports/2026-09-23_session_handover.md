\# Handover: контекст сессии 2026-09-22 — 2026-09-23



\*\*Цель:\*\* передать контекст следующей сессии без потери деталей.



\---



\## 1. Проект



\*\*RAG-system\*\* — обработка PDF для RAG-индексации авиационных нормативов.

Два контура:



\- \*\*Контур 1\*\* (`run\_extract\_pages`) — текст страницы: TLQ, классы A/B/C/D, OCR, VLM.

\- \*\*Контур 2\*\* (`run\_regions`) — регионы: формулы (UniMERNet), таблицы (Docling/img2table), рисунки (VLM).



Подробное описание архитектуры: `docs/полный пайплайн.md`.



\*\*Стек:\*\*

\- Python 3.12, venv `.venv/`

\- UniMERNet в отдельном `.venv-unimernet/` (subprocess, transformers 4.42.4)

\- RapidOCR (3 головы: cyrillic + latin + greek) — ONNX, offline модели в `data/models/ocr/`

\- Docling, img2table, pylatexenc

\- VLM через llama-server (CPU, \~5 мин/страница) — \*\*отключён по умолчанию\*\*



\*\*Тестовый корпус:\*\*

\- `\_1954.pdf` (254 стр.) — сканы, class C, \*\*основной тест-документ\*\*

\- 30+ PDF в `data/raw/` — авиационные нормативы (нативные, сканы, смешанные)



\---



\## 2. Что уже сделано (Issues 1–10)



\### Issues 1–5 (коммиты `2e77c79`, `b3ca6a9`) — закрыты

\- Ложные таблицы на тексте (`region\_detect.py`)

\- OCR путает кириллицу и греческий в ячейках (`table\_pipeline.py`)

\- `max\_seq\_len` мал (проверили на рабочем ПК)

\- Дублирование formula+table (`region\_detect.py`)

\- Ложные figure на формулах (`region\_detect.py`)



\### Issue 6: `merge\_formula\_spans` (коммиты `e6f2da3`, `75247fa`, `36dc770`) — закрыт

\- Реализован строгий `is\_real\_formula`

\- Отсев OCR-каши

\- Zero-formula тесты (T7–T10) стали 0 формул



\### Коммит `c4542c2` — Rewrite merge\_formula\_spans

\- Восстановлен v2 single-span path

\- Добавлена пиксельная проверка черты дроби

\- Фикс `formula\_pipeline.py`: crop < 10 px не идёт в OpenCV

\- \*\*Результат:\*\* T4\_greek 0/12 → 11/11 ok (OpenCV crash закрыт)



\### Issue 7: Semantic filter (коммит `0b3f807`) — закрыт

\- `is\_real\_formula\_semantic()` в `formula\_pipeline.py` (правила A–H)

\- Новый `ExtractStatus.SUSPICIOUS\_SEMANTIC`

\- Контекст (`page\_figures`, `page\_tables`, `dpi`) передаётся через `process\_regions.py`

\- 23/23 unit-тестов

\- \*\*Результат:\*\* precision `ok` 65% → 90%, 0 мусора



\### Issue 8 (кодировка) — \*\*закрыт диагнозом\*\*

\- JSON корректен (UTF-8), кракозябры были в PowerShell 5

\- `UNREADABLE` в ячейках — реальный отказ OCR (не баг кодировки)

\- Остаток: retry OCR с upsample 2.5× для пустых ячеек



\### Issue 9 (прогресс-бар) — \*\*не делали\*\*, низкий приоритет



\### Issue 10: bbox формул режет длинные дроби — \*\*открыт, средний приоритет\*\*

\- Примеры: `M = Φ/Φ`, `q = 37,5 i`, `c\_τ = ds\_τ/ds`

\- Решение: расширять bbox вправо в `region\_detect`, пока продолжается черта



\### Issue 11 (новый, создан по итогам ночного прогона) — открыт, высокий приоритет

Три фикса:

\- \*\*A.\*\* `validate\_latex`: tolerant parsing \*\*до\*\* проверки braces (19 случаев `unbalanced\_braces` — баг)

\- \*\*B.\*\* `long\_letter\_run`: не `suspicious\_semantic`, а `caption\_or\_header` (32 заголовка теряются)

\- \*\*C.\*\* `figure\_caption`: gap < 15pt + проверка latex (2 ложных срабатывания)



Файл: `docs/issues/011\_semantic\_filter\_calibration.md`.



\---



\## 3. Ключевые тестовые прогоны



\### Page-tests v6 (`\_1954.pdf`, 10 тестов)



```

T1\_title          0 форм / 1 табл

T3\_equations      18 ok / 9 sem / 1 susp

T4\_greek          8 ok / 3 sem

T5\_figures        0 ok / 1 sem (регресс закрыт)

T6\_real\_tables    4 ok / 3 sem / tab=2

T7\_bibliography   0

T8\_toc            0

T9\_twocolumn      0

T10\_dense\_text    0

```



\### Ночной прогон (1451 стр., 9 PDF)



Результаты: `data/ir/night\_run\_20260923\_175717/summary.md`.



| Документ | Стр. | ok | sem | susp | tab | fig |

|---|---|---|---|---|---|---|

| rm178b\_full | 32 | 12 | 2 | 5 | 0 | 0 |

| kt178c\_full | 108 | 3 | 6 | 3 | 7 | 0 |

| kt160g\_book1\_full | 142 | 12 | 21 | 13 | 9 | 6 |

| ac29\_full | 1144 | 27 | 17 | 18 | 21 | 9 |

| \*\*Итого\*\* | \*\*1451\*\* | \*\*58\*\* | \*\*51\*\* | \*\*39\*\* | \*\*38\*\* | \*\*18\*\* |



\*\*Скорость:\*\*

\- class A (нативный): \*\*\~1 сек/стр.\*\*

\- class C (скан): \*\*\~20 сек/стр.\*\*

\- смешанный: \~2.5 сек/стр.

\- \*\*Всего \~2.5 часа на 1451 стр.\*\*



\*\*Fail: 0.\*\* Ни одного падения, ни одного OpenCV-краша.



\---



\## 4. Диагностика проблем (Issue 11)



\### A. `unbalanced\_braces` — 19 случаев

Баг `validate\_latex`: балансировка скобок отсеивает LaTeX \*\*до\*\* tolerant parsing.

Пример: `\\begin{array}{l}{{\\bf\\Theta\_{\\mathrm{{T}}\\mathrm{{e y c h}\\mathrm{{H e}...` (не хватает `}`).



\*\*Фикс:\*\* попробовать `LatexWalker(tolerant\_parsing=True)` \*\*до\*\* проверки баланса.



\### B. `long\_letter\_run` — 39 случаев

Правило правильно отсеивает OCR-кашу (`TyIIdTe`, `KHHTAI`, `noxenHX`), но это

\*\*заголовки и подписи\*\*, а не формулы.



\*\*Фикс:\*\* ввести `ExtractStatus.CAPTION\_OR\_HEADER`, сохранять `latex` для RAG.



\### C. `figure\_caption` — 4 случая, 2 ложных

Пример: `5Σ+` рядом с figure, но не подпись.



\*\*Фикс:\*\* `gap < 15pt` + проверка `long\_letter\_run` в latex.



\### D. `no\_math\_signal`, `prose\_like` (20 случаев) — ✅ правильно

`validate\_latex` корректно отсеивает прозу без math-сигналов.



\---



\## 5. Что делать дальше (приоритеты)



\### Сразу

1\. \*\*Коммит Issue 11\*\* + ночной summary в `docs/reports/`.

2\. \*\*Разработчик берётся за Issue 11\*\* (три фикса).

3\. \*\*Повторный ночной прогон\*\* после Issue 11 → сверить с ожиданием:

&#x20;  - `ok: 58 → \~70`

&#x20;  - `sem: 51 → \~30`

&#x20;  - `susp: 39 → \~15`



\### Средний приоритет

4\. \*\*Issue 10\*\* (bbox формул) — расширять bbox вправо.

5\. \*\*Issue 8 остаток\*\* — retry OCR для `UNREADABLE` ячеек.



\### Дальше

6\. \*\*Полный прогон `\_1954.pdf`\*\* (254 стр.) — понять масштаб.

7\. \*\*Расширить корпус\*\* — прогнать все 30+ PDF.

8\. \*\*Перейти к контуру 1\*\* — если контур 2 стабилен.



\---



\## 6. Ключевые файлы и команды



\### Git

\- Репозиторий: `https://github.com/IlintsevaLV/RAG-system`

\- HEAD: `0b3f807` (semantic filter)

\- Основные коммиты: `c4542c2` (merge\_formula\_spans), `0b3f807` (semantic filter)



\### Env

```powershell

cd C:\\Users\\l.ilyintseva\\Desktop\\RAG-system

$env:PYTHONIOENCODING = "utf-8"

$env:PYTHONUTF8 = "1"

$env:ENABLE\_VLM = "false"          # VLM на CPU 5 мин/стр, отключён

$env:ENABLE\_UNIMERNET = "true"

$env:UNIMERNET\_CONFIG\_PATH = ".\\data\\models\\unimernet\\unimernet\_tiny.yaml"

$PythonMain = ".\\.venv\\Scripts\\python.exe"

```



\### Основные команды

```powershell

\# Detection-only (быстро, без UniMERNet)

\& $PythonMain scripts\\check\_formula\_detection.py "data\\raw\\\_1954.pdf" --out "...json"



\# Unit-тесты semantic filter

\& $PythonMain scripts\\check\_semantic\_filter.py



\# Полный прогон run\_regions

\& $PythonMain -m ingestion.run\_regions "data\\raw\\\_1954.pdf" `

&#x20; --pages 34,36,37,40,41,44 `

&#x20; --out-dir "data\\ir\\page\_tests\\T3\_equations\_v7" `

&#x20; --enable-unimernet

```



\### Структура JSON регионов

```json

{

&#x20; "doc\_id": "\_1954",

&#x20; "pages": \[{

&#x20;   "page": 34,

&#x20;   "blocks": \[{

&#x20;     "type": "formula",

&#x20;     "region\_id": "formula\_00",

&#x20;     "bbox": {"x1": ..., "y1": ..., "x2": ..., "y2": ...},

&#x20;     "content": {

&#x20;       "latex": "...",

&#x20;       "status": "ok|suspicious|suspicious\_semantic|caption\_or\_header|failed"

&#x20;     },

&#x20;     "quality": {"notes": \["semantic\_reason=...", "preview\_wh=..."]},

&#x20;     "provenance": {"method": "unimernet|unimernet\_retry", "confidence": 0.85}

&#x20;   }]

&#x20; }]

}

```



\---



\## 7. Соглашения



\- VLM на CPU — 5 мин/стр, включать только для отдельных страниц.

\- UniMERNet в отдельном `.venv-unimernet/` (subprocess, persistent worker).

\- OCR-модели offline в `data/models/ocr/` (`latin\_\*`, `el\_\*`, `cyrillic\_\*`).

\- Кодировка: `PYTHONIOENCODING=utf-8`, `PYTHONUTF8=1`.

\- PowerShell 5 искажает вывод — использовать `\[Console]::OutputEncoding = \[Text.Encoding]::UTF8`.



\---



\## 8. Открытые вопросы



1\. \*\*`suspicious` без причины в старом диагностическом скрипте\*\* — это был баг скрипта, не кода. Причины есть (`unbalanced\_braces`, `no\_math\_signal`, `prose\_like`).

2\. \*\*`UNREADABLE` в таблицах\*\* — реальный отказ OCR, не кодировка.

3\. \*\*VLM recovery rate\*\* — не измерено в этом прогоне, требует GPU для оценки.



\---



\## 9. Контакты / ресурсы



\- Разработчик: `IlintsevaLV` (коммиты на GitHub)

\- Документация архитектуры: `docs/полный пайплайн.md`

\- Issues: `docs/issues/001\_\*.md` ... `docs/issues/011\_\*.md`

\- Отчёты: `docs/reports/2026-09-2\*.md`



\---



\*\*Конец handover. В новой сессии начать с `git pull` и чтения этого файла.\*\*

