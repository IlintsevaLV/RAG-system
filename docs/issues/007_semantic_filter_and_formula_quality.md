\# Issue 7: Semantic filter для формул + качество детекции



\*\*Дата:\*\* 2026-09-23  

\*\*Статус:\*\* открыт  

\*\*Приоритет:\*\* высокий  

\*\*Автор:\*\* Ильинцева Л.В.  

\*\*Связанные коммиты:\*\* `c4542c2` (Rewrite merge\_formula\_spans), `36dc770` (Reject OCR math garbage)



\---



\## 1. Контекст



После серии фиксов в контуре 2 (`run\_regions`) достигнуты значительные улучшения:



| Что | Коммит | Статус |

|---|---|---|

| Zero-formula тесты (T7–T10) | `75247fa`, `36dc770` | ✅ закрыто |

| OpenCV crash на crop < 10 px | `c4542c2` | ✅ закрыто |

| Merge formula fragments | `c4542c2` | ✅ частично |



\*\*OpenCV crash полностью ушёл.\*\* На T4\_greek было `0/12 ok`, стало `11/11 ok`. На T3\_equations: `19/27` → `27/28`. На T6\_real\_tables: `4/7` → `7/7`.



\*\*НО\*\* осталась системная проблема: `status=ok` \*\*не значит «настоящая формула»\*\*. UniMERNet успешно распознаёт \*\*любой\*\* crop — включая мусор, подписи к рисункам и числа из таблиц. `validate\_latex` проверяет только синтаксис.



\---



\## 2. Что показали тесты (2026-09-23)



\### Сводная таблица



| Тест | v5 (`36dc770`) | final (`c4542c2`) | Δ ok | Precision `ok` |

|---|---|---|---|---|

| T1\_title | 0 форм / 1 табл | 0 форм / 1 табл | — | — |

| T2\_table\_typo | 0 форм / 1 табл | (не перезапускался) | — | — |

| \*\*T3\_equations\*\* | 27 дет / 19 ok | \*\*28 дет / 27 ok\*\* | +8 | \*\*\~63%\*\* |

| \*\*T4\_greek\*\* | 12 дет / \*\*0 ok\*\* | \*\*11 дет / 11 ok\*\* | \*\*+11\*\* | \*\*\~73%\*\* |

| \*\*T5\_figures\*\* | 6 дет / 4 ok | \*\*1 дет / 1 ok\*\* | -5 дет | \*\*0%\*\* |

| \*\*T6\_real\_tables\*\* | 7 дет / 4 ok | 7 дет / \*\*7 ok\*\* | +3 | \*\*\~71%\*\* |

| T7\_bibliography | 0 форм | 0 форм | — | — |

| T8\_toc | 0 форм | 0 форм | — | — |

| T9\_twocolumn | 0 форм | 0 форм | — | — |

| T10\_dense\_text | 0 форм | 0 форм | — | — |



\*\*Precision `ok`\*\* — доля настоящих формул среди `ok`-детекций, оценено вручную по PDF `\_1954.pdf`.



\*\*Вывод:\*\* в RAG-индекс попадает \*\*\~30% мусора\*\* среди формул.



\---



\## 3. Примеры мусора, проходящего как `status=ok`



\### 3.1. OCR-каша (single-span path)



| Тест | Стр. | LaTeX / текст | Проблема |

|---|---|---|---|

| T3 | 34 | `\\begin{array}{l}{T y I I d T e . i l b H z d . \~ c K o p o c T b \~}` | 4+ буквы подряд без math |

| T3 | 41 | `-\\simeq\_∘, \\dot{Π} OJyqHM` | номер формулы OCR-нут как формула |

| T4 | 47 | `\\stackrel{·}{f}^B \\stackrel{\_}{r}` | OCR-мусор |

| T4 | 48 | `\\frac{"}{e\_r^{α\_ω} R}` | обрубок |

| T6 | 61 | `c\_x = -\\underbrace{75N}\_2` | склейка из двух OCR-строк |

| T6 | 68 | `\\frac{G=0,05}{1}` | обрубок |



\### 3.2. Подписи к рисункам и графикам



| Тест | Стр. | Текст | Что это |

|---|---|---|---|

| \*\*T5\*\* | \*\*25\*\* | `\\begin{array}{rl}\&{\\qquad\\qquad\\qquad\\quad\\cdots^{mm\\times n...` | \*\*ложная формула\*\* на подписи к Фиг. I.18/I.19 |

| T3 | 36 | `.μ = 0,75 (хороший несущий винт) \\| μ =0,5 (ппохой несущий винт)` | подписи к кривым Фиг. II.3 |



\### 3.3. Текст, случайно похожий на формулу



| Тест | Стр. | Текст | Что это |

|---|---|---|---|

| T3 | 44 | `\\vert n p u \\ N = 2 0 0 \\pi \\ c \\vert \\sum (C=12Z5×z)` | \*\*«при N = 200…»\*\* — обычный текст |

| T3 | 44 | `(N=200000, C=1225 KZ)` | то же, вторая строка |



\### 3.4. Значения из таблиц



| Тест | Стр. | Текст | Что это |

|---|---|---|---|

| T6 | 68 | `6 = 0,06` | число из таблицы |

| T6 | 68 | `mk=0,00052 \\| mx=0,00088` | значения из таблицы |



\### 3.5. Битая кодировка в таблицах (отдельная проблема)



| Тест | Стр. | Как выглядит | Должно быть |

|---|---|---|---|

| T1 | 3 | `РқР°РҝРөСҮР°СӮР°РҪРҫ` | `Напечатано` |

| T6 | 61 | `в•ЁРӘв•ӨРҗв•ӨР“в•ӨР’в•Ёв•‘в•Ёв–‘` | `Крутка` |

| T6 | 61 | `UNREADABLE` | (OCR не прочитал) |

| T6 | 189 | `в•ЁРів•ӨР‘в•Ёв•—в•ӨРӣ...` | `Условия работы` |



\---



\## 4. Что должен сделать разработчик



\### 4.1. Добавить `is\_real\_formula\_semantic()` в `formula\_pipeline.py`



Фильтр вызывается \*\*после\*\* `validate\_latex`, \*\*до\*\* записи в JSON.



\*\*Сигнатура:\*\*



```python

def is\_real\_formula\_semantic(

&#x20;   latex: str,

&#x20;   \*,

&#x20;   bbox\_pt: tuple\[float, float, float, float],

&#x20;   page\_figures: list\[tuple\[float, float, float, float]],

&#x20;   page\_tables: list\[tuple\[float, float, float, float]],

&#x20;   source: str,          # "unimernet" | "unimernet\_retry" | "vlm\_recovery"

&#x20;   raw\_crop\_wh: tuple\[int, int],   # размер crop'а в пикселях

) -> tuple\[bool, str]:

&#x20;   """Возвращает (True, "") для настоящей формулы,

&#x20;   (False, reason) для мусора."""

```



\*\*Правила отклонения (вернуть `False`):\*\*



\*\*A. OCR-каша:\*\* 4+ буквы подряд в любом script без math-контекста.



```python

import re

\# Латинские/кириллические буквы, идущие подряд 4+ символа

\# НЕ внутри \\cmd

\_CMD\_RE = re.compile(r"\\\\\[a-zA-Z]+")

def \_strip\_cmds(text: str) -> str:

&#x20;   return \_CMD\_RE.sub("", text)



if re.search(r"\[A-Za-zА-Яа-яЁё]{4,}", \_strip\_cmds(latex)):

&#x20;   return False, "long\_letter\_run"

```



Исключения: `\\theta`, `\\alpha`, `\\frac`, `\\sqrt` — они удалены `\_strip\_cmds`.



\*\*B. Русский текст:\*\* 2+ русских слова длиной >3.



```python

if re.search(r"\[А-Яа-яЁё]{3,}\\s+\[А-Яа-яЁё]{3,}", latex):

&#x20;   return False, "russian\_words"

```



\*\*C. Обрубок без оператора:\*\* нет math-маркеров вообще.



```python

\_MATH\_MARKERS = re.compile(

&#x20;   r"\[=+\\-\*/^\_<>≤≥≠≈]"

&#x20;   r"|\\\\frac|\\\\sqrt|\\\\int|\\\\sum|\\\\prod|\\\\lim|\\\\partial|\\\\infty"

&#x20;   r"|\\\\alpha|\\\\beta|\\\\gamma|\\\\theta|\\\\lambda|\\\\mu|\\\\pi|\\\\sigma|\\\\omega"

&#x20;   r"|\\\\kappa|\\\\rho|\\\\hbar|\\\\phi|\\\\Delta"

)

if not \_MATH\_MARKERS.search(latex):

&#x20;   return False, "no\_math\_markers"

```



\*\*D. Паттерн «число = число»:\*\* значения из таблиц.



```python

\_stripped = latex.replace(" ", "").replace("{", "").replace("}", "")

if re.fullmatch(r"\[0-9.,=×xX;:%]+", \_stripped):

&#x20;   return False, "pure\_numbers"

```



\*\*E. Паттерн номера формулы:\*\* `(I.1)`, `(III.7)`, `(II.23)`, `(17.11)` — это

не формулы, а их номера, OCR-нутые отдельно.



```python

if re.fullmatch(r"\\(\\s\*\[IVX]+\\.\\s\*\\d+\\s\*\\)", latex.replace(" ", "")):

&#x20;   return False, "formula\_number"



if re.fullmatch(r"\\(\\s\*\\d+\\.\\d+\\s\*\\)", latex.replace(" ", "")):

&#x20;   return False, "formula\_number"

```



\*\*F. Подпись к рисунку:\*\* bbox рядом с figure снизу.



```python

def \_is\_near\_figure(bbox, figures, \*, max\_dist\_pt=30, overlap\_frac=0.3):

&#x20;   x1, y1, x2, y2 = bbox

&#x20;   for fx1, fy1, fx2, fy2 in figures:

&#x20;       # Формула снизу от рисунка, на расстоянии < max\_dist\_pt

&#x20;       if y1 >= fy2 - 5 and y1 <= fy2 + max\_dist\_pt:

&#x20;           # X-диапазоны перекрываются хотя бы на overlap\_frac

&#x20;           ox = max(0, min(x2, fx2) - max(x1, fx1))

&#x20;           width = max(1, min(x2 - x1, fx2 - fx1))

&#x20;           if ox / width >= overlap\_frac:

&#x20;               return True

&#x20;   return False



if \_is\_near\_figure(bbox\_pt, page\_figures):

&#x20;   return False, "figure\_caption"

```



\*\*G. Значение внутри таблицы:\*\* bbox внутри table + нет math-операторов.



```python

def \_inside\_any(bbox, regions, \*, min\_share=0.7):

&#x20;   x1, y1, x2, y2 = bbox

&#x20;   for rx1, ry1, rx2, ry2 in regions:

&#x20;       ox = max(0, min(x2, rx2) - max(x1, rx1))

&#x20;       oy = max(0, min(y2, ry2) - max(y1, ry1))

&#x20;       if ox <= 0 or oy <= 0:

&#x20;           continue

&#x20;       area = max(1, (x2 - x1) \* (y2 - y1))

&#x20;       if (ox \* oy) / area >= min\_share:

&#x20;           return True

&#x20;   return False



if \_inside\_any(bbox\_pt, page\_tables) and not \_MATH\_MARKERS.search(latex):

&#x20;   return False, "table\_value"

```



\*\*H. Слишком маленький crop:\*\* если `raw\_crop\_wh` < 20×8 px — вряд ли настоящая формула.



```python

w, h = raw\_crop\_wh

if w < 20 or h < 8:

&#x20;   return False, "tiny\_crop"

```



\### 4.2. Ввести `status=suspicious\_semantic`



Не удалять мусор — пометить. Поле `content.status` в JSON:



| Значение | Смысл |

|---|---|

| `ok` | прошёл синтаксис + семантику |

| `suspicious\_semantic` | валидный LaTeX, но не похож на формулу |

| `suspicious` | не прошёл `validate\_latex` |

| `failed` | не удалось распознать |



Добавить в `ingestion/models.py`:



```python

class ExtractStatus(str, Enum):

&#x20;   OK = "ok"

&#x20;   SUSPICIOUS = "suspicious"

&#x20;   SUSPICIOUS\_SEMANTIC = "suspicious\_semantic"  # NEW

&#x20;   FAILED = "failed"

&#x20;   NEEDS\_VLM = "needs\_vlm"

```



В `quality.notes` сохранить `semantic\_reason`, например:

```json

"notes": \["semantic\_reason=long\_letter\_run", "preview\_wh=180x22"]

```



\### 4.3. Передавать контекст в фильтр



В `process\_regions.py` при вызове `process\_formula\_region` собрать:



```python

\# Собрать bbox всех figure и table регионов на странице

page\_figures = \[r.bbox\_pt\_tuple() for r in regions if r.type == BlockType.FIGURE]

page\_tables  = \[r.bbox\_pt\_tuple() for r in regions if r.type == BlockType.TABLE]

```



И передать в `formula\_pipeline.process\_formula\_region(...)`.



\### 4.4. Тесты для фильтра



Добавить `scripts/check\_semantic\_filter.py` (или расширить `check\_formula\_detection.py`).



\*\*Набор тестовых случаев:\*\*



| Вход (LaTeX) | Контекст | Ожидаемо |

|---|---|---|

| `= \\frac{Tv}{75N}` | bbox в тексте | `ok` |

| `dM\_κ = k·½·ρ(vr)²b(c\_xp+βc\_y)r dr` | bbox в тексте | `ok` |

| `(V\_y/2 + σ\_a ωR/16)(...)` | bbox в тексте | `ok` |

| `m\_κ = m\_κp + m\_κi` | bbox в тексте | `ok` |

| `q = G/N = 1700/170 = 10` | bbox в тексте | `ok` |

| `\\begin{array}{l}{T y I I d T e . i l b H z d . \~ c K o p o c T b \~}` | bbox в тексте | `suspicious\_semantic` (`long\_letter\_run`) |

| `\\vert n p u \\ N = 2 0 0` | bbox в тексте | `suspicious\_semantic` (`long\_letter\_run`) |

| `\\begin{array}{rl}\&{\\qquad\\qquad...` | bbox \*\*под figure\*\* | `suspicious\_semantic` (`figure\_caption`) |

| `6 = 0,06` | bbox \*\*внутри table\*\* | `suspicious\_semantic` (`table\_value`) |

| `(I.7)` | bbox в тексте | `suspicious\_semantic` (`formula\_number`) |

| `\\frac{"}{e\_r^{α\_ω} R}` | bbox в тексте | `suspicious\_semantic` (`no\_math\_markers`) |

| `M = -` | bbox в тексте | `suspicious\_semantic` (`no\_math\_markers`) |



\---



\## 5. Ожидаемый результат после фикса



| Тест | Было (final) | Цель (semantic) |

|---|---|---|

| T1\_title | 0 форм / 1 табл | 0 форм / 1 табл |

| T3\_equations | 28 дет / 27 ok | 28 дет / \*\*\~17 ok\*\*, \~11 `suspicious\_semantic` |

| T4\_greek | 11 дет / 11 ok | 11 дет / \*\*\~8 ok\*\*, \~3 `suspicious\_semantic` |

| \*\*T5\_figures\*\* | \*\*1 дет / 1 ok\*\* | \*\*0 дет / 0 ok\*\*, figure=2 |

| T6\_real\_tables | 7 дет / 7 ok | 7 дет / \*\*\~5 ok\*\*, \~2 `suspicious\_semantic` |

| T7–T10 | 0 форм | 0 форм |



\*\*Precision `ok` → 95%+.\*\*



\---



\## 6. Как проверить



```powershell

cd C:\\Users\\l.ilyintseva\\Desktop\\RAG-system

git pull



$env:ENABLE\_VLM = "false"

$env:ENABLE\_UNIMERNET = "true"

$env:UNIMERNET\_CONFIG\_PATH = ".\\data\\models\\unimernet\\unimernet\_tiny.yaml"

$PythonMain = ".\\.venv\\Scripts\\python.exe"



\# T3 — много настоящих формул + мусор

\& $PythonMain -m ingestion.run\_regions "data\\raw\\\_1954.pdf" `

&#x20; --pages 34,36,37,40,41,44 `

&#x20; --out-dir "data\\ir\\page\_tests\\T3\_equations\_semantic" `

&#x20; --enable-unimernet



\# T4 — греческие формулы

\& $PythonMain -m ingestion.run\_regions "data\\raw\\\_1954.pdf" `

&#x20; --pages 47,48,51 `

&#x20; --out-dir "data\\ir\\page\_tests\\T4\_greek\_semantic" `

&#x20; --enable-unimernet



\# T5 — регресс: ложная формула на подписи

\& $PythonMain -m ingestion.run\_regions "data\\raw\\\_1954.pdf" `

&#x20; --pages 9,11,15,20,25 `

&#x20; --out-dir "data\\ir\\page\_tests\\T5\_figures\_semantic" `

&#x20; --enable-unimernet



\# T6 — таблицы + формулы

\& $PythonMain -m ingestion.run\_regions "data\\raw\\\_1954.pdf" `

&#x20; --pages 61,68,69,189 `

&#x20; --out-dir "data\\ir\\page\_tests\\T6\_real\_tables\_semantic" `

&#x20; --enable-unimernet

```



\*\*Проверка JSON:\*\*



```powershell

\& $PythonMain -c @"

import json

from pathlib import Path



tests = \[

&#x20;   'T3\_equations\_semantic',

&#x20;   'T4\_greek\_semantic',

&#x20;   'T5\_figures\_semantic',

&#x20;   'T6\_real\_tables\_semantic',

]

for t in tests:

&#x20;   p = Path(f'data/ir/page\_tests/{t}/\_1954\_regions.json')

&#x20;   if not p.is\_file():

&#x20;       print(f'{t}: NO JSON'); continue

&#x20;   d = json.load(open(p, encoding='utf-8'))

&#x20;   n\_ok = n\_sem = n\_susp = n\_fail = 0

&#x20;   for pg in d\['pages']:

&#x20;       for b in pg\['blocks']:

&#x20;           if b\['type'] != 'formula':

&#x20;               continue

&#x20;           st = b\['content'].get('status')

&#x20;           if st == 'ok': n\_ok += 1

&#x20;           elif st == 'suspicious\_semantic': n\_sem += 1

&#x20;           elif st == 'suspicious': n\_susp += 1

&#x20;           elif st == 'failed': n\_fail += 1

&#x20;   print(f'{t}: ok={n\_ok} sem={n\_sem} susp={n\_susp} fail={n\_fail}')

"@

```



\*\*Ожидаемый вывод:\*\*



```

T3\_equations\_semantic: ok=17 sem=11 susp=0 fail=0

T4\_greek\_semantic:     ok=8  sem=3  susp=0 fail=0

T5\_figures\_semantic:   ok=0  sem=1  susp=0 fail=0

T6\_real\_tables\_semantic: ok=5 sem=2 susp=0 fail=0

```



\---



\## 7. Файлы для изменения



| Файл | Что |

|---|---|

| `ingestion/formula\_pipeline.py` | `is\_real\_formula\_semantic()` + вызов после `validate\_latex` |

| `ingestion/models.py` | добавить `ExtractStatus.SUSPICIOUS\_SEMANTIC` |

| `ingestion/process\_regions.py` | собрать `page\_figures`, `page\_tables`, передать в formula\_pipeline |

| `scripts/check\_semantic\_filter.py` | (новый) unit-тесты правил A–H |

| `docs/reports/2026-09-23\_semantic\_filter.md` | (новый) отчёт после реализации |



\---



\## 8. Что НЕ входит в этот issue (отдельные задачи)



\### Issue 8: битая кодировка таблиц



`UNREADABLE` и `в•ЁРӘв•ӨРҗ...` — это \*\*неправильная кодировка\*\* (UTF-8 vs Windows-1251).

Нужно:



\- найти источник в `table\_pipeline.py` (`str.encode("cp1251").decode("utf-8")` или наоборот);

\- убедиться, что все `json.dump` используют `ensure\_ascii=False, encoding="utf-8"`;

\- заменить `UNREADABLE` на `suspicious` + OCR retry с upsample 2.5×.



Примеры из тестов 2026-09-23:



```

T1 p3:  'РқР°РҝРөСҮР°СӮР°РҪРҫ'  → должно быть 'Напечатано'

T6 p61: 'в•ЁРӘв•ӨРҗв•ӨР“в•ӨР’в•Ёв•‘в•Ёв–‘' → должно быть 'Крутка'

T6 p61: 'UNREADABLE' → это ячейка, OCR не смог

```



\### Issue 9: прогресс-бар для длинных прогонов



Сейчас `run\_regions` на 10 страницах идёт 3–4 минуты и печатает `regions\_page` в лог

с tqdm-стилем. Для массовых прогонов (254 страницы) нужен \*\*единый progress bar\*\* с

ETA.



\---



\## 9. Итог



| Подсистема | Статус |

|---|---|

| OpenCV crash (crop < 10 px) | ✅ \*\*закрыто\*\* (T4: 0/12 → 11/11) |

| Merge formula fragments | ✅ \*\*закрыто\*\* (T5: 6 → 1 детекция) |

| Zero-formula регресс | ✅ \*\*закрыто\*\* (T7–T10 = 0) |

| \*\*Semantic filter формул\*\* | ❌ \*\*этот issue\*\* |

| Подписи к рисункам как формулы | ❌ этот issue (правило F) |

| Значения таблиц как формулы | ❌ этот issue (правило G) |

| Битая кодировка таблиц | ❌ Issue 8 |

| `UNREADABLE` в таблицах | ❌ Issue 8 |



\*\*Precision `ok` сейчас \~65–70%. Цель — 95%+.\*\*

