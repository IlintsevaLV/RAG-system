\# Issue 11: Калибровка validate\_latex и semantic filter по результатам ночного прогона



\*\*Дата:\*\* 2026-09-23  

\*\*Статус:\*\* открыт  

\*\*Приоритет:\*\* высокий  

\*\*Автор:\*\* Ильинцева Л.В.  

\*\*Связан с:\*\* \[#007](./007\_semantic\_filter\_and\_formula\_quality.md)  

\*\*Источник данных:\*\* ночной прогон `data/ir/night\_run\_20260923\_175717/`



\---



\## Контекст



После реализации semantic filter (Issue 7, коммит `0b3f807`) был проведён ночной

прогон `run\_regions` по 9 PDF разного типа — всего \*\*1451 страница\*\* без VLM,

с UniMERNet. Прогон занял \~2.5 часа, \*\*0 падений\*\*, но вскрыл \*\*две калибровочные

проблемы\*\*:



| Метрика | Значение |

|---|---|

| Всего страниц | 1451 |

| `formula ok` | 58 (39%) |

| `formula suspicious\_semantic` | 51 (34%) |

| `formula suspicious` | 39 (27%) |

| `fail` | 0 ✅ |



\*\*34% + 27% = 61% формул не попадают в RAG.\*\* Это очень много. Диагностика

показала: большая часть отсеивается \*\*неправильно\*\* или \*\*должна сохраняться

как текст\*\*, а не как формула.



\---



\## Сводная таблица прогона



| Документ | Стр. | ok | sem | susp | tab | fig |

|---|---|---|---|---|---|---|

| `rm178b\_full` (native A) | 32 | 12 | 2 | 5 | 0 | 0 |

| `kt178c\_full` (scan C) | 108 | 3 | 6 | 3 | 7 | 0 |

| `kt160g\_book1\_full` (scan C) | 142 | 12 | 21 | 13 | 9 | 6 |

| `ac29\_full` (mixed A+C) | 1144 | 27 | 17 | 18 | 21 | 9 |

| \*\*Итого (по всем)\*\* | \*\*1451\*\* | \*\*58\*\* | \*\*51\*\* | \*\*39\*\* | \*\*38\*\* | \*\*18\*\* |



\---



\## Диагностика причин `suspicious\_semantic` (sem)



```

long\_letter\_run           39    ← 🚨 главная проблема

figure\_caption             4

prose\_source\_text          3

no\_math\_markers            2

truncated                  2

numeric\_value              1

```



\## Диагностика причин `suspicious` (susp)



```

unbalanced\_braces         19    ← 🚨 баг validate\_latex

no\_math\_signal            16    ← ✅ правильно (не формулы)

prose\_like                 4    ← ✅ правильно

```



\---



\## Проблема A: `validate\_latex` отсеивает LaTeX с `unbalanced\_braces`



\### Что происходит



\*\*19 случаев из 90\*\* — LaTeX с недостающими 1-2 закрывающими скобками во вложенных `\\mathrm{}` или `\\mathbf{}`. Это UniMERNet \*\*сгенерировал битый LaTeX\*\* на сложных кириллических сканах.



Пример из `kt160g\_book1\_full`:

```

p055 latex: '\\begin{array}{l}{{\\bf\\Theta\_{\\mathrm{{T}}\\mathrm{{e y c h}\\mathrm{{H e}\\ c...'

```



Считаем скобки:

\- `{l}` — 0

\- `{{\\bf\\Theta\_{\\mathrm{{T}}` — +2

\- `\\mathrm{{e y c h}` — +1 (`{` открыт, но второй `}` отсутствует)

\- \*\*баланс не сходится\*\*



\### Текущее поведение (`formula\_pipeline.py:66-77`)



```python

def validate\_latex(latex: str) -> tuple\[bool, list\[str]]:

&#x20;   notes: list\[str] = \[]

&#x20;   if not latex or len(latex) < 2:

&#x20;       return False, \["empty"]

&#x20;   # balanced braces

&#x20;   bal = 0

&#x20;   for ch in latex:

&#x20;       if ch == "{":

&#x20;           bal += 1

&#x20;       elif ch == "}":

&#x20;           bal -= 1

&#x20;           if bal < 0:

&#x20;               return False, \["unbalanced\_braces"]   # ← выходим здесь

&#x20;   if bal != 0:

&#x20;       return False, \["unbalanced\_braces"]            # ← и здесь

&#x20;   # ...

&#x20;   # try pylatexenc if available

&#x20;   try:

&#x20;       from pylatexenc.latexwalker import LatexWalker

&#x20;       LatexWalker(latex, tolerant\_parsing=True).get\_latex\_nodes()  # ← tolerant parsing!

```



\*\*Баг:\*\* `balanced braces` проверка возвращает `False` \*\*до\*\* того, как

попробуется `LatexWalker(tolerant\_parsing=True)`. Но `tolerant\_parsing=True`

как раз для того и предназначен, чтобы \*\*чинить\*\* битый LaTeX.



\### Фикс



Не возвращать `False` сразу при `unbalanced\_braces`. Сначала \*\*попробовать\*\*

толерантный парсер:



```python

def validate\_latex(latex: str) -> tuple\[bool, list\[str]]:

&#x20;   notes: list\[str] = \[]

&#x20;   if not latex or len(latex) < 2:

&#x20;       return False, \["empty"]



&#x20;   # Balanced braces: не блокирующее условие, а сигнал для tolerant parsing

&#x20;   braces\_ok = True

&#x20;   bal = 0

&#x20;   for ch in latex:

&#x20;       if ch == "{":

&#x20;           bal += 1

&#x20;       elif ch == "}":

&#x20;           bal -= 1

&#x20;           if bal < 0:

&#x20;               braces\_ok = False

&#x20;               break

&#x20;   if bal != 0:

&#x20;       braces\_ok = False



&#x20;   # Prose check (unchanged)

&#x20;   words = \_LATEX\_WORD.findall(latex)

&#x20;   has\_signal = bool(\_LATEX\_MATH\_SIGNAL.search(latex))

&#x20;   if not has\_signal and len(words) >= 2:

&#x20;       return False, \["no\_math\_signal", "prose\_like"]

&#x20;   if len(words) >= 5 and not any(

&#x20;       token in latex for token in ("\\\\frac", "\\\\sqrt", "\\\\sum", "\\\\int", "^", "\_")

&#x20;   ):

&#x20;       return False, \["prose\_like"]



&#x20;   # Try tolerant parse regardless of braces balance

&#x20;   try:

&#x20;       from pylatexenc.latexwalker import LatexWalker

&#x20;       LatexWalker(latex, tolerant\_parsing=True).get\_latex\_nodes()

&#x20;       if braces\_ok:

&#x20;           notes.append("pylatexenc\_ok")

&#x20;       else:

&#x20;           notes.append("braces\_repaired")

&#x20;       return True, notes

&#x20;   except ImportError:

&#x20;       notes.append("pylatexenc\_not\_installed")

&#x20;       if any(tok in latex for tok in ("=", "\_", "^", "\\\\frac", "\\\\sum", "\\\\int", "+", "-")):

&#x20;           return True, notes + \["heuristic\_ok"]

&#x20;       return braces\_ok and len(latex) >= 3, notes + \["heuristic\_weak"]

&#x20;   except Exception as exc:

&#x20;       return False, \[f"pylatexenc\_fail:{exc}"]

```



\*\*Ожидаемый эффект:\*\* \~15–19 случаев `unbalanced\_braces` перейдут в `status=ok`

с note `braces\_repaired=true`. RAG получит формулы, которые сейчас теряются.



\---



\## Проблема B: `long\_letter\_run` не различает OCR-кашу и заголовки



\### Что происходит



\*\*39 случаев\*\* (76% от всех `suspicious\_semantic`) — это \*\*заголовки\*\*, OCR-нутые

греческим head'ом:



```

p004 src: 'text=KT 160Γ/14Γ'                        ← заголовок «КТ-160Г/14Г»

p011 src: 'text=ΚΗΑΓΑ Ι. Ο6μΜΕ ποπΟΣΚεΗμα,'         ← «КНИГА I. ОБЩИЕ ПОЛОЖЕНИЯ»

p046 src: 'text=минимум 10 Σ/мин.'                  ← «минимум 10°С/мин»

p082 src: 'text=UcπβΙταΗμe 1: φ = 11 Γu | φ2 = ...' ← «Испытание 1: φ = 11 Гц...»

```



После удаления команд, `\\mathrm{}` и пробелов получаются длинные последовательности

букв: `KHHTAI`, `noxenHX`, `TyIIdTe`, `cKopocTb`. `\_SEM\_LETTER\_RUN` их ловит

и помечает как мусор.



\*\*Но это не мусор\*\* — это \*\*полезный текст\*\*, который должен попасть в RAG

(пусть не как формула, а как \*\*заголовок\*\* или \*\*подпись\*\*).



\### Диагностика



```python

>>> \_latex\_letter\_runs(r'\\begin{array}{l}{{\\mathrm{T y I I d T e . i l b H z d . \~ c K o p o c T b\~}}}')

\['TyIIdTe', 'ilbHzd', 'cKopocTb']       ← правильно, это OCR-каша



>>> \_latex\_letter\_runs(r'\\mathbf{K H H T A \\ I . \\ O 6 m u e \\Pi 0 . n o x e n H X}')

\['KHHTAI', 'noxenHX']                    ← правильно, это «КНИГА I» OCR-каша



>>> \_latex\_letter\_runs(r'\\begin{array}{c}{\\mathrm{KT}\\ \\ 1 6 0 \\mathrm{G}/1 4 \\mathrm{G}}')

\[]                                       ← правильно, это ничего не даёт

```



\*\*Функция работает корректно.\*\* Проблема в том, \*\*что делается со результатом\*\*.



\### Фикс



Ввести новый статус `CAPTION\_OR\_HEADER` в `ExtractStatus`:



```python

class ExtractStatus(str, Enum):

&#x20;   OK = "ok"

&#x20;   SUSPICIOUS = "suspicious"

&#x20;   SUSPICIOUS\_SEMANTIC = "suspicious\_semantic"

&#x20;   CAPTION\_OR\_HEADER = "caption\_or\_header"       # ← NEW

&#x20;   FAILED = "failed"

&#x20;   NEEDS\_VLM = "needs\_vlm"

```



В `is\_real\_formula\_semantic` при `long\_letter\_run`, \*\*если внутри latex нет

math-операторов\*\* (`=`, `+`, `\\frac`, `\\sum`, `^`, `\_`), возвращать

\*\*не\*\* `False, "long\_letter\_run"`, а специальный маркер:



```python

if long\_letter\_run\_present:

&#x20;   # math-операторы внутри \\mathrm{}/\\mathbf{} не считаются

&#x20;   s\_clean = re.sub(

&#x20;       r"\\\\(?:mathrm|mathbf|text|mathit|mathtt|mathsf|mathcal|operatorname)\\{\[^{}]\*\\}",

&#x20;       "",

&#x20;       latex,

&#x20;   )

&#x20;   has\_math\_outside = bool(\_SEM\_MATH\_MARKERS.search(s\_clean))

&#x20;   if not has\_math\_outside:

&#x20;       return False, "caption\_or\_header"          # ← новая причина

&#x20;   return False, "long\_letter\_run"

```



В `process\_formula\_region`:



```python

if not sem\_ok:

&#x20;   if sem\_reason == "caption\_or\_header":

&#x20;       status = ExtractStatus.CAPTION\_OR\_HEADER.value

&#x20;   else:

&#x20;       status = ExtractStatus.SUSPICIOUS\_SEMANTIC.value

```



\*\*Ожидаемый эффект:\*\* \~30 блоков перейдут из `suspicious\_semantic`

в `caption\_or\_header`. Их `latex` и `text` останутся в JSON для RAG (как

заголовки/подписи), но не будут индексироваться как формулы.



\---



\## Проблема C: `figure\_caption` — ложные срабатывания



\### Пример



```

p034 \[semantic\_reason=figure\_caption]

&#x20;  latex: '\\sqrt { \\frac { \\varphi } { 1 } }'

&#x20;  src:   'text=5Σ+'

```



Это \*\*не\*\* подпись к рисунку, а `5Σ+` в таблице, случайно оказавшаяся рядом

с figure.



\### Фикс



Ужесточить условие:



```python

\_SEM\_FIGURE\_GAP\_PT = 15.0   # было 30.0



def \_is\_near\_figure(bbox, figure):

&#x20;   if \_share\_inside(bbox, figure) >= \_SEM\_FIGURE\_SHARE:

&#x20;       return True

&#x20;   x\_overlap = max(0.0, min(bbox.x2, figure.x2) - max(bbox.x1, figure.x1))

&#x20;   if x\_overlap < \_SEM\_FIGURE\_SHARE \* max(1e-6, bbox.x2 - bbox.x1):

&#x20;       return False

&#x20;   # строго под рисунком + gap < 15pt

&#x20;   if not (figure.y2 <= bbox.y1 <= figure.y2 + \_SEM\_FIGURE\_GAP\_PT):

&#x20;       return False

&#x20;   # И в latex должен быть длинный текст (не короткая формула)

&#x20;   return bool(\_SEM\_LETTER\_RUN.search(latex) or \_SEM\_RUSSIAN\_WORDS.search(latex))

```



`\_is\_near\_figure` нужно передать `latex`, чтобы проверить содержимое.



\---



\## Что ожидаем после всех трёх фиксов



| Документ | Было ok/sem/susp | Цель |

|---|---|---|

| `rm178b\_full` | 12 / 2 / 5 | \*\*12 / 2 / 1\*\* |

| `kt178c\_full` | 3 / 6 / 3 | \*\*4 / 4 / 1\*\* |

| `kt160g\_book1\_full` | 12 / 21 / 13 | \*\*16 / 8 / 5\*\* |

| `ac29\_full` | 27 / 17 / 18 | \*\*32 / 10 / 5\*\* |

| \*\*Итого\*\* | \*\*58 / 51 / 39\*\* | \*\*\~70 / \~30 / \~15\*\* |



Новый статус `caption\_or\_header` даст \*\*\~30 блоков\*\*, доступных для RAG как

заголовки/подписи.



\*\*Метрики после фикса:\*\*

\- ok: 58 → \*\*\~70\*\* (+20%),

\- sem: 51 → \*\*\~30\*\* (-40%),

\- susp: 39 → \*\*\~15\*\* (-60%),

\- \*\*real formula precision:\*\* цель \*\*>90%\*\* на `ok`-блоках.



\---



\## Как проверить



После реализации — \*\*перезапустить тот же ночной прогон\*\*:



```powershell

cd C:\\Users\\l.ilyintseva\\Desktop\\RAG-system

$env:ENABLE\_VLM = "false"

$env:ENABLE\_UNIMERNET = "true"

$env:UNIMERNET\_CONFIG\_PATH = ".\\data\\models\\unimernet\\unimernet\_tiny.yaml"

$PythonMain = ".\\.venv\\Scripts\\python.exe"



\# Smoke test — 5 PDF × 5 страниц, где были ошибки

\& $PythonMain -m ingestion.run\_regions "data\\raw\\КТ-160G-14G (Книга I)\_compressed.pdf" `

&#x20; --pages 1,7,30,34,46,55,82,96,122 --out-dir "data\\ir\\issue11\_smoke\_kt160g" `

&#x20; --enable-unimernet

```



\*\*Ожидание (сверить с текущим `night\_run`):\*\*

\- `p007, p011, p046, p055, p082, p096, p122` — должны быть `caption\_or\_header`, не `suspicious\_semantic`.

\- `p004, p011` — `caption\_or\_header`.

\- `p034` — `ok` (было `figure\_caption`).

\- Блоки с `unbalanced\_braces` (p055, p096) — `ok` с `braces\_repaired=true`.



\### Unit-тесты



Обновить `scripts/check\_semantic\_filter.py`:



\- Добавить кейсы `caption\_or\_header` (заголовки `KHHTAI`, `noxenHX`, `TyIIdTe`).

\- Добавить кейсы `braces\_repaired` (битый LaTeX с недостающей скобкой).

\- Добавить кейсы ложных `figure\_caption` (короткая формула рядом с figure).



\---



\## Файлы для изменения



| Файл | Что |

|---|---|

| `ingestion/formula\_pipeline.py` | `validate\_latex` (braces\_repaired), `is\_real\_formula\_semantic` (caption\_or\_header), `\_is\_near\_figure` (gap 15pt + проверка latex) |

| `ingestion/models.py` | `ExtractStatus.CAPTION\_OR\_HEADER` |

| `ingestion/process\_regions.py` | учёт `caption\_or\_header` в summary |

| `scripts/check\_semantic\_filter.py` | тесты для новых случаев |

| `docs/reports/2026-09-23\_semantic\_filter.md` | обновить итоговые цифры |



\---



\## Приложение: сырые данные ночного прогона



Полный JSON прогона: `data/ir/night\_run\_20260923\_175717/summary.md`.



Логи: `data/ir/night\_run\_20260923\_175717/logs/\_master.log`.



Примеры конкретных проблемных блоков приведены выше; при необходимости

можно воспроизвести через `scripts/check\_semantic\_filter.py --ir <path>`.

