\# Issue 18: Figure detector — отсев полноразмерных bbox



\*\*Дата:\*\* 2026-09-25  

\*\*Приоритет:\*\* средний  

\*\*Связан с:\*\* \[#016](./016\_figure\_crop\_cyrillic.md)



\---



\## Проблема



На rdk89 многие figure имеют bbox = \*\*вся страница\*\*:



```

p0001: bbox=(0,0,371,791)         ← 98% площади страницы

p0009: bbox=(0,10,605,804)        ← 100%

p0010: bbox=(0,0,615,805)         ← 100%

p0025: bbox=(0,0,607,787)         ← 100%

p0026: bbox=(0,0,615,803)         ← 100%

p0039: bbox=(0,0,612,790)         ← 99%

p0157: bbox=(0,0,606,799)         ← 100%

```



\*\*6+ из 10 crop'ов — вся страница.\*\* `ink\_blob` детектит \*\*сканированный

лист\*\* как один большой figure.



\*\*Это не рисунок\*\* — это вся страница с текстом, таблицами, формулами.



\## Фикс



В `detect\_figure\_regions` (или в `merge\_regions`):



```python

def \_is\_valid\_figure\_bbox(bbox, \*, page\_w, page\_h):

&#x20;   x1, y1, x2, y2 = bbox

&#x20;   w, h = x2 - x1, y2 - y1

&#x20;   page\_area = page\_w \* page\_h

&#x20;   bbox\_area = w \* h



&#x20;   # Figure не может занимать >60% площади страницы

&#x20;   if bbox\_area > 0.6 \* page\_area:

&#x20;       return False, "too\_large"



&#x20;   # Figure не может быть почти как страница

&#x20;   if w > 0.9 \* page\_w and h > 0.9 \* page\_h:

&#x20;       return False, "full\_page"



&#x20;   # Минимальный размер: 50x50 pt

&#x20;   if w < 50 or h < 50:

&#x20;       return False, "too\_small"



&#x20;   # Aspect ratio: 0.15 < w/h < 7

&#x20;   ratio = w / max(h, 1)

&#x20;   if ratio < 0.15 or ratio > 7:

&#x20;       return False, "bad\_aspect"



&#x20;   return True, ""

```



Применить в `region\_detect.py` перед добавлением figure в regions.



\## Ожидаемый эффект



| Документ | fig было | fig цель |

|---|---|---|

| rdk89 | 92 | \*\*\~20–30\*\* |

| usarmy | 111 | \*\*\~80–90\*\* (там реально много рисунков) |

| ac29 | 9 | \~9 (не сломать) |



\## Тесты



```powershell

\# После фикса

\& $PythonMain -m ingestion.run\_regions "data\\raw\\РДК\_молниезащита 89г.pdf" `

&#x20; --pages 1,9,10,25,26,39 --out-dir "data\\ir\\issue18\_verify" --enable-unimernet

```



\*\*Ожидание:\*\* `figure=0` на этих страницах (были ложные полноразмерные bbox).



\## Файлы для изменения



\- `ingestion/region\_detect.py` — `\_is\_valid\_figure\_bbox`.

\- `ingestion/merge\_regions` — если там формируются figure.

