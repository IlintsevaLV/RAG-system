\# Issue 16: Figure crop теряется на кириллических именах (cv2.imwrite silently fails)



\*\*Дата:\*\* 2026-09-25  

\*\*Приоритет:\*\* критический  

\*\*Связан с:\*\* \[#017](./017\_figure\_caption\_fallback.md), \[#018](./018\_figure\_bbox\_filter.md)



\---



\## Проблема



На `night\_run\_v2\_20260924\_125552`:

\- \*\*rdk89\*\*: 92 figure в JSON, \*\*0 crop-файлов\*\* на диске.

\- \*\*ac29\*\*: 9 figure, \*\*0 crop-файлов\*\*.

\- \*\*kt160g\_book1\*\*: 6 figure, \*\*0 crop-файлов\*\*.

\- \*\*usarmy\*\*: 111 figure, \*\*111 crop-файлов ✅\*\* (латиница).

\- \*\*\_1954\*\*: работает ✅ (латиница).



\*\*Crop'ы не сохраняются, если `doc\_id` содержит кириллицу.\*\*



\## Диагностика (воспроизведено)



```python

import cv2, numpy as np

from pathlib import Path



img = np.zeros((100, 100, 3), dtype=np.uint8)



p1 = Path('data/cache/figures/РДК\_молниезащита 89г/test\_ru.png')

p1.parent.mkdir(parents=True, exist\_ok=True)

ok1 = cv2.imwrite(str(p1), img)

print(f'Russian: ok={ok1}, exists={p1.is\_file()}')



p2 = Path('data/cache/figures/test\_lat/test\_lat.png')

p2.parent.mkdir(parents=True, exist\_ok=True)

ok2 = cv2.imwrite(str(p2), img)

print(f'Latin:   ok={ok2}, exists={p2.is\_file()}')

```



\*\*Результат:\*\*

```

Russian: ok=False, exists=False   ← ПРОВАЛ, файла нет

Latin:   ok=True,  exists=True    ← OK

```



\*\*Причина:\*\* `cv2.imwrite` использует `fopen()` в C++, который на Windows

не поддерживает Unicode-пути. Возвращает `False` \*\*молча\*\*, без исключения.



\## Где именно ломается



`ingestion/table\_pipeline.py:825-829`:

```python

crop = crop\_region\_bgr(image\_bgr, region, pad=6)

out\_dir = cache\_dir / "figures" / doc\_id          # ← doc\_id = "РДК\_молниезащита 89г"

out\_dir.mkdir(parents=True, exist\_ok=True)

crop\_path = out\_dir / f"p{page:04d}\_{region\_id}.png"

cv2.imwrite(str(crop\_path), crop)                  # ← silently fails

```



`out\_dir.mkdir` работает (Python через `pathlib` умеет Unicode).

`cv2.imwrite` — не работает.



\## Фикс



\### A. Использовать `cv2.imencode` + `Path.write\_bytes` (рекомендуется)



```python

\# Заменяем:

cv2.imwrite(str(crop\_path), crop)



\# На:

ok, buf = cv2.imencode('.png', crop)

if ok:

&#x20;   crop\_path.write\_bytes(buf.tobytes())

else:

&#x20;   log.warning(f"crop\_encode\_failed: {crop\_path}")

```



`Path.write\_bytes` использует `open()` в Python → Unicode-safe.



\### B. Или PIL



```python

from PIL import Image

rgb = cv2.cvtColor(crop, cv2.COLOR\_BGR2RGB)

Image.fromarray(rgb).save(crop\_path)

```



PIL — Unicode-safe.



\### C. Или нормализовать `doc\_id`



```python

import re

safe\_doc\_id = re.sub(r'\[^\\w\\-.]', '\_', doc\_id)

out\_dir = cache\_dir / "figures" / safe\_doc\_id

```



\*\*Минус:\*\* имя папки не совпадает с `doc\_id` в JSON — неудобно искать.

\*\*Плюс:\*\* гарантирует работу во всех местах (`cv2.imwrite`, `cv2.imread`).



\### D. Применить тот же фикс для таблиц



В `table\_pipeline.py:97` и `695` — \*\*тот же `cv2.imwrite`\*\*:



```python

\# table\_pipeline.py:97

img\_path = tmp\_dir / "table\_crop.png"

cv2.imwrite(str(img\_path), image\_bgr)      # ← тоже fails на кириллице



\# table\_pipeline.py:695

tmp = cache\_dir / "tables" / doc\_id / f"p{page:04d}\_{region\_id}"

tmp.mkdir(parents=True, exist\_ok=True)

cv2.imwrite(str(tmp / "crop.png"), crop)   # ← тоже fails

```



\*\*Docling может не найти `table\_crop.png`\*\* → fallback не работает → таблицы на кириллических PDF теряются.



\### E. Проверка после записи



```python

if not crop\_path.is\_file():

&#x20;   log.warning(f"crop\_not\_saved: {crop\_path}")

&#x20;   content\["crop\_path"] = ""    # не обещаем то, чего нет

```



\## Тесты



\### Unit-тест (в `scripts/check\_crop\_save.py`)



```python

def test\_crop\_save\_cyrillic():

&#x20;   img = np.zeros((100, 100, 3), dtype=np.uint8)

&#x20;   for doc\_id in \["\_1954", "РДК\_молниезащита 89г", "КТ-160G-14G"]:

&#x20;       out = Path("data/cache/figures") / doc\_id

&#x20;       out.mkdir(parents=True, exist\_ok=True)

&#x20;       p = out / "test.png"

&#x20;       ok, buf = cv2.imencode('.png', img)

&#x20;       assert ok

&#x20;       p.write\_bytes(buf.tobytes())

&#x20;       assert p.is\_file(), f"crop not saved: {p}"

&#x20;       p.unlink()

```



\### Интеграционный тест



```powershell

\& $PythonMain -m ingestion.run\_regions "data\\raw\\РДК\_молниезащита 89г.pdf" `

&#x20; --pages 9 --out-dir "data\\ir\\issue16\_verify" --enable-unimernet



\# Проверить:

Get-ChildItem "data\\cache\\figures\\РДК\_молниезащита 89г" | Select Name

```



\*\*Ожидание:\*\* crop-файлы появляются.



\## Ожидаемый эффект



| Документ | Было crop | Цель |

|---|---|---|

| rdk89 | 0 / 92 | \*\*92\*\* |

| ac29 | 0 / 9 | \*\*9\*\* |

| kt160g\_book1 | 0 / 6 | \*\*6\*\* |

| usarmy | 111 / 111 | 111 (не сломать) |

| \_1954 | OK | OK |



\## Файлы для изменения



| Файл | Что |

|---|---|

| `ingestion/table\_pipeline.py:829` | `cv2.imwrite` → `cv2.imencode` + `write\_bytes` |

| `ingestion/table\_pipeline.py:97` | то же для `table\_crop.png` |

| `ingestion/table\_pipeline.py:695` | то же для `crop.png` таблиц |

| `ingestion/preprocess.py:348` | проверить — там тоже `cv2.imwrite` |

| Новый `scripts/check\_crop\_save.py` | unit-тест Unicode-пути |



\## Связанные issues



\- \*\*#017\*\* — Figure caption fallback (нужен crop → нужен файл).

\- \*\*#018\*\* — Figure detector precision (полноразмерные bbox).

