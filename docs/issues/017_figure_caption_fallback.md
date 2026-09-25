\# Issue 17: Figure caption fallback без VLM



\*\*Дата:\*\* 2026-09-25  

\*\*Приоритет:\*\* средний  

\*\*Связан с:\*\* \[#016](./016\_figure\_crop\_cyrillic.md)



\---



\## Проблема



VLM отключён (`ENABLE\_VLM=false`). Все figure имеют `caption=''`.

Для RAG figure без caption \*\*бесполезен\*\* — пользователь не поймёт, что там.



\*\*Сейчас в JSON figure:\*\*

```json

{

&#x20; "type": "figure",

&#x20; "content": {

&#x20;   "caption": "",

&#x20;   "crop\_path": "data\\\\cache\\\\figures\\\\...\\\\p0009\_figure\_00.png",

&#x20;   "status": "ok"

&#x20; },

&#x20; "provenance": {"method": "crop\_only", "confidence": 0.35}

}

```



\*\*Что есть:\*\* crop\_path (файл, после Issue 16).

\*\*Чего нет:\*\* caption, figure\_number, размеры, тип (chart/photo/scheme).



\## Фикс



\### A. OCR fallback (без VLM)



Если VLM недоступен, но crop есть — прогнать \*\*RapidOCR по crop\*\*:



```python

\# table\_pipeline.py::process\_figure\_region

caption = ""

method = "crop\_only"



if vlm is not None:

&#x20;   try:

&#x20;       caption = vlm.read\_ndarray(crop, "Describe this figure ...")

&#x20;       method = "vlm\_caption"

&#x20;   except Exception as exc:

&#x20;       method = f"vlm\_failed:{exc}"



if not caption:

&#x20;   # OCR fallback

&#x20;   try:

&#x20;       ocr\_lines = recognize\_image(crop, dpi=dpi, page=page, doc\_id="figure")

&#x20;       caption = " ".join(ln.text.strip() for ln in ocr\_lines if ln.text.strip())

&#x20;       caption = caption\[:500]

&#x20;       method = "ocr\_caption" if caption else "crop\_only"

&#x20;   except Exception as exc:

&#x20;       method = f"ocr\_failed:{exc}"

```



\*\*Для схем и чертежей OCR найдёт надписи, размеры, обозначения — это уже полезный caption.\*\*



\### B. Поиск подписи рядом



Искать в spans паттерны `Фиг. N.M.` / `Fig. N.M.` / `Рис. N`:



```python

\_CAPTION\_RE = re.compile(r"(Фиг\\.|Fig\\.|Рис\\.|Figure)\\s\*\\d+\[\\.\\d]\*", re.IGNORECASE)



\# В process\_regions.py, передать page\_spans в process\_figure\_region

for sp in page\_spans:

&#x20;   if \_CAPTION\_RE.search(sp.text):

&#x20;       if \_bbox\_close(sp.bbox, region.bbox\_pt, y\_gap=80):

&#x20;           content\["figure\_number"] = sp.text.strip()\[:120]

&#x20;           break

```



\### C. Сохранить в content



```python

content = {

&#x20;   "caption": caption,               # VLM или OCR

&#x20;   "caption\_source": method,         # vlm\_caption | ocr\_caption | crop\_only

&#x20;   "figure\_number": figure\_number,   # "Фиг. I.1" если найдено

&#x20;   "crop\_path": str(crop\_path),

&#x20;   "status": "ok",

}

```



\## Ожидаемый эффект



На rdk89 92 figure:

\- \*\*92 crop-файла\*\* (после Issue 16).

\- \*\*\~60–70 caption\*\* через OCR (тексты на схемах).

\- \*\*\~10–15 figure\_number\*\* (найденные подписи).



\## Файлы для изменения



\- `ingestion/table\_pipeline.py::process\_figure\_region`

\- `ingestion/process\_regions.py` — передача `page\_spans` в figure.

\- `ingestion/models.py` — если нужно новое поле.



\## Тесты



```powershell

\& $PythonMain -m ingestion.run\_regions "data\\raw\\\_1954.pdf" `

&#x20; --pages 11 --out-dir "data\\ir\\issue17\_verify" --enable-unimernet

```



\*\*Ожидание:\*\*

```json

{

&#x20; "caption": "Фиг. I.1. Одновинтовой вертолет Белл Н-13.",

&#x20; "caption\_source": "ocr\_caption",

&#x20; "figure\_number": "Фиг. I.1"

}

```

