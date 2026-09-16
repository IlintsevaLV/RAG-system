# roadmap_new — улучшение обработки технических документов

## Цель

Расширить исходный `roadmap.md` от OCR-ориентированного pipeline до управляемого document-understanding pipeline для технических документов:

```text
PDF
→ анализ источника
→ предобработка
→ блоки
→ специализированное распознавание
→ единый JSON/IR
→ quality control
→ структура/семантика
→ retrieval + graph
```

---

## 1. Анализ файла ДО предобработки

### 1.1 Text layer

Сначала проверить наличие и качество text layer.

Предлагаемый **Text Layer Quality Score (TLQ)**:

```text
TLQ =
0.30 × printable_ratio
+ 0.20 × language_score
+ 0.20 × geometry_score
+ 0.15 × text_density_score
+ 0.15 × visual_agreement
```

Где:
- `printable_ratio` — доля нормальных печатных символов;
- `language_score` — соответствие ожидаемому языку/алфавиту;
- `geometry_score` — корректность bbox и порядка текста;
- `text_density_score` — разумная плотность текста;
- `visual_agreement` — согласованность text layer с визуальным содержимым.

Важно: **объём text layer не является признаком качества**. Мусорный или дублирующий слой может содержать тысячи символов.

Роутинг:

```text
TLQ ≥ threshold
    → использовать text layer

TLQ < threshold / text layer отсутствует
    → render → preprocessing → OCR
```

Порог определить на golden set, а не задавать произвольно.

---

## 2. Предобработка

Сначала определить **Image Quality Score (IQS)** по:
- разрешению;
- blur;
- noise;
- contrast;
- skew;
- compression artifacts.

Три уровня:

### GOOD

```text
render
→ deskew при необходимости
→ OCR/layout
```

### MEDIUM

```text
render
→ deskew
→ mild denoise
→ contrast normalization
→ OCR/layout
```

### BAD

```text
high-resolution render
→ deskew
→ denoise
→ contrast enhancement
→ layout
→ high-resolution crops / bands
→ OCR/VLM при необходимости
```

Бинаризация — только экспериментальная ветка. В исходном roadmap текущие измерения показывают ухудшение качества RapidOCR и VLM, поэтому не включать её по умолчанию.

---

## 3. Разбиение страницы на блоки

После анализа страницы выделять:

```text
TEXT
FORMULA
CHART
DRAWING
TABLE
```

Разделять **layout detection** и **recognition**.

Каждый блок получает единый интерфейс:

```json
{
  "doc_id": "doc_001",
  "page": 17,
  "region_id": "r4",
  "type": "chart",
  "bbox": [x1, y1, x2, y2],
  "tiles": [],
  "source": "raster",
  "quality": {},
  "content": {},
  "provenance": {
    "model": "...",
    "version": "..."
  }
}
```

Обязательно сохранять исходные координаты страницы. Tile — технический фрагмент, а не отдельный смысловой блок.

---

## 4. TEXT

Baseline:

```text
OCR
→ normalization
→ structural validation
```

Метрики:

```text
CER
WER
line recall
line precision
coverage = OCR / detector
```

`coverage` — автоматический production-сигнал; настоящий recall требует ground truth.

### Разбиение на полосы

Не использовать квадратные tiles как основной способ. Исходный roadmap показал, что tiles 1600 px находят больше строк, но ухудшают порядок и дробят строки.

Основные варианты для эксперимента:

**A. Horizontal full-width bands — основной кандидат**

```text
████████████████
████████████████
████████████████
```

Overlap: 10–15%.

**B. Adaptive bands**

Высота полосы зависит от плотности текста/детектора.

**C. Layout-based bands**

Сначала найти текстовые области, затем обрабатывать горизонтальными полосами внутри них.

Рекомендуется начать с A и сравнить несколько высот полос на golden set.

---

## 5. CHART

Предлагаемый pipeline:

```text
source analysis
→ vector extraction +/or image
→ CV/Tiny
→ OCR
→ graph reconstruction
→ quality check
→ VLM recovery/semantic analysis при необходимости
→ final quality check
→ JSON
```

### Vector

Не использовать правило `vector OR image`. Если доступны vector objects, использовать их вместе с raster/CV.

```text
vector = точная геометрия
CV     = геометрическая классификация
OCR    = текст/числа
```

### CV/Tiny

Выделять:
- plot area;
- axes;
- curves;
- points;
- bars;
- grid.

### OCR

Извлекать:
- axis labels;
- units;
- tick numbers;
- title;
- legend;
- annotations.

Все OCR objects сохранять с bbox.

### Graph reconstruction

Связать:

```text
axis ↔ label
tick ↔ value
curve ↔ legend
pixel ↔ data coordinates
```

### VLM

Использовать не как основной extractor, а:
- для recovery при подозрительном качестве;
- для сложных annotations;
- для семантической интерпретации;
- для конфликтов CV/OCR.

### Метрики

Не сводить всё к одному score:

```text
region IoU
element precision / recall / F1
OCR CER/WER
axis calibration error
data reconstruction MAE/RMSE
trend accuracy
completeness / coverage
```

`trend accuracy` — дополнительная semantic metric, не замена численной точности.

Добавить cross-check:

```text
CV: 3 curves
OCR legend: 4 series
→ suspicious
```

---

## 6. FORMULA

Отдельный экспериментальный pipeline с UniMERNet:

```text
formula detection
→ high-resolution crop
→ UniMERNet
→ LaTeX / syntax
→ normalization
→ validation
```

Метрики должны включать не только текстовую похожесть, но и структурную корректность формулы.

Сохранять:

```text
original image
raw LaTeX
normalized LaTeX
text/syntax representation
confidence
validation status
bbox
```

Текстовое/синтаксическое представление использовать далее для объяснений LLM.

---

## 7. TABLE

Pipeline:

```text
table detection
→ Docling structure
→ rows / columns / merged cells
→ OCR per cell
→ cell-level validation
→ table JSON/Markdown/HTML
```

Внутри каждой ячейки применять те же OCR-принципы, что и для текста:

```text
CER
WER
confidence
coverage
normalization
```

Для спецификаций сохранить отдельный детерминированный XLSX parser из исходного roadmap.

---

## 8. Unified IR

Все типы блоков должны иметь общий envelope:

```text
doc_id
page
region_id
type
bbox
tiles
source
model/version
quality
content
provenance
```

При этом `content` специализирован:

```text
TEXT    → text
FORMULA → latex
TABLE   → cells/structure
CHART   → axes/series/data
DRAWING → zones/text/geometry
```

Raw и normalized данные хранить отдельно.

---

## 9. DRAWING

Для ESKD/A1 не отправлять весь лист в VLM.

Использовать геометрию зон:

```text
title block
technical requirements
side fields
drawing field
```

Pipeline:

```text
zone detection
→ high-resolution crop
→ OCR/VLM per zone
→ geometry/OCR validation
```

Side fields при необходимости поворачивать на 90°.

Для drawing field использовать RapidOCR/CV на исходном разрешении, bands/crops при необходимости.

Поля основной надписи позднее использовать как свойства entities в Stage 5.

---

## 10. Quality routing

Quality layer должен работать после каждого специализированного этапа.

```text
GOOD
→ accept

SUSPICIOUS
→ second pass / VLM / high-res crop

FAILED
→ fallback/manual review
```

Не использовать confidence как единственный критерий.

Использовать комбинацию:

```text
confidence
+ coverage
+ element count
+ layout/OCR mismatch
+ structural validation
+ cross-method consistency
```

---

## 11. Graph / Embeddings

Для технической документации **только embeddings недостаточно**.

Embeddings хорошо отвечают:

```text
"найди похожее / релевантное"
```

Но плохо выражают точные отношения:

```text
деталь A → изготовлена из → материал B
деталь A → указана на → чертеже C
стандарт D → задаёт требование → E
документ A → версия → B
узел A → состоит из → B
```

Поэтому сложную иерархию нужно сохранить, но не превращать весь документ в граф.

Рекомендуемая модель:

```text
Document IR
     │
     ├── Qdrant
     │     └── semantic/hybrid retrieval
     │
     └── Neo4j
           └── entities + relations
```

Сначала:

```text
OCR
→ normalization
→ structure
→ entity extraction
→ entity resolution
→ graph
```

Не строить graph непосредственно из OCR.

---

# Основные изменения относительно исходного roadmap

1. **Добавить pre-OCR source analysis**: text layer + vector/raster analysis.
2. **Ввести TLQ и IQS** для routing до OCR.
3. **Формально добавить layout/region stage**.
4. Разделить ingestion на **TEXT / FORMULA / TABLE / CHART / DRAWING**.
5. Ввести **Unified Document IR** с обязательным bbox/provenance.
6. Для текста расширить OCR evaluation: **line recall/precision + coverage**.
7. Для крупных страниц протестировать **horizontal bands** вместо квадратных tiles.
8. Добавить отдельный **chart reconstruction pipeline**.
9. Добавить отдельный **formula pipeline с UniMERNet**.
10. Для таблиц использовать **structure → OCR per cell**, а не только line OCR.
11. Для A1 drawing закрепить **zone-based processing**.
12. Ввести **multi-signal quality routing** и cross-check между CV/OCR/vector.
13. Graph сохранить, но использовать его **вместе с Qdrant**, а не вместо embeddings.
14. VLM использовать преимущественно как **recovery/semantic layer**, а не как основной OCR всего документа.

## Итоговая схема

```text
PDF
 ↓
SOURCE ANALYSIS
 ├─ text layer ──→ если TLQ хороший → text
 ├─ vector
 └─ raster
 ↓
PREPROCESSING (IQS: good / medium / bad)
 ↓
LAYOUT / REGIONS
 ├─ TEXT
 ├─ FORMULA
 ├─ TABLE
 ├─ CHART
 └─ DRAWING
 ↓
SPECIALIZED PROCESSING
 ↓
UNIFIED IR
 ↓
QUALITY / ROUTING
 ↓
NORMALIZATION + STRUCTURE
 ↓
ENTITIES / RELATIONS
 ├─ Qdrant
 └─ Neo4j
 ↓
RAG
```
