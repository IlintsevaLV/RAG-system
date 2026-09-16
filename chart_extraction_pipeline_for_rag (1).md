# Chart Extraction Pipeline для RAG

## 1. Цель

Построить надёжный pipeline извлечения информации из графиков для последующего использования в RAG-системах.

Ключевой принцип:

> Не строить pipeline как последовательность моделей. Строить его как `evidence extraction → graph model → validation → selective recovery`.

Основные цели:

- максимально точно восстановить численные данные;
- сохранить геометрию и provenance;
- отделить извлечение фактов от семантической интерпретации;
- использовать VLM только там, где deterministic pipeline обнаруживает неопределённость;
- сделать результат пригодным одновременно для структурированного анализа и RAG.

---

# 2. Предлагаемая архитектура

```text
                    SOURCE
                       │
                       ▼
               CHART DETECTION
                       │
              ┌────────┴────────┐
              ▼                 ▼
           VECTOR             RASTER
              │                 │
              └────────┬────────┘
                       ▼
              ELEMENT EXTRACTION
              ┌────────┼────────┐
              ▼        ▼        ▼
             CV       OCR     VECTOR
              └────────┼────────┘
                       ▼
                 GRAPH MODEL
                       │
                       ▼
                AXIS CALIBRATION
                       │
                       ▼
              DATA RECONSTRUCTION
                       │
                       ▼
                CONSISTENCY CHECK
                       │
                 ┌─────┴─────┐
                 │           │
               PASS        FAIL
                 │           │
                 │          VLM
                 │           │
                 │      recovery /
                 │       resolution
                 │           │
                 └─────┬─────┘
                       ▼
                FINAL VALIDATION
                       │
                       ▼
             ┌─────────┴─────────┐
             ▼                   ▼
       STRUCTURED JSON       SEMANTIC FACTS
             │                   │
             └─────────┬─────────┘
                       ▼
                      RAG
```

---

# 3. Главный принцип: evidence first

Не следует заставлять OCR/CV/VLM сразу отвечать на вопрос «что означает график».

Сначала извлекается evidence:

```text
image
vector objects
OCR tokens
geometric objects
coordinates
colors
paths
```

Затем evidence превращается в `Graph Model`.

И только после этого выполняются:

- data reconstruction;
- semantic interpretation;
- generation of RAG facts.

Это уменьшает количество преждевременных ошибок и позволяет проверять результаты между независимыми источниками.

---

# 4. Vector + Raster

Не использовать стратегию:

```text
vector OR image
```

Правильная стратегия:

```text
vector + raster/CV
```

Разные источники отвечают за разные свойства.

```text
VECTOR = geometry + topology
RASTER = appearance + pixels
CV     = object classification
OCR    = textual content
```

## Vector

Vector extraction особенно полезен для:

- точных координат;
- линий;
- paths;
- rectangles;
- circles;
- текста;
- размеров;
- порядка и пересечений объектов;
- topology.

Если PDF содержит настоящий vector layer, это часто более надёжный источник геометрии, чем image processing.

## Raster

Raster нужен для:

- screenshot-based charts;
- anti-aliased линий;
- цветов;
- визуальных особенностей;
- annotations;
- случаев, когда PDF содержит только rasterized chart.

---

# 5. Chart Detection

Первый этап должен определить:

```text
где находится график
```

Результат:

```json
{
  "chart_id": "chart_03",
  "bbox": [x1, y1, x2, y2],
  "page": 17
}
```

На этом этапе не требуется восстанавливать данные.

Важно получить максимально качественный crop, потому что последующие OCR/CV работают уже на ограниченной области.

---

# 6. Element Extraction

CV, OCR и vector extraction лучше рассматривать как единый логический этап.

```text
ELEMENT EXTRACTION
 ├── CV
 ├── OCR
 └── VECTOR
```

При этом элементы удобно разделять на две категории.

## Structural elements

```text
plot area
axes
grid
legend box
```

## Data-bearing elements

```text
bars
points
curves
areas
error bars
```

Сначала желательно определить structural elements:

```text
plot area
    ↓
coordinate system
    ↓
data-bearing elements
```

Это уменьшает пространство поиска и упрощает reconstruction.

---

# 7. CV / Tiny

CV не должен решать семантическую задачу.

Его задача — геометрическая классификация и detection.

Основные объекты:

- plot area;
- x/y axes;
- grid;
- bars;
- curves;
- points;
- error bars;
- legend box.

Результат должен содержать bbox/geometry и confidence.

Например:

```json
{
  "type": "curve",
  "id": "curve_01",
  "geometry": [...],
  "confidence": 0.96
}
```

---

# 8. OCR

OCR должен работать в два этапа:

```text
text detection
      ↓
recognition
```

То есть OCR отвечает:

> где находится текст и что там написано?

Но не должен самостоятельно решать:

> это axis label или legend?

Каждый OCR object необходимо сохранять с bbox.

```json
{
  "text": "2024",
  "bbox": [412, 580, 450, 604],
  "confidence": 0.98
}
```

## Что извлекать

- title;
- axis labels;
- axis units;
- tick labels;
- tick numbers;
- legend labels;
- annotations.

## Важное дополнение

Для графиков нужна отдельная метрика `numeric accuracy`.

Например:

```text
OCR: 1,250 → 1,280
```

Небольшая текстовая ошибка может быть критической численной ошибкой.

Поэтому обычного CER недостаточно.

---

# 9. Semantic classification OCR objects

После OCR отдельный слой должен определить роль текста:

```text
"2024"
    ↓
tick_candidate

"Revenue"
    ↓
axis_label_candidate

"Germany"
    ↓
legend_candidate
```

Таким образом:

```text
OCR = text evidence
semantic layer = text role
```

Это лучше, чем смешивать recognition и interpretation.

---

# 10. Graph Model

После extraction необходимо построить промежуточное canonical representation.

Это не должен быть финальный RAG JSON.

Пример:

```json
{
  "type": "line_chart",

  "axes": [...],

  "series": [
    {
      "id": "series_1",
      "label": "Revenue",
      "geometry": [...]
    }
  ],

  "text": [...],

  "annotations": [...]
}
```

Graph Model описывает:

> что физически находится на графике?

А semantic layer описывает:

> что это означает?

Это разделение желательно сохранять архитектурно.

---

# 11. Axis Calibration

`pixel ↔ data coordinates` следует сделать отдельным центральным модулем.

```text
Axis Calibration
       │
       ├── X transform
       └── Y transform
```

Например:

```text
pixel_y = 742
      ↓
Y-axis calibration
      ↓
value = 125.4
```

Необходимо поддерживать как минимум:

- linear;
- logarithmic;
- categorical;
- datetime;
- dual axis;
- broken axis.

Axis calibration особенно критична, потому что ошибка в ней может испортить большое количество восстановленных значений.

---

# 12. Graph Reconstruction

Основная задача:

```text
axis ↔ label
tick ↔ value
curve ↔ legend
bar ↔ category
point ↔ coordinate
pixel ↔ data coordinate
```

Для каждой series желательно восстановить:

```json
{
  "series": "Revenue",
  "data": [
    {"x": 2020, "y": 120},
    {"x": 2021, "y": 145},
    {"x": 2022, "y": 180}
  ]
}
```

При этом каждое значение должно быть связано с исходной геометрией.

---

# 13. Constraint-based Validation

Quality check лучше строить не только на confidence scores.

Сам график содержит ограничения, позволяющие проверить результат.

## Пример: series count

```text
CV: 3 curves
OCR legend: 4 series
```

Результат:

```text
SUSPICIOUS
```

## Пример: consistent legend

```text
3 curves
3 legend entries
3 corresponding colors
```

Результат:

```text
CONSISTENT
```

## Пример: monotonic trend

Если geometry показывает:

```text
x1 < x2 < x3
y1 < y2 < y3
```

то semantic interpretation «значение растёт» должна быть совместима с reconstruction.

## Другие constraints

Проверять:

- количество series;
- количество bars;
- количество points;
- соответствие legend ↔ series;
- соответствие ticks ↔ labels;
- порядок категорий;
- диапазон значений;
- соответствие geometry и reconstructed values;
- baseline;
- наличие dual axes;
- consistency цветов/линий.

---

# 14. Confidence Vector

Не следует сводить всё к одному confidence score.

Лучше хранить вектор:

```json
{
  "geometry": 0.97,
  "ocr": 0.91,
  "calibration": 0.98,
  "series_mapping": 0.73,
  "data_reconstruction": 0.95,
  "semantic": 0.82
}
```

Это позволяет понять, где именно проблема.

---

# 15. Selective Escalation

VLM не должен запускаться на каждом графике.

Pipeline должен использовать confidence-driven escalation:

```text
deterministic extraction
        ↓
validation
        ↓
confidence
   /           high            low
 │               │
accept          VLM
                 │
          recovery/resolution
```

Примеры правил:

```text
calibration < 0.90
    → VLM / alternative recovery

series_mapping < 0.85
    → VLM

OCR confidence < 0.80
    → second OCR / higher resolution

geometry < 0.85
    → alternative CV
```

Пороговые значения должны быть подобраны экспериментально на validation set.

---

# 16. VLM как Expert Witness

VLM лучше использовать не как основной extractor, а как механизм разрешения неопределённости.

Плохой запрос:

```text
Extract this chart.
```

Лучший запрос:

```text
Detected:
- 3 curves
- 4 legend entries
- 17 OCR tokens
- x-axis: 2018–2024
- y-axis: 0–100

Conflict:
3 curves != 4 legend entries

Task:
determine whether the fourth legend entry corresponds
to a hidden or overlapping series.
```

VLM получает:

- chart crop;
- CV results;
- OCR results;
- vector evidence;
- detected conflicts.

Таким образом VLM решает узкую задачу recovery/resolution.

---

# 17. Three representations для RAG

После extraction желательно сохранять три слоя.

## A. Evidence

Максимально близко к источнику:

```json
{
  "bbox": [...],
  "text": "2024",
  "source": "ocr"
}
```

## B. Structured

Нормализованные данные:

```json
{
  "x": 2024,
  "y": 182.4,
  "series": "Revenue"
}
```

## C. Semantic

Факты для retrieval:

```text
Revenue was approximately 182.4 million in 2024.
```

Архитектура:

```text
                 CHART
                   │
          ┌────────┼────────┐
          ▼        ▼        ▼
       Evidence Structured Semantic
          │        │        │
          │        └────┐   │
          │             ▼   ▼
          │           RAG INDEX
          │               │
          └───────────────┘
                  provenance
```

---

# 18. RAG JSON

Финальный JSON может выглядеть так:

```json
{
  "document_id": "report_2025",
  "page": 17,
  "chart_id": "chart_03",

  "title": "Revenue by region",

  "chart_type": "bar",

  "x_axis": {
    "label": "Region",
    "categories": [
      "Europe",
      "Asia",
      "North America"
    ]
  },

  "y_axis": {
    "label": "Revenue",
    "unit": "USD million",
    "scale": "linear"
  },

  "series": [
    {
      "name": "2025",
      "data": [
        {
          "x": "Europe",
          "y": 320,
          "provenance": {
            "bbox": [...]
          }
        },
        {
          "x": "Asia",
          "y": 450,
          "provenance": {
            "bbox": [...]
          }
        }
      ]
    }
  ],

  "quality": {
    "geometry": 0.97,
    "ocr": 0.91,
    "calibration": 0.98,
    "data_reconstruction": 0.95
  },

  "source": {
    "document": "report_2025.pdf",
    "page": 17
  }
}
```

---

# 19. Provenance

Для RAG provenance является обязательной частью модели.

Каждое число желательно связывать с:

```text
document
page
chart_id
element_id
bbox
extraction_method
confidence
```

Например:

```json
{
  "value": 450,
  "source": {
    "document": "report.pdf",
    "page": 17,
    "chart_id": "chart_03",
    "element_id": "bar_07",
    "bbox": [412, 285, 463, 421],
    "method": "vector+cv",
    "confidence": 0.94
  }
}
```

Это позволяет не только ответить:

> Revenue in Asia was $450M.

но и показать:

> из какого элемента графика получено значение.

---

# 20. Метрики

Метрики лучше организовать по слоям.

## Detection

```text
region IoU
element precision
element recall
element F1
```

## OCR

```text
CER
WER
numeric accuracy
```

## Reconstruction

```text
axis calibration error
MAE
RMSE
series assignment accuracy
point/bar reconstruction accuracy
```

## Semantics

```text
trend accuracy
legend-series accuracy
annotation interpretation accuracy
```

`trend accuracy` является дополнительной semantic metric и не заменяет численную точность.

## Completeness

```text
coverage
missing-series rate
missing-point rate
```

## RAG-level

### Chart QA accuracy

Проверять конечный результат:

```text
Q: What was revenue in 2023?
A: 184M
```

Это важнее многих промежуточных метрик, поскольку непосредственно измеряет пригодность extraction pipeline для RAG.

### Provenance accuracy

Проверять цепочку:

```text
answer
  ↓
value
  ↓
source element
  ↓
bbox
  ↓
page
```

---

# 21. Рекомендуемый набор cross-checks

Минимальный набор:

```text
CV elements
      ↕
Vector elements

OCR tick labels
      ↕
Axis geometry

OCR legend
      ↕
Detected series

Series geometry
      ↕
Reconstructed data

Axis calibration
      ↕
Tick values

Reconstructed trend
      ↕
Visual trend
```

Любое существенное расхождение должно повышать suspicion score и потенциально запускать recovery.

---

# 22. Упрощённая концептуальная модель

Вместо большого количества независимых этапов систему можно мыслить как четыре уровня:

```text
1. EXTRACT
   ↓
2. MODEL
   ↓
3. VALIDATE
   ↓
4. RECOVER
```

### EXTRACT

```text
vector
raster
CV
OCR
```

### MODEL

```text
Graph Model
Axis Calibration
Data Reconstruction
```

### VALIDATE

```text
constraints
cross-checks
confidence vector
metrics
```

### RECOVER

```text
VLM
alternative OCR
alternative CV
manual review
```

После этого:

```text
FINAL MODEL
    ↓
JSON
    +
Semantic Facts
    ↓
RAG
```

---

# 23. Ключевые архитектурные принципы

## 1. Extract evidence, don't interpret prematurely

Сначала:

```text
geometry
text
objects
coordinates
```

Потом:

```text
meaning
```

## 2. Use all available modalities

Не выбирать между vector и raster.

Использовать:

```text
vector + raster + CV + OCR
```

## 3. Validate through constraints

Confidence модели недостаточен.

Использовать внутреннюю согласованность графика.

## 4. Escalate only uncertainty

VLM должен быть дорогим recovery-механизмом, а не обязательным extractor'ом.

## 5. Preserve provenance

Каждый важный факт должен быть трассируем до исходного элемента.

## 6. Optimize for the final RAG task

Помимо extraction metrics необходимо измерять:

```text
Can RAG answer questions about the chart correctly?
```

---

# 24. Итоговая рекомендация

Оптимальная production-архитектура:

```text
SOURCE
  ↓
CHART DETECTION
  ↓
VECTOR + RASTER
  ↓
CV + OCR + VECTOR EXTRACTION
  ↓
GRAPH MODEL
  ↓
AXIS CALIBRATION
  ↓
DATA RECONSTRUCTION
  ↓
CONSTRAINT / CROSS-MODAL VALIDATION
  ↓
CONFIDENCE VECTOR
  │
  ├── HIGH → FINALIZE
  │
  └── LOW → VLM / alternative extractor
                  ↓
             FINAL VALIDATION
                  ↓
              FINAL MODEL
              ├── Evidence
              ├── Structured data
              └── Semantic facts
                  ↓
               RAG INDEX
```

Главное упрощение по сравнению с исходной схемой заключается в том, что **VLM перестаёт быть этапом pipeline и становится механизмом selective recovery**, а `Graph Model` становится центральным промежуточным представлением.

Это делает систему:

- дешевле;
- детерминированнее;
- проще для отладки;
- проще для оценки;
- лучше подходящей для provenance;
- удобнее для последующего RAG.
