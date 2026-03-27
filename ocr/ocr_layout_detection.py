#!/usr/bin/env python
# coding: utf-8

# # Сравнение методов layout-анализа PDF-документов
# 
# 
# 
# В ноутбуке проводится сравнение методов **layout-анализа PDF-документов** (выделение текста, таблиц, изображений и др.).
# 
# Документ преобразуется в изображения, после чего к нему применяются различные модели: **Surya**, **PP-DocLayout**, **YOLO-based**, а также кратко рассматриваются **LayoutLMv3** и **Detectron2**.
# 
# Для каждого подхода оцениваются:
# 
# * качество детекции,
# * скорость работы,
# * удобство интеграции.
# 
# В конце приводится сравнение решений и выбор наиболее эффективного подхода.
# 

# ## Содержание
# 
# * [Импорт библиотек](#0)
# 
# * [1. Данные для анализа](#1)
# * [2. DocLayout решения](#2)
#     * [2.1 Surya](#2-1)
#     * [2.2 PP-DocLayout](#2-2)
#     * [2.3 LayoutLMv3](#2-3)
#     * [2.4 YOLO-based layout](#2-4)
#     * [2.5 Detectron2](#2-5)
# * [3. Сравнение решений](#3)

# ###

# <a id=0></a>
# ## Импорт библиотек

# In[2]:


import pandas as pd
import numpy as np

from time import time

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image

import pdfplumber

from pdf2image import convert_from_path

import torch

from surya.foundation import FoundationPredictor
from surya.layout import LayoutPredictor
from surya.settings import settings

from doclayout_yolo import YOLOv10

from paddleocr import LayoutDetection


# In[3]:


device = "cuda" if torch.cuda.is_available() else "cpu"


# ###

# <a id=1></a>
# # 1. Данные для анализа
# 
# В качестве документа, на котором будут тестироваться решения, возьмем `text.pdf`.

# In[4]:


PDF_PATH = 'text1.pdf'


# In[5]:


with pdfplumber.open(PDF_PATH) as pdf:
    first_page = pdf.pages[0]
    print(first_page.extract_text().strip()[:100])


# В документе присутствует текстовый слой, однако это бывает не всегда. В связи с этим требуется инструмент для анализа `layout` (структуры документа), обладающий высоким качеством распознавания.

# Многие алгоритмы не принимают `pdf-документы`. Конвертируем наш документ в набор изображений с помощью библиотеки `pdf2image`. Под капотом она использует программу `poppler`, которую нужно скачать на компьютер перед работой.

# In[6]:


images = convert_from_path(PDF_PATH, poppler_path=r"C:\poppler\Library\bin")


# ###

# <a id=2></a>
# # 2. DocLayout решения

# <a id=2-1></a>
# ## 2.1 Surya

# ### Обработка документа

# In[7]:


# Инициализация инструмента
layout_predictor = LayoutPredictor(FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT))

st_time = time()

# Предсказания
layout_predictions = layout_predictor(images)

surya_time = time() - st_time


# ### Визуализация обработки

# In[8]:


# Страницы для визуализации
pages2vis = [0, 4, 9]


# In[9]:


fig, axs = plt.subplots(1, 3, figsize=(19, 7))

for ax, page_num in zip(axs.ravel(), pages2vis):
    np_img = np.array(images[page_num])
    ax.imshow(np_img)

    # Получаем предсказания для страницы
    bboxes = layout_predictions[page_num].bboxes

    # Проходимся по каждому ббоксу
    for box in bboxes:
        x1, y1, x2, y2 = box.bbox

        # Прямоугольник
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=1,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)

        ax.text(
            x1, y1 + 10,
            f"{box.label} ({box.confidence:.2f})",
            color='red',
            fontsize=5,
            backgroundcolor='white'
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'Страница {page_num + 1}')


plt.suptitle('Surya DocLayout', fontsize=20);


# <a id=2-2></a>
# ## 2.2 PP-DocLayout
# 
# Модель **PP-DocLayout-L** из библиотеки PaddlePaddle (размер около 123 МБ) используется для анализа структуры документа (layout detection). Для корректной работы рекомендуется устанавливать проверенные версии зависимостей:
# 
# ```bash
# pip install paddlepaddle==3.2.0 paddleocr==3.3.3
# ```
# 
# Следует учитывать, что работа с данной библиотекой может сопровождаться проблемами совместимости зависимостей.
# 
# Также важно отметить особенность загрузки модели: если в пути присутствуют русские символы, модель может не скачиваться или не загружаться корректно. В этом случае необходимо вручную переместить модель в путь без кириллицы, например:
# 
# ```
# C:/models/paddle/PP-DocLayout-L
# ```
# 
# После этого путь к модели следует явно указать (захардкодить) в модуле `LayoutDetection`.
# 

# In[10]:


# Инициализация модели
model = LayoutDetection(model_name="PP-DocLayout-L",
                        model_dir="C:/models/paddle/PP-DocLayout-L")


# ### Обработка документа

# In[11]:


st_time = time()

# Передаем в PP-DocLayout весь документ
output = model.predict('text1.pdf', batch_size=10, layout_nms=True)

pp_time = time() - st_time


# ### Визуализация обработки

# In[12]:


# Страницы для визуализации
pages2vis = [0, 4, 9]


# In[13]:


fig, axs = plt.subplots(1, 3, figsize=(19, 7))

for ax, page_num in zip(axs.ravel(), pages2vis):
    np_img = output[page_num]['input_img']
    ax.imshow(np_img)

    # Получаем предсказания для страницы
    bboxes = output[page_num]['boxes']

    # Проходимся по каждому ббоксу
    for box in bboxes:
        x1, y1, x2, y2 = map(float, box['coordinate'])

        # Прямоугольник
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=1,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)

        ax.text(
            x1, y1 + 10,
            f"{box['label']} ({box['score']:.2f})",
            color='red',
            fontsize=5,
            backgroundcolor='white'
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'Страница {page_num + 1}')


plt.suptitle('PP-DocLayout', fontsize=20);


# <a id=2-3></a>
# ## 2.3 LayoutLMv3
# 
# **LayoutLMv3** — модель для анализа документов (текст + layout + изображение).
# 
# В данной работе не используем, так как требует внешний OCR и явной передачи слов и координат (bbox), что усложняет пайплайн и делает решение зависимым от качества распознавания. Предпочтение отдано более простым и современным подходам.
# 

# <a id=2-4></a>
# ## 2.4 YOLO-based layout
# 
# Для задачи layout-анализа также используется YOLO-подход. В качестве предобученной модели можно взять веса **DocLayout-YOLO-DocStructBench** из репозитория `juliozhao/DocLayout-YOLO-DocStructBench`.
# 
# Необходимо скачать файл весов:
# 
# ```
# doclayout_yolo_docstructbench_imgsz1024.pt
# ```
# 
# Данная модель обучена на датасете DocStructBench и предназначена для детекции структурных элементов документа (заголовки, текст, таблицы, изображения и др.).

# ### Обработка документа

# In[14]:


# Загружаем модель с заранее скачанными весами
model = YOLOv10("doclayout_yolo_docstructbench_imgsz1024.pt")


# In[15]:


st_time = time()

# Запускаем обработку
det_res = model.predict(
    images,            # Список изображений
    imgsz=1024,        # Размер изображений
    conf=0.2,          # Пороговый скор
    device="cuda:0"    # Используем видеокарту
)

yolo_time = time() - st_time


# ### Визуализация обработки

# In[16]:


# Страницы для визуализации
pages2vis = [0, 4, 9]


# In[17]:


fig, axs = plt.subplots(1, 3, figsize=(19, 7))

for ax, page_num in zip(axs.ravel(), pages2vis):
    page_res = det_res[page_num]
    class_names = page_res.names

    np_img = page_res.orig_img
    ax.imshow(np_img)

    # Получаем предсказания для страницы
    bboxes = page_res.boxes.xyxy
    labels = [class_names[i.item()] for i in page_res.boxes.cls]
    conf = page_res.boxes.conf

    # Проходимся по каждому ббоксу
    for box, label, confidence in zip(bboxes, labels, conf):
        x1, y1, x2, y2 = (i.item() for i in box)

        # Прямоугольник
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=1,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)

        ax.text(
            x1, y1 + 10,
            f"{label} ({confidence:.2f})",
            color='red',
            fontsize=5,
            backgroundcolor='white'
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'Страница {page_num + 1}')


plt.suptitle('YOLO-based Layout', fontsize=20);


# <a id=2-5></a>
# 
# ## 2.5 Detectron2
# 
# 
# **Detectron2** — это фреймворк от **Meta** для задач компьютерного зрения, включая детекцию объектов и *layout-анализ документов* (bbox + классы).
# 
# Однако он плохо устанавливается на **Windows**.
# 
# Тем не менее, это **мощный инструмент для layout detection**, который стоит рассмотреть в будущем.
# 

# ###

# <a id=3></a>
# # 3. Сравнение решений

# In[18]:


results = pd.DataFrame({
    "Метод": [
        "Surya",
        "PP-DocLayout",
        "YOLO",
        "LayoutLMv3",
        "Detectron2"
    ],
    "Время (сек)": [
        surya_time,
        pp_time,
        yolo_time,
        9999.,
        9999.
    ]
})

# сортировка по скорости
results = results.sort_values("Время (сек)")

results


# Все модели показали хорошее качество распознавания.
# Самым быстрым решением стал `YOLO-based Layout`!
