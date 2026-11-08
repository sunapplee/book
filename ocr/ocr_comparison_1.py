#!/usr/bin/env python
# coding: utf-8

# # Сравнение OCR-решений для обработки PDF. Часть 1
# 
# В ноутбуке проводится сравнение OCR-решений для извлечения текста из PDF-документа `text1.pdf`. Рассмотрены инструменты: **Pytesseract**, **Marker-PDF (Surya)**, **Docling**, **Nanonets-OCR** и **PaddleOCR-VL** (не запущен из-за проблем с зависимостями).
# 
# Оценка выполняется по скорости и качеству распознавания. Реализована обработка изображений и замер времени работы каждого метода.

# ## Содержание
# 
# * [Импорт библиотек](#0)
# 
# * [1. Данные для анализа](#1)
# * [2. OCR решения](#2)
#     * [2.1 Pytesseract](#2-1)
#     * [2.2 Marker-PDF](#2-2)
#     * [2.3 Docling](#2-3)
#     * [2.4 Nanonets-OCR](#2-4)
#     * [2.5 PaddleOCR-VL](#2-5)
# * [3. Сравнение решений](#3)
# 
# 

# ###

# <a id=0></a>
# ## Импорт библиотек

# In[1]:


import pdfplumber
import pandas as pd

import pytesseract

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

from docling.document_converter import DocumentConverter

from transformers import pipeline

import torch

from pdf2image import convert_from_path

import time
from joblib import Parallel, delayed
from tqdm import tqdm


# ###

# <a id=1></a>
# # 1. Данные для анализа
# 
# В качестве документа, на котором будут тестироваться решения, возьмем `text.pdf`.

# In[2]:


PDF_PATH = 'text1.pdf'


# In[3]:


with pdfplumber.open(PDF_PATH) as pdf:
    first_page = pdf.pages[0]
    print(first_page.extract_text().strip()[:100])


# В документе присутствует текстовый слой. Однако такое бывает **не всегда**. В связи с этом нужен `OCR инструмент`, обладающий хорошим качеств распознавания и скоростью обработки.

# <a id=2></a>
# # 2. OCR решения 

# In[4]:


device = "cuda" if torch.cuda.is_available() else "cpu"


# <a id=2-1></a>
# ## 2.1 Pytesseract

# `Tesseract` не принимает `pdf-документы`. Конвертируем наш документ в набор изображений с помощью библиотеки `pdf2image`. Под капотом она использует программу `poppler`, которую нужно скачать на компьютер перед работой.

# In[5]:


images = convert_from_path(PDF_PATH, poppler_path=r"C:\poppler\Library\bin")


# Для распознавания нам нужно установать программу `tesseract`.

# In[6]:


# Путь к tesseract
pytesseract.pytesseract.tesseract_cmd = r"E:\tesseract\tesseract.exe"


# Реализуем функцию для распознавания текста.

# In[7]:


def ocr_pytesseract(img):
    return pytesseract.image_to_string(
        img,
        lang="rus",
        config="--oem 3 --psm 6"
    )


# ### Запуск распознавания

# In[8]:


get_ipython().run_cell_magic('time', '', '\nstart_time = time.time()\n\n# Обработка потоками\npytesseract_texts = Parallel(n_jobs=-1, backend="threading")(\n    delayed(ocr_pytesseract)(img)\n    for img in tqdm(images, desc=\'Pytesseract Recognition\')\n)\n\nelapsed_time_pytesseract = time.time() - start_time\n')


# ### Обзор распознавания

# ### Обзор распознавания

# In[25]:


print(pytesseract_texts[8])


# <a id=2-2></a>
# ## 2.2 Marker-PDF

# ### Запуск распознавания

# In[10]:


get_ipython().run_cell_magic('time', '', '\nstart_time = time.time()\n\n# Список обработок (берем лишь часть из пула возможных)\nprocessor_list = [\n    "marker.processors.order.OrderProcessor",\n    "marker.processors.line_merge.LineMergeProcessor",\n    "marker.processors.text.TextProcessor",\n]\n\nconverter = PdfConverter(\n    artifact_dict=create_model_dict(),\n    processor_list=processor_list,\n)\nrendered = converter(PDF_PATH)\ntext, _, _ = text_from_rendered(rendered, )\n\nelapsed_time_marker_surya = time.time() - start_time\n')


# ### Обзор распознавания

# In[11]:


print(text[3000:9000])


# In[12]:


torch.cuda.empty_cache()


# <a id=2-3></a>
# ## 2.3 Docling 

# ### Запуск распознавания

# In[13]:


get_ipython().run_cell_magic('time', '', '\nstart_time = time.time()\n\n# Обработка с Docling\nconverter = DocumentConverter()\ndoc = converter.convert(PDF_PATH).document\n\ntext = doc.export_to_markdown()\n\nelapsed_time_docling = time.time() - start_time\n')


# ### Обзор распознавания

# In[31]:


print(text[3500:9000])


# In[15]:


torch.cuda.empty_cache()


# <a id=2-4></a>
# ## 2.4 Nanonets-OCR

# ### Запуск распознавания

# In[16]:


# Иницализация модели
pipe = pipeline("image-text-to-text", model="nanonets/Nanonets-OCR2-3B", temperature=0.1)


# Модель может обрабатывать только 1 изображение, пройдемся циклом по всему документу.

# In[17]:


get_ipython().run_cell_magic('time', '', '\nstart_time = time.time()\n\nresults = []\n\nfor img in images:\n\n    messages = [\n    {\n        "role": "user",\n        "content": [\n            {"type": "image"},\n            {"type": "text", "text": "Extract text from this document"}\n        ]\n    }\n    ]\n\n    res = pipe(\n        images=img,\n        text=messages\n    )\n    results.append(res)\n\n# Объединяем текст\ntext = \'\'.join([page[0][\'generated_text\'][1][\'content\'] for page in results])\nelapsed_time_nanonets = time.time() - start_time\n')


# ### Обзор распознавания

# In[37]:


print(text[3000:])


# In[19]:


torch.cuda.empty_cache()


# <a id=2-5></a>
# ## 2.5 PaddleOCR-VL
# 
# 
# Попытка использования PaddleOCR-VL для извлечения текста из PDF оказалась неуспешной из-за ряда технических проблем. Основные ошибки были связаны с некорректной загрузкой токенизатора (отсутствие `tokenizer.model`), вызванной повреждённым или рассинхронизированным кэшем Hugging Face.
# 
# Дополнительно при использовании через библиотеку `PaddlePaddle` возникли проблемы с зависимостями, что препятствовало корректному запуску.
# 
# В результате модель не удалось стабильно запустить в текущем окружении, поэтому было принято решение использовать альтернативные решения.

# <a id=3></a>
# # 3. Сравнение решений

# In[20]:


results = pd.DataFrame({
    "Метод": [
        "Pytesseract",
        "Marker (Surya)",
        "Docling",
        "Nanonets OCR",
        "PaddleOCR-VL"
    ],
    "Время (сек)": [
        elapsed_time_pytesseract,
        elapsed_time_marker_surya,
        elapsed_time_docling,
        elapsed_time_nanonets,
        0.0
    ]
})

# сортировка по скорости
results = results.sort_values("Время (сек)")

results


# Самым быстрым решением стал `tesseract`! Не уступая по качеству другим более совремнным моделям, он отлично справился со своей задачей! 
