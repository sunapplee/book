#!/usr/bin/env python
# coding: utf-8

# # Сравнение OCR-решений для обработки PDF. Часть 2
# 
# В ноутбуке проводится сравнение OCR-решений для извлечения текста из PDF-документа `text1.pdf`. Рассмотрены инструменты: **PP-OCRv5**, **GLM-OCR**, **DeepSeek OCR** (не запущен из-за проблем с зависимостями).
# 
# Оценка выполняется по скорости и качеству распознавания. Реализована обработка изображений и замер времени работы каждого метода.

# ## Содержание
# 
# * [Импорт библиотек](#0)
# 
# * [1. Данные для анализа](#1)
# * [2. OCR решения](#2)
#     * [2.1 PP-OCRv5](#2-1)
#     * [2.2 GLM-OCR](#2-2)
#     * [2.3 DeepSeek OCR](#2-3)
# * [3. Сравнение решений](#3)
# 
# 

# ###

# <a id=0></a>
# ## Импорт библиотек

# In[1]:


import warnings
warnings.filterwarnings("ignore")


# In[17]:


import pdfplumber
import pandas as pd

from paddleocr import PaddleOCR

from glmocr import GlmOcr

from transformers import AutoModel, AutoTokenizer
import torch

import sys
import re
from pdf2image import convert_from_path
import os
import io
import time
import base64
from io import BytesIO
from tqdm import tqdm


# ###

# <a id=1></a>
# # 1. Данные для анализа
# 
# В качестве документа, на котором будут тестироваться решения, возьмем `text.pdf`.

# In[3]:


PDF_PATH = 'text1.pdf'


# In[4]:


with pdfplumber.open(PDF_PATH) as pdf:
    first_page = pdf.pages[0]
    print(first_page.extract_text().strip()[:100])


# В документе присутствует текстовый слой. Однако такое бывает **не всегда**. В связи с этом нужен `OCR инструмент`, обладающий хорошим качеств распознавания и скоростью обработки.

# `Многие алгоритмы` не принимают `pdf-документы`. Конвертируем наш документ в набор изображений с помощью библиотеки `pdf2image`. Под капотом она использует программу `poppler`, которую нужно скачать на компьютер перед работой.

# In[5]:


images = convert_from_path(PDF_PATH, poppler_path=r"C:\poppler\Library\bin")


# ###

# <a id=2></a>
# # 2. OCR решения 

# <a id=2-1></a>
# ## 2.1 PP-OCRv5
# 
# Попытка использования PaddleOCR-VL для извлечения текста из PDF оказалась неуспешной. Основные проблемы были связаны с ошибками загрузки токенизатора (отсутствие `tokenizer.model`, вероятно из-за повреждённого кэша Hugging Face), а также с конфликтами зависимостей в PaddlePaddle. Кроме того, VL-версия нестабильно работает на Windows, что не позволило добиться корректного запуска.
# 
# В результате было принято решение перейти на более стабильные решения. Для анализа структуры документа используется модель PP-DocLayout-L (~123 МБ) с зафиксированными версиями зависимостей:
# 
# ```
# pip install paddlepaddle==3.2.0 paddleocr==3.3.3
# ```
# 
# Также важно учитывать ограничение PaddlePaddle: некорректная работа с путями, содержащими кириллицу. Все модели должны храниться в директориях с латиницей, например:
# 
# ```
# C:/models/paddle/PP-OCRv5_server_det
# ```
# 
# Для OCR используются локальные модели с явным указанием путей:
# 
# ```python
# text_detection_model_dir = r"C:\models\paddle\PP-OCRv5_server_det"
# text_recognition_model_dir = r"C:\models\paddle\cyrillic_PP-OCRv3_mobile_rec"
# ```
# 
# Таким образом, стабильная работа достигается за счёт отказа от VL-модели, контроля версий и использования путей без Unicode.

# ### Запуск распознавания

# In[6]:


# Инциализируем модель
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    text_detection_model_name="PP-OCRv5_server_det",
    text_detection_model_dir=r"C:\models\paddle\PP-OCRv5_server_det",
    text_recognition_model_name="cyrillic_PP-OCRv5_mobile_rec",
    text_recognition_model_dir=r"C:\models\paddle\cyrillic_PP-OCRv5_mobile_rec",
)


# In[7]:


st_time = time.time()
result = ocr.predict('text1.pdf')
paddle_time = time.time() - st_time


# ### Обзор распознавания

# In[8]:


# Объединяем в связный текст
text = ' '.join([' '.join(page['rec_texts']) for page in result])

text[5000:7000]


# <a id=2-2></a>
# ## 2.2 GLM-OCR
# 

# ## Локальный запуск GLM-OCR через Ollama
# 
# 1. **Скачать модель**
# 
# ```bash
# ollama pull glm-ocr:latest
# ```
# 
# 2. **Запустить Ollama**
# 
# ```bash
# ollama serve
# ```
# 
# API доступен на:
# 
# ```
# http://localhost:11434
# ```
# 
# 3. **config.yaml**
# 
# ```yaml
# pipeline:
#   maas:
#     enabled: false
#   
#   ocr_api:
#     api_host: localhost
#     api_port: 11434
#     api_path: /api/generate
#     model: glm-ocr:latest
#     api_mode: ollama_generate
#   
#   enable_layout: false
# ```

# ### Запуск распознавания

# Модель принимает путь к изображению или его байтового представление. Представим наши сканы в формате `BytesIO`.

# In[9]:


# Функция обработки изображения
def get_bytes_image(img):
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


# In[10]:


start_time = time.time()

# Инициализация через кастомный конфиг (с олламой)
with GlmOcr(config_path="config.yaml") as parser:
    # Передаем список изображений
    result = parser.parse([get_bytes_image(img) for img in images])
    # Объединяем текст
    text = '\n'.join([page.markdown_result for page in result])

glm_time = time.time() - start_time


# ### Обзор распознавания

# In[11]:


text[3000:7000]


# <a id=2-3></a>
# ## 2.3 DeepSeek OCR

# требовательный к версии трансформерс
# pip install transformers==4.46.0

# In[12]:


# Инициализация модели
model_name = 'deepseek-ai/DeepSeek-OCR-2'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, _attn_implementation='eager', trust_remote_code=True, use_safetensors=True)
model = model.eval().cuda().to(torch.bfloat16)


# ### Запуск распознавания

# Модель не возвращает результат напрямую, а выводит его в виде сырых логов в stdout (включая служебные токены, bbox и отладочную информацию). Поэтому мы перехватываем этот вывод, извлекаем текст и очищаем его от лишних данных, оставляя только полезное содержимое.

# In[13]:


# Функция перехватки вывода
def infer_capture(model, tokenizer, **kwargs):
    buffer = io.StringIO()
    sys_stdout = sys.stdout
    sys.stdout = buffer
    try:
        model.infer(tokenizer, **kwargs)
    finally:
        sys.stdout = sys_stdout

    return buffer.getvalue()

deepseek_time = time.time() - start_time


# In[14]:


# Функци отчистки логов
def extract_clean_text(text):
    # 1. Убираем блок с ===== и логами
    text = re.sub(r"=+\n.*?=+\n", "", text, flags=re.DOTALL)

    # 2. Убираем строки с BASE / PATCHES
    text = re.sub(r"BASE:.*\n", "", text)
    text = re.sub(r"PATCHES:.*\n", "", text)

    # 3. Убираем bbox
    text = re.sub(r"<\|det\|>.*?<\|/det\|>", "", text)

    # 4. Убираем служебные токены (<|ref|>, text, image)
    text = re.sub(r"<\|.*?\|>", "", text)
    text = re.sub(r"\b(text|image)\b", "", text)

    # 5. Убираем лишние пустые строки
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()


# In[19]:


start_time = time.time()

prompt = "<image>\n<|grounding|>Convert the document to markdown. "
texts = []
# Проходимся по каждой странице
for img in tqdm(images):

    tmp_path = "temp.jpg"
    img.save(tmp_path)

    res = infer_capture(
        model,
        tokenizer,
        prompt=prompt,
        image_file=tmp_path,
        output_path="outputs",
        save_results=False,
        base_size=1024,
        image_size=768,
        crop_mode=True
    )
    # Сохраняем распознанный текст
    texts.append(extract_clean_text(res))


# In[20]:


text = '\n'.join(texts)


# ### Обзор распознавания

# In[21]:


text[10000:16000]


# ###

# <a id=3></a>
# # 3. Сравнение решений

# In[22]:


results = pd.DataFrame({
    "Метод": [
        "PP-OCRv5",
        "GLM-OCR",
        "Deepseek-OCR"
    ],
    "Время (сек)": [
        paddle_time,
        glm_time,
        deepseek_time,
    ]
})

# сортировка по скорости
results = results.sort_values("Время (сек)")

results


# Самым быстрым решением стал `GLM-OCR`! Не уступая по качеству другим моделям моделям, он отлично справился со своей задачей! Однако требует дополнительного развертывания в виде backend для работы.
