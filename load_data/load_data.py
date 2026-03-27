#!/usr/bin/env python
# coding: utf-8

# # Module A для табличных данных

# ### Загрузка CSV

# In[1]:


import pandas as pd

def load_csv(file_path, chunksize=None):
    """
    Загружает CSV файл.
    Если chunksize указан, возвращает генератор DataFrame для ленивой загрузки.
    """
    if chunksize:
        return pd.read_csv(file_path, chunksize=chunksize)
    else:
        return pd.read_csv(file_path)

# Пример:
# df = load_csv("data.csv")
# для больших файлов (возвращает список из DataFrame'ов): 
df_iter = load_csv("data.csv", chunksize=1000)

[i for i in df_iter][0].shape


# 
# ### Загрузка Excel

# In[2]:


import pandas as pd

def load_excel(file_path, sheet_name=0):
    """
    Загружает Excel файл (xls/xlsx).
    sheet_name=0 загружает первый лист, можно указать название листа.
    """
    return pd.read_excel(file_path, sheet_name=sheet_name)

# Пример:
df = load_excel("excel_file.xlsx")
df.sample(2)


# ### Загрузка TXT (как таблицу, разделитель по табуляции или другой)

# In[3]:


import pandas as pd

def load_txt(file_path, sep=',', chunksize=None):
    """
    Загружает TXT файл как таблицу.
    sep — разделитель (по умолчанию табуляция).
    chunksize — для ленивой загрузки больших файлов.
    """
    if chunksize:
        return pd.read_csv(file_path, sep=sep, chunksize=chunksize)
    else:
        return pd.read_csv(file_path, sep=sep)

# Пример:
df = load_txt("data.txt")
# для больших файлов: df_iter = load_txt("data.txt", chunksize=10000)
df.sample(3)


# ### Загрузка Parquet

# In[4]:


import pandas as pd

def load_parquet(file_path):
    """
    Загружает Parquet файл.
    """
    return pd.read_parquet(file_path)

# Пример:
df = load_parquet("data.parquet")
df.sample(3)


# ### Загрузка бинарных файлов

# In[5]:


import struct
import numpy as np

def load_bin_image(path):
    """
    Загружает бинарный файл с изображением в numpy.ndarray
    Формат:
    - magic (4 байта)
    - array_length (uint32)
    - version (uint8)
    - data_type_code (uint8)
    - reserved (2 байта)
    - data (array_length * dtype)
    """
    with open(path, 'rb') as file:
        # Заголовок
        magic = file.read(4)
        array_length = struct.unpack('<I', file.read(4))[0]
        version = struct.unpack('<B', file.read(1))[0]
        data_type_code = struct.unpack('<B', file.read(1))[0]
        reserved = file.read(2)

        # Таблица типов данных
        data_types = {
            0: np.uint8,
            1: np.float32,
            2: np.int32
        }

        if data_type_code not in data_types:
            raise ValueError(f"Неизвестный data_type_code: {data_type_code}")

        dtype = data_types[data_type_code]
        element_size = np.dtype(dtype).itemsize

        # Чтение данных (без лишних копий)
        raw = file.read(array_length * element_size)
        data = np.frombuffer(raw, dtype=dtype)

        # Восстановление формы (квадратное изображение)
        side = int(array_length ** 0.5)
        image = data.reshape((side, side))

    return image


# In[6]:


load_bin_image('data.mybin').shape


# При работе с бинарным файлом любого типа (изображение, таблица, аудио, временной ряд и т.д.) в первую очередь необходимо строго определить и описать его формат: структуру заголовка (сигнатура, версия, размеры, тип данных, служебные байты), порядок байтов (endianness) и способ хранения данных; затем файл следует читать последовательно в бинарном режиме (rb), извлекая метаданные через struct.unpack, после чего основные данные загружать в подходящую структуру без лишних копий (например, numpy.frombuffer или numpy.memmap при ограниченной оперативной памяти), приводя их к нужной форме (shape) и типу (dtype), а уже на этом уровне интерпретировать содержимое как изображение, таблицу (DataFrame), аудиосигнал или другой объект для дальнейшего анализа, визуализации или обучения моделей.

# ### Загрузка изображений (PIL)

# In[7]:


from PIL import Image
from pathlib import Path
import io
import os
import glob

def load_images(folder_path, extensions=None):
    """
    Загружает изображения из папки как PIL.Image
    """

    images = []
    for path in os.listdir(folder_path):
        img = Image.open(folder_path + path)
        images.append(img)

    return images

# Пример:
images = load_images("images/")
images[3]


# Если файлов очень много — лучше хранить пути, а не сами изображения.

# **Ленивая загрузка изображений через iterdir()**: При загрузке большого количества изображений (например, 15k) list comprehension загружает все файлы в память сразу — это приводит к лагам. Правильный подход — итеративная загрузка через `Path.iterdir()`:

# In[ ]:


# BAD — загружает все в память сразу
# images = [Image.open(p) for p in path]

# GOOD — итеративный обход
FOLDER = "images/"

dict_ = dict()
for item in Path(FOLDER).iterdir():
    if item.is_dir():
        for subitem in item.iterdir():
            if subitem.is_dir():
                for file in subitem.iterdir():
                    dict_[file] = Image.open(file)
    break


# **Фильтрация по расширению** — вместо ручной фильтрации используем `Path.glob()`:

# In[ ]:


# Получить только .jpg файлы
jpg_paths = list(Path(FOLDER).glob("*.jpg"))


# **Принцип «явное лучше неявного»**: при любой работе с изображениями всегда показывай промежуточный результат перед сохранением:

# In[ ]:


# BAD — не видим результат
# Image.open(io.BytesIO(content)).rotate(45, expand=False).save("1.png", format="PNG")

# GOOD — показываем промежуточный результат
img = Image.open(io.BytesIO(content)).rotate(45, expand=False)
display(img)
img.save("1.png", format="PNG")

# ### Загрузка аудио

# In[8]:


import librosa
import os
import glob

def load_audio(folder_path):
    """
    Загружает аудиофайлы с помощью librosa
    """

    audio_data = []
    for path in os.listdir(folder_path):
        y, sr = librosa.load(folder_path + path)
        audio_data.append({
            "path": path,
            "signal": y,
            "sr": sr
        })

    return audio_data

# Пример:
audios = load_audio("audio/")
audios[2]


# ### Загрузка видео

# In[13]:


import cv2
import os
import glob

def load_videos(folder_path):
    """
    Загружает видео как cv2.VideoCapture (ленивое чтение)
    """

    videos = []
    for path in os.listdir(folder_path):
        cap = cv2.VideoCapture(folder_path + path)
        videos.append({
            "path": path,
            "capture": cap
        })

    return videos


videos = load_videos('videos/')
videos[1]


# Работа с видео сводится к обработке каждого кадра, также как к изображениям.

# In[14]:


videos[1]['capture'].read()


# ### Загрузка байтовых файлов

# `io.BytesIO` — для бинарных данных (изображения, аудио); `io.StringIO` — для текстовых данных:

# In[ ]:


import io

FILE = "data/image.jpg"

with open(FILE, "rb") as f:
    content = f.read()

# Открыть изображение из байт
Image.open(io.BytesIO(content))

# Прочитать CSV из байт
pd.read_csv(io.StringIO(content.decode("utf-8")))


# ### Загрузка битого CSV

# **Способ 1**: pandas пропускает повреждённые строки:

# In[ ]:


import csv

df = pd.read_csv("data.csv", on_bad_lines="skip")
df.head(15)


# **Способ 2**: встроенный `csv` для точного контроля — находим строки с неправильным числом колонок:

# In[ ]:


with open("data.csv", "r") as f:
    reader = csv.reader(f)
    for i, line in enumerate(reader):
        if i == 0:
            cols_sz = len(line)
            continue
        if len(line) != cols_sz:
            print(i, line)


# ### Загрузка JSON / JSONL

# Сравнение библиотек по скорости:
# - **json** — стандартная библиотека
# - **orjson** — быстрее в ~5 раз (написана на Rust)
# - **ijson** — потоковая обработка очень больших JSON файлов

# In[ ]:


import json
import orjson

# Загрузка JSONL через pandas
df = pd.read_json("file.jsonl", lines=True)

# Построчная загрузка JSONL через orjson (быстрее)
data = []
with open("file.jsonl") as f:
    for line in f:
        data.append(orjson.loads(line))


# ### Загрузка PDF (pdfplumber)

# `pdfplumber` извлекает текст и изображения из PDF:

# In[ ]:


import pdfplumber
from tqdm import tqdm

imgs = []
with pdfplumber.open("file.pdf") as pdf:
    for page in tqdm(pdf.pages):
        txt = page.extract_text()
        img = page.images
        if img:
            for i in img:
                try:
                    imgs.append(
                        Image.open(io.BytesIO(i["stream"].rawdata))
                    )
                except:
                    pass


# ### Загрузка строк (ast.literal_eval)

# `ast.literal_eval` — безопасное преобразование строкового представления Python-объектов. В отличие от `eval()` работает только с литералами (строки, числа, списки, словари, bool, None):

# In[ ]:


import ast

ast.literal_eval("True")          # True

string = "[1, 2, 3, 5]"
ast.literal_eval(string)          # [1, 2, 3, 5]

