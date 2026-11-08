#!/usr/bin/env python
# coding: utf-8

# # Задание по лекции от 15.02.2026
# 
# Разбор 1, 2 Модуля **DIGITAL SKILLS 2023**
# 
# Задание состоит за **5 частей**:
# 1. Загрузка данных
# 2. Предобработка данных
# 3. Расширение набора данных с помощью методов аугментации
# 4. Сегментация изображений «без учителя»
# 5. Формирование итогового набора расширенных и предобработанных данных

# ## Содержание
# 
# * [Импорт библиотек.](#0)
# * [1. Загрузка данных.](#1)
# * [2. Предобработка данных. Анализ и корректировка изображений.](#2)
#     * [2.1. Алгоритм нахождения облачности на изображении.](#2-1)
#     * [2.2. Нахождение облачности во всем наборе.](#2-2)
#     * [2.3. Визуализация облачности.](#2-3)
#     * [2.4. Подготовка описания данных по указанной форме.](#2-4)
#     * [2.5. Сохранение таблицы в pdf.](#2-5)
#  
# * [3. Расширение набора данных с помощью методов аугментации.](#3)
#     * [3.1. Сформирование 5 комбинаций признаков.](#3-1)
#     * [3.2. Аугментация](#3-2)
#         * [3.2.1. Обычные аугментации](#3-2-1)
#         * [3.2.2. Аугментации с маской](#3-2-2)
# 
# 
# * [4. Сегментация изображений «без учителя».](#4)
#     * [4.1. Обучение алгоритма.](#4-1)
#     * [4.2. Визуализация кластеризации.](#4-2)
#     * [4.3. Описание кластеров.](#4-3)
#     * [4.4. Расчет метрик.](#4-4)
# * [5. Формирование итогового набора расширенных и предобработанных данных](#5)

# ###

# <a id=0></a>
# ## Импорт библиотек

# In[1]:


import numpy as np
import pandas as pd

import random
from PIL import Image, ImageEnhance

import os
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from docx import Document
from htmldocx import HtmlToDocx
import shutil

from matplotlib import pyplot as plt

from sklearn.cluster import MiniBatchKMeans
from skimage import morphology
from skimage.measure import regionprops, label

import cv2
import torchvision.transforms.v2 as tfs
import torchvision.transforms.functional as F
import albumentations as A
from segmentation_models_pytorch import metrics
import torch


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# ###

# <a id=1></a>
# # 1. Загрузка данных

# Загрузим полностью основной набор изображений и масок из папки ```train_new```.

# In[3]:


# Директория данных
image_paths = Path('train_new/images')


# In[4]:


data = {}

# Проходимся по каждому изображению
for path in tqdm(image_paths.iterdir(), total=4620):
    # Для каждого изображения находим маску
    name = path.stem
    label_path = f'train_new/labels/label{name[5:]}.jpg'

    # Используем контекстный менеджер, чтобы закрывать файл автоматически
    with Image.open(path) as img:
        img = img.copy()  # копируем в память

    # Используем контекстный менеджер, чтобы закрывать файл автоматически
    with Image.open(label_path) as label_:
        label_ = label_.copy()   # копируем в память

    # Сохраняем изображению и маску вместе
    data[name] = {
        'image': img,
        'label': label_
    }


# Проверим загрузку данных.

# In[5]:


# Берем случайное изображение, также достаем маску
image, label_ = data['image_image_01010_20210720_const_0325'].values()

fig, axs = plt.subplots(1, 2, figsize=(8, 4))

# Визуализируем изображение с маской
for i, (ax, img) in enumerate(zip(axs, [image, label_])):
    ax.set_title('Исходное изображение' if i == 0 else 'Маска')
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])



# ###

# <a id=2></a>
# # 2. Предобработка данных. Анализ и корректировка изображений

# <a id=2-1></a>
# ## 2.1. Алгоритм нахождения облачности на изображении. 
# 
# Каждый пиксель изображения в градациях серого принимает значение от 0 до 255, где 0 соответствует черному цвету, а 255 — белому. Поскольку облака на спутниковых снимках обычно имеют высокую яркость, можно предположить, что пиксели с интенсивностью, близкой к 255, с большой вероятностью соответствуют облачным областям.

# Проверим данную гипотезу экспериментально, применив пороговую сегментацию.

# In[6]:


# Берем случайное изображение
image, _ = data['image_image_00001_20210420_const_0016'].values()

# Переводим изображение в градации серого
image_gray = image.convert('L')

# Определяем маску облачности по порогу 180 в градациях серого
cloud_mask = (np.asarray(image_gray) > 180).astype('int8')


# Визуализируем исходное изображение с маской облачности.

# In[7]:


fig, axs = plt.subplots(1, 2, figsize=(8, 4))

# Визуализируем изображение с маской облачности
for i, (ax, img) in enumerate(zip(axs, [image, cloud_mask])):
    ax.set_title('Исходное изображение' if i == 0 else 'Маска облачности')
    ax.imshow(img, cmap='binary')
    ax.set_xticks([])
    ax.set_yticks([])


# Как видим из визуализации, алгоритм отрабатывает отлично.

# <a id=2-2></a>
# ## 2.2. Нахождение облачности во всем наборе

# Пройдемся по всем изображениям и определим долю облачности в каждом изображении.

# In[8]:


# Анализиурем каждое изображение
for i in tqdm(data):
    img = data[i]['image']

    # Переводим изображение в градации серого
    image_gray = img.convert('L')

    # Определяем маску облачности по порогу 180 в градациях серого
    cloud_mask = (np.asarray(image_gray) > 180).astype('int8')

    # Применяем морфологические преобразования, с помощью библиотеки skimage
    # для улучшения качества маски
    cloud_mask = morphology.closing(morphology.opening(cloud_mask))

    # Оценим долю облачности на каждом найденном изображении.
    cloudness = cloud_mask.sum() / (cloud_mask.shape[0] * cloud_mask.shape[1]) * 100

    # Обновляем информацию в наборе данных
    data[i]['cloud_mask'] = cloud_mask
    data[i]['cloudness'] = cloudness


# <a id=2-3></a>
# 
# ## 2.3. Визуализация облачности

# В качестве проверки выведем исходные изображения из основного набора, содержащие облачность и полученные маски облачности для этих изображений.  А
# также процент пикселей, принадлежащих облачности для каждого изображения.

# In[9]:


n_imgs = 1

# Проходимся по набору данных
for i in data:

    image = data[i]['image']
    cloudness = data[i]['cloudness']
    cloud_mask = data[i]['cloud_mask']

    # Работаем только с изображениями, содержащими облачность
    if cloudness < 1:
        continue

    # Визуализируем изображение с маской облачности
    fig, axs = plt.subplots(1, 2, figsize=(6, 2))
    for i, (ax, img) in enumerate(zip(axs, [image, cloud_mask])):
        ax.set_title('Исходное изображение' if i == 0 else f'Процент облачности: {round(cloudness, 2)}%')
        ax.imshow(img, cmap='binary')
        ax.set_xticks([])
        ax.set_yticks([])
    fig.show()

    # Визуализируем больше 10 изображений
    if n_imgs > 11:
        break
    n_imgs += 1


# <a id=2-4></a>
# ## 2.4. Подготовка описания данных по указанной форме:
# 
# 
# <table border="1" cellpadding="6" cellspacing="0">
#     <thead>
#         <tr>
#             <th>№</th>
#             <th>Файл</th>
#             <th>Дата</th>
#             <th>Есть ли с/х поле?</th>
#             <th>% с/х угодий</th>
#             <th>Количество областей (с/х поля)</th>
#             <th>Наличие облачности</th>
#             <th>Доля облачности</th>
#         </tr>
#     </thead>
#     <tbody>
#         <tr>
#             <td> </td>
#             <td></td>
#             <td></td>
#             <td></td>
#             <td></td>
#             <td></td>
#             <td></td>
#             <td></td>
#         </tr>
# </table>
# 

# #### Перед тем как собирать таблицу, разберем, как определять С/Х угодья.
# 
# Сами угодья представлены в папки labels, их мы уже загрузили.

# In[10]:


# Берем случайное изображение, также достаем маску
image, label_mask, _, _ = data['image_image_01010_20210720_const_0325'].values()

# Преобразуем в оттенки серого
label_gray = label_mask.convert('L')
label_gray


# С/Х поле представлено в виде белого пикселя в градациях серого (255). Отнормируем маску, чтобы поле означало ```1```, а любой другой объект ```0```.

# In[11]:


label_norm = (np.asarray(label_gray) > 155).astype('int8')


# Определяем наличие С/Х поля, по наличию поля в маске. Если сумма пикселей > 0, то считаем, что угодья есть.

# In[12]:


label_norm.sum() > 0


# Для подсчета % С/Х угодий, подсчитаем долю поля в маске

# In[13]:


label_norm.sum() / (label_norm.shape[0] * label_norm.shape[1]) * 100


# In[14]:


label_norm


# Количество областей (с/х полей) будем определять с помощью метода ```skimage.measure.regionprops```.

# In[15]:


# Определяем кол-во областей
labels = label(label_norm)
regions = regionprops(labels)

print(f'Областей на маске: {len(regions)}')


# ### Подготовка таблицы
# 
# После того как определились с инструментами для работы с изображениями, начнем собирать таблицу.

# In[16]:


table = []

# Проходимся по каждому изображению
for indx, i in enumerate(tqdm(data)):
    image = data[i]['image']
    date = datetime.strptime(i.split('_')[3], '%Y%m%d')
    label_mask = data[i]['label']
    cloudness = data[i]['cloudness']
    cloud_mask = data[i]['cloud_mask']

    # Преобразуем маску c угодьями в оттенки серого
    label_gray = label_mask.convert('L')
    # Нормируем маску
    label_norm = (np.asarray(label_gray) > 155).astype('int8')
    # Определяем, есть ли угодья
    is_field = label_norm.sum() > 0
    # Определяем процент С/Х угодий
    field_prop = label_norm.sum() / (label_norm.shape[0] * label_norm.shape[1]) * 100
    # Определяем кол-во областей
    labels = label(label_norm)
    regions = regionprops(labels)
    count_fields = len(regions)

    # Добавляем все в новый словарь
    table.append({
        '№': indx,
        'Файл': f'{i}.jpg',
        'Дата': date.date(),
        'Есть ли с/х поле?': 'Да' if is_field else 'Нет',
        '% с/х угодий': f'{field_prop:.2f}%',
        'Количество областей (с/х поля)': count_fields,
        'Наличие облачности': 'Есть' if cloudness != 0 else 'Нет',
        'Доля облачности': f'{cloudness:.2f}%'
    })


# Итоговый вид таблицы:

# In[17]:


table_df = pd.DataFrame(table)

table_df.sample(5)


# <a id=2-5></a>
# ## 2.5. Сохранение таблицы в pdf
# 
# Сохраним таблицу в excel, дальше конвертируем в pdf. Это самый быстрый способ сохранения таблицы.

# In[18]:


table_df.to_excel("table.xlsx", index=False)


# ###

# <a id=3></a>
# # 3. Расширение набора данных с помощью методов аугментации

# <a id=3-1></a>
# ## 3.1. Сформирование 5 комбинаций признаков. 
# 
# 1. нет облачности, малая доля (<3%) с/х угодий
# 2. малая доля (<3%) облачности, 3 с/х области
# 3. средняя доля облачности (25-75%), есть с/х поле
# 4. нет облачности, средняя доля с/х угодий (25-75%)
# 5. высокая доля облачности (>75%), нет с/х полей

# In[19]:


# 1 признак
feature_1 = table_df[(table_df['Наличие облачности'] == 'Нет') & (table_df['% с/х угодий'].str.rstrip('%').astype(float) < 3)]
feature_1['feature'] = 1
print(f'Признак 1, количество строк:', feature_1.shape[0])

# 2 признак
feature_2 = table_df[(table_df['Доля облачности'].str.rstrip('%').astype(float) < 3) & (table_df['Количество областей (с/х поля)'] == 3)]
feature_2['feature'] = 2
print(f'Признак 2, количество строк:', feature_2.shape[0])

# 3 признак
feature_3 = table_df[(table_df['Доля облачности'].str.rstrip('%').astype(float) > 25) & \
                (table_df['Доля облачности'].str.rstrip('%').astype(float) < 75) & \
                (table_df['Есть ли с/х поле?'] == 'Да')]
feature_3['feature'] = 3
print(f'Признак 3, количество строк:', feature_3.shape[0])

# 4 признак
feature_4 = table_df[(table_df['Наличие облачности'] == 'Нет') & \
                (table_df['% с/х угодий'].str.rstrip('%').astype(float) > 25) & \
                (table_df['% с/х угодий'].str.rstrip('%').astype(float) < 75)]
feature_4['feature'] = 4
print(f'Признак 4, количество строк:', feature_4.shape[0])

# 5 признак
feature_5 = table_df[(table_df['Доля облачности'].str.rstrip('%').astype(float) > 75) & \
                        (table_df['Есть ли с/х поле?'] == 'Нет')]
feature_5['feature'] = 5
print(f'Признак 5, количество строк:', feature_5.shape[0])


# In[20]:


# Сформируем итоговой датасет для аугментации
featured_table = pd.concat([feature_1, feature_2, feature_3, feature_4, feature_5])
featured_table.sample(4)


# <a id=3-2></a>
# ## 3.2. Аугментация

# Реализуем функции аугментации на различных библиотеках:
# - PIL
# - cv2
# - torchvision
# - albumentations
# 
# А также реализуем функции аугментации для изображений с учетом сегментационных масок с помощью:
# - torchvision
# - albumentations

# <a id=3-2-1></a>
# ## 3.2.1. Обычные аугментации

# ### PIL аугментации

# In[21]:


def pil_augmentations(table, data):

    # Сюда будем сохранять все аугментации
    aug_data = {}

    for i, row in tqdm(table.iterrows(), total=2059):
        img_path = row['Файл']
        # Открываем загруженную картинку
        image = data[img_path[:-4]]['image']

        # Зеркальное отражение по горизонтали
        mirroring = image.transpose(Image.FLIP_LEFT_RIGHT)

        # Поворот
        angles = [90, 180, 270]
        rand_angle = random.choice(angles)
        turning = image.rotate(rand_angle, expand=False)

        # Приближение/Отдаление
        w, h = image.size
        # Определяем коэффициент зума, >1 - приближение, <1 - отдаление
        scale = random.uniform(0.7, 1.5)
        # Новый размер
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = image.resize((new_w, new_h), Image.BICUBIC)

        if scale > 1:
            # Приближения → случайный crop до исходного размера
            left = random.randint(0, new_w - w)
            top = random.randint(0, new_h - h)
            zooming = resized.crop((left, top, left + w, top + h))

        else:
            # Отдаления → вставляем в черный фон
            zooming = Image.new("RGB", (w, h))
            left = (w - new_w) // 2
            top = (h - new_h) // 2
            zooming.paste(resized, (left, top))

        # Смена яркости
        # Случайный коэффициент яркости
        scale = random.uniform(0.5, 1.5)
        brightning = ImageEnhance.Brightness(image).enhance(scale)

        # Смена контрастности
        # Случайный коэффициент контрастности
        scale = random.uniform(0.5, 1.5)
        contrasting = ImageEnhance.Contrast(image).enhance(scale)

        # Случайные обрезки
        left = random.randint(0, w)
        top = random.randint(0, h)
        cropping = resized.crop((left, top, left + w, top + h))

        # Случайный наклон
        k = k = random.uniform(-0.3, 0.3)  # коэффициент наклона
        matrix = (1, k, 0,
                  k, 1, 0)
        shearing = image.transform(image.size, Image.AFFINE, matrix)

        # Случайное вращение
        angle =  random.randint(1, 360)
        rotating = image.rotate(angle, expand=False)

        aug_data[img_path] = {
            'original': image,
            'mirroring': mirroring,
            'turning': turning,
            'zooming': zooming,
            'brightning': brightning,
            'contrasting': contrasting,
            'cropping': cropping,
            'shearing': shearing,
            'rotating': rotating
        }

    return aug_data


# Проверка функции

# In[22]:


pil_augs = pil_augmentations(featured_table, data)


# In[23]:


sample = pil_augs['image_image_01007_20210720_const_0284.jpg']
sample


# In[24]:


fig, axs = plt.subplots(3, 3, figsize=(9, 9))

for ax, (name, value) in zip(axs.ravel(), sample.items()):

    ax.imshow(value)
    ax.set_title(name)

    ax.set_xticks([])
    ax.set_yticks([])


# Как видно из изображений, аугментации отработали успешно.

# ### cv2 аугментации

# In[25]:


def cv2_augmentations(table, data):

    # Сюда будем сохранять все аугментации
    aug_data = {}

    for i, row in tqdm(table.iterrows(), total=2059):
        img_path = row['Файл']
        # Открываем загруженную картинку
        image = data[img_path[:-4]]['image']
        # Переводим в numpy array, чтобы cv2 мог работать с изображениями
        image = np.asarray(image)

        # Зеркальное отражение по горизонтали
        mirroring = cv2.flip(image, 0)

        # Поворот
        angles = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
        rand_angle = random.choice(angles)
        turning = cv2.rotate(image, rand_angle)

        # Приближение/Отдаление
        h, w = image.shape[:2]
        # Определяем коэффициент зума, >1 - приближение, <1 - отдаление
        scale = random.uniform(0.7, 1.5)
        # Новый размер
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h))

        if scale > 1:
            # Приближения → случайный crop до исходного размера
            left = random.randint(0, new_w - w)
            top = random.randint(0, new_h - h)
            zooming = resized[top:top+h, left:left+w]

        else:
            # Отдаления → вставляем в черный фон
            zooming = np.zeros((h, w, 3), dtype=image.dtype)
            left = (w - new_w) // 2
            top = (h - new_h) // 2
            zooming[top:top+new_h, left:left+new_w] = resized

        # Смена яркости
        # Случайный коэффициент яркости
        scale = random.randint(-70, 70)
        brightning = cv2.convertScaleAbs(image, alpha=1.0, beta=scale)

        # Смена контрастности
        # Случайный коэффициент контрастности
        scale = random.uniform(0.2, 1.8)
        contrasting = cv2.convertScaleAbs(image, alpha=scale, beta=0)

        # Случайные обрезки
        scale = random.uniform(0.3, 0.9)
        cropping = np.zeros((h, w, 3), dtype=image.dtype)
        crop_w = int(w * scale)
        crop_h = int(h * scale)
        left = random.randint(0, w - crop_w)
        top = random.randint(0, h - crop_h)
        cropping[top:top+crop_h, left:left+crop_w] = image[top:top+crop_h, left:left+crop_w]

        # Случайный наклон
        k = random.uniform(-0.5, 0.5)  # коэффициент наклона
        M = np.float32([
        [1, k, 0],
        [k, 1, 0]])
        shearing = cv2.warpAffine(image, M, (w, h))

        # Случайное вращение
        angle = random.uniform(-30, 30)

        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)

        rotating = cv2.warpAffine(image, M, (w, h))

        aug_data[img_path] = {
            'original': image,
            'mirroring': mirroring,
            'turning': turning,
            'zooming': zooming,
            'brightning': brightning,
            'contrasting': contrasting,
            'cropping': cropping,
            'shearing': shearing,
            'rotating': rotating
        }

    return aug_data


# Проверка функции.

# In[26]:


cv2_augs = cv2_augmentations(featured_table, data)


# In[27]:


sample = cv2_augs['image_image_01029_20210720_const_0082.jpg']


# In[28]:


fig, axs = plt.subplots(3, 3, figsize=(9, 9))

for ax, (name, value) in zip(axs.ravel(), sample.items()):

    ax.imshow(value)
    ax.set_title(name)

    ax.set_xticks([])
    ax.set_yticks([])


# Как видно из изображений, аугментации отработали успешно.

# ### torchvision аугментации

# In[29]:


def torchvision_augmentations(table, data):

    # Сюда будем сохранять все аугментации
    aug_data = {}

    for i, row in tqdm(table.iterrows(), total=2059):
        img_path = row['Файл']
        # Открываем загруженную картинку
        image = data[img_path[:-4]]['image']

        # Сохарняем размеры изображений
        h, w = image.size

        # Зеркальное отражение по горизонтали
        transform = tfs.RandomHorizontalFlip(p=1)
        mirroring = transform(image)

        # Поворот
        angles = [90, 180, 270]
        angle = random.choice(angles)
        turning = F.rotate(image, angle)

        # Случайное Приближение/Отдаление
        scale = random.uniform(0.5, 3)
        if scale < 1:
            # Приближение
            transform = tfs.RandomResizedCrop((w, h), scale=(scale, scale))
        else:
            # Отдаление
            transform = tfs.Compose([
                tfs.RandomZoomOut(p=1, side_range=(scale, scale)),
                tfs.Resize((w, h))])

        zooming = transform(image)

        # Смена яркости
        transform = tfs.ColorJitter(brightness=(0, 2))
        brightning = transform(image)

        # Смена контрастности contrast
        # Случайный коэффициент контрастности
        transform = tfs.ColorJitter(contrast=(0, 2))
        contrasting = transform(image)

        # Случайные обрезки
        transform = tfs.RandomResizedCrop(size=(w, h))
        cropping = transform(image)

        # Случайный наклон
        transform = tfs.RandomAffine(degrees=0, shear=(-90, 90))
        shearing = transform(image)

        # Случайное вращение
        transform = tfs.RandomRotation(degrees=(-90, 90))
        rotating = transform(image)


        aug_data[img_path] = {
            'original': image,
            'mirroring': mirroring,
            'turning': turning,
            'zooming': zooming,
            'brightning': brightning,
            'contrasting': contrasting,
            'cropping': cropping,
            'shearing': shearing,
            'rotating': rotating
        }

    return aug_data


# Проверка функции.

# In[30]:


tv_augs = torchvision_augmentations(featured_table, data)


# In[31]:


sample = tv_augs['image_image_01011_20210720_const_0318.jpg']


# In[32]:


fig, axs = plt.subplots(3, 3, figsize=(9, 9))

for ax, (name, value) in zip(axs.ravel(), sample.items()):

    ax.imshow(value)
    ax.set_title(name)

    ax.set_xticks([])
    ax.set_yticks([])


# Как видно из изображений, аугментации отработали успешно.

# ### albumentation аугментации

# In[33]:


def albumentations_augmentations(table, data):

    # Сюда будем сохранять все аугментации
    aug_data = {}

    for i, row in tqdm(table.iterrows(), total=2059):
        img_path = row['Файл']
        # Открываем загруженную картинку
        image = data[img_path[:-4]]['image']
        # Переводим в numpy array, чтобы albumentations мог работать с изображениями
        image = np.asarray(image)
        h, w = image.shape[:2]

        # Зеркальное отражение по горизонтали
        transform = A.HorizontalFlip(p=1.0)
        mirroring = transform(image=image)['image']

        # Поворот
        transform = A.RandomRotate90(p=1.0)
        turning = transform(image=image)['image']

        # Случайное Приближение/Отдаление
        transform = A.Affine(
                scale=(0.3, 2),   # 0.3 = Отдаление, 2 = Приближение
                p=1.0
            )
        zooming = transform(image=image)['image']

        # Смена яркости
        transform = A.RandomBrightnessContrast(brightness_limit=(-0.5, 0.5), p=1)
        brightning = transform(image=image)['image']

        # Смена контрастности contrast
        transform = A.RandomBrightnessContrast(contrast_limit=(-0.5, 0.5), p=1)
        contrasting = transform(image=image)['image']

        # Случайные обрезки
        transform = A.RandomResizedCrop(
                    size=(h, w),
                    scale=(0.2, 1.0),     # какая часть изображения будет взята
                    p=1.0)
        cropping = transform(image=image)['image']

        # Случайный наклон
        transform = A.Affine(
                shear=(-60, 60),
                p=1.0
            )
        shearing = transform(image=image)['image']

        # Случайное вращение
        transform = A.Rotate(limit=(-90, 90), p=1)
        rotating = transform(image=image)['image']


        aug_data[img_path] = {
            'original': image,
            'mirroring': mirroring,
            'turning': turning,
            'zooming': zooming,
            'brightning': brightning,
            'contrasting': contrasting,
            'cropping': cropping,
            'shearing': shearing,
            'rotating': rotating
        }

    return aug_data


# Проверим функцию.

# In[34]:


A_augs = albumentations_augmentations(featured_table, data)


# In[35]:


sample = A_augs['image_image_01011_20210720_const_0318.jpg']


# In[36]:


fig, axs = plt.subplots(3, 3, figsize=(9, 9))

for ax, (name, value) in zip(axs.ravel(), sample.items()):

    ax.imshow(value)
    ax.set_title(name)

    ax.set_xticks([])
    ax.set_yticks([])


# <a id=3-2-2></a>
# ## 3.2.2. Аугментации с маской

# ### torchvision аугментации с маской

# In[37]:


def torchvision_augmentations_with_mask(table, data):

    # Сюда будем сохранять все аугментации
    aug_data = {}

    for i, row in tqdm(table.iterrows(), total=2059):
        img_path = row['Файл']
        # Открываем загруженную картинку и маску
        image = data[img_path[:-4]]['image']
        label_field = data[img_path[:-4]]['label']

        # Сохраняем размеры изображений
        h, w = image.size

        # Зеркальное отражение по горизонтали
        transform = tfs.RandomHorizontalFlip(p=1)
        mirroring, mirroring_mask = transform(image, label_field)

        # Поворот
        angles = [90, 180, 270]
        angle = random.choice(angles)
        transform = tfs.RandomRotation(
            degrees=(angle, angle),   # фиксированный угол
        )
        turning, turning_mask = transform(image, label_field)

        # Случайное Приближение/Отдаление
        scale = random.uniform(0.5, 3)
        if scale < 1:
            # Приближение
            transform = tfs.RandomResizedCrop((w, h), scale=(scale, scale))
        else:
            # Отдаление
            transform = tfs.Compose([
                tfs.RandomZoomOut(p=1, side_range=(scale, scale)),
                tfs.Resize((w, h))])

        zooming, zooming_mask = transform(image, label_field)

        # Смена яркости
        transform = tfs.ColorJitter(brightness=(0, 2))
        brightning, brightning_mask = transform(image, label_field)

        # Смена контрастности contrast
        # Случайный коэффициент контрастности
        transform = tfs.ColorJitter(contrast=(0, 2))
        contrasting, contrasting_mask = transform(image, label_field)

        # Случайные обрезки
        transform = tfs.RandomResizedCrop(size=(w, h))
        cropping, cropping_mask = transform(image, label_field)

        # Случайный наклон
        transform = tfs.RandomAffine(degrees=0, shear=(-30, 30))
        shearing, shearing_mask = transform(image, label_field)

        # Случайное вращение
        transform = tfs.RandomRotation(degrees=(-90, 90))
        rotating, rotating_mask = transform(image, label_field)


        aug_data[img_path] = {
            'original': image,
            'original_mask': label_field,
            'mirroring': mirroring,
            'mirroring_mask': mirroring_mask,
            'turning': turning,
            'turning_mask': turning_mask,
            'zooming': zooming,
            'zooming_mask': zooming_mask,
            'brightning': brightning,
            'brightning_mask': brightning_mask,
            'contrasting': contrasting,
            'contrasting_mask': contrasting_mask,
            'cropping': cropping,
            'cropping_mask': cropping_mask,
            'shearing': shearing,
            'shearing_mask': shearing_mask,
            'rotating': rotating,
            'rotating_mask': rotating_mask
        }

    return aug_data


# Проверим функцию.

# In[38]:


torchvision_with_mask_augs = torchvision_augmentations_with_mask(featured_table, data)


# In[39]:


sample = torchvision_with_mask_augs['image_image_01029_20210720_const_0015.jpg']


# In[40]:


fig, axs = plt.subplots(3, 6, figsize=(16, 6))

for ax, (name, value) in zip(axs.ravel(), sample.items()):

    ax.imshow(value)
    ax.set_title(name)

    ax.set_xticks([])
    ax.set_yticks([])


# ### Albumentations аугментации с маской

# In[41]:


def albumentations_augmentations_with_mask(table, data):

    # Сюда будем сохранять все аугментации
    aug_data = {}

    for i, row in tqdm(table.iterrows(), total=2059):
        img_path = row['Файл']
        # Открываем загруженную картинку и маску
        image = data[img_path[:-4]]['image']
        label_field = data[img_path[:-4]]['label']
        # Запоминаем размерность
        h, w = image.size[:2]

        # Переводим в numpy array, чтобы albumentations мог работать с изображениями
        image = np.asarray(image)
        label_field = np.asarray(label_field)

        # Зеркальное отражение по горизонтали
        transform = A.HorizontalFlip(p=1.0)
        mirroring = transform(image=image, mask=label_field)
        mirroring, mirroring_mask = mirroring['image'], mirroring['mask']

        # Поворот
        transform = A.RandomRotate90(p=1.0)
        turning = transform(image=image, mask=label_field)
        turning, turning_mask = turning['image'], turning['mask']

        # Случайное Приближение/Отдаление
        transform = A.Affine(
                scale=(0.3, 2),   # 0.3 = Отдаление, 2 = Приближение
                p=1.0
            )
        zooming = transform(image=image, mask=label_field)
        zooming, zooming_mask = zooming['image'], zooming['mask']

        # Смена яркости
        transform = A.RandomBrightnessContrast(brightness_limit=(-0.5, 0.5), p=1)
        brightning = transform(image=image, mask=label_field)
        brightning, brightning_mask = brightning['image'], brightning['mask']

        # Смена контрастности contrast
        transform = A.RandomBrightnessContrast(contrast_limit=(-0.5, 0.5), p=1)
        contrasting = transform(image=image, mask=label_field)
        contrasting, contrasting_mask = contrasting['image'], contrasting['mask']

        # Случайные обрезки
        transform = A.RandomResizedCrop(
                    size=(h, w),
                    scale=(0.2, 1.0),     # какая часть изображения будет взята
                    p=1.0)
        cropping = transform(image=image, mask=label_field)
        cropping, cropping_mask = cropping['image'], cropping['mask']

        # Случайный наклон
        transform = A.Affine(
                shear=(-60, 60),
                p=1.0
            )
        shearing = transform(image=image, mask=label_field)
        shearing, shearing_mask = shearing['image'], shearing['mask']

        # Случайное вращение
        transform = A.Rotate(limit=(-90, 90), p=1)
        rotating = transform(image=image, mask=label_field)
        rotating, rotating_mask = rotating['image'], rotating['mask']


        aug_data[img_path] = {
            'original': image,
            'original_mask': label_field,
            'mirroring': mirroring,
            'mirroring_mask': mirroring_mask,
            'turning': turning,
            'turning_mask': turning_mask,
            'zooming': zooming,
            'zooming_mask': zooming_mask,
            'brightning': brightning,
            'brightning_mask': brightning_mask,
            'contrasting': contrasting,
            'contrasting_mask': contrasting_mask,
            'cropping': cropping,
            'cropping_mask': cropping_mask,
            'shearing': shearing,
            'shearing_mask': shearing_mask,
            'rotating': rotating,
            'rotating_mask': rotating_mask
        }

    return aug_data


# Проверим функцию.

# In[42]:


albumentations_with_mask_augs = albumentations_augmentations_with_mask(featured_table, data)


# In[43]:


sample = albumentations_with_mask_augs['image_image_01029_20210720_const_0119.jpg']


# In[44]:


fig, axs = plt.subplots(3, 6, figsize=(16, 6))

for ax, (name, value) in zip(axs.ravel(), sample.items()):

    ax.imshow(value)
    ax.set_title(name)

    ax.set_xticks([])
    ax.set_yticks([])


# С аугментацией лучше всего справилась библиотека ```albumentations```. Она выполнила аугментацию качественно и быстро, так как является специлизированным инструментом для такой задачи. Для дальнейшнего обучения модели оставляем аугментированный этой библиотекой набор данных.

# In[45]:


# Удаляем из памяти другие наборы

del pil_augs, cv2_augs, tv_augs, torchvision_with_mask_augs


# ##

# <a id=4></a>
# 
# # 4. Сегментация изображений «без учителя»

# Для сегментации областей рассмотрим 3 варианта алгоритмов. 
# 
# **Теорема Клейнберга (2002)** утверждает, что **не существует** алгоритма кластеризации, который одновременно удовлетворяет **трём естественным требованиям**:
# 1. **Scale-Invariance** — при умножении всех расстояний на константу результат не меняется.
# 
# 2. **Richness** — алгоритм способен выдавать любое возможное разбиение.
# 
# 3. **Consistency** — если внутри кластера расстояния уменьшают, а между кластерами увеличивают, разбиение должно сохраниться.
# 
# **Вывод:** ```Универсального идеального метода кластеризации не существует. Поэтому выбор алгоритма зависит от данных и целей.```
# 
# #### Алгоритмы сегментации (кластеризации):
# 
# 1. **MiniBatch K-means**
# * Классический метод кластеризации общего назначения.
# * Алгоритм, работающий на батчах, идеально подходит для больших изображений и датасетов с аугментациями.
# * Эффективно работает, когда нужно небольшое число кластеров (например, 5).
# 2. **Иерархическая кластеризация**
# * Строит дендрограмму, объединяя пиксели в группы.
# * Не масштабируется под аугментации и большие датасеты.
# * Сложность делает метод непрактичным.
# 3. **SLIC**
# * Специализированный алгоритм для изображений.
# * Слишком детализирует изображение, если требуется разделить сцену всего на 5 областей.
# * Требует подбора гиперпараметров и тонкой настройки.
# 
# 
# Для реализации выберем **MiniBatch K-means** — лучший выбор для сегментации изображений в условиях, когда требуется небольшое число цветовых кластеров, есть большой объём данных (включая аугментации) и важны скорость, стабильность и простота интерпретации результата.

# ####

# <a id=4-1></a>
# ## 4.1. Обучение алгоритма.
# 
# Процесс клстеризации разделен на **2** части:
# - **Обучение алгоритма**
# - **Кластеризация областей**

# ### Обучение алгоритма

# In[46]:


# Инициализация модели
# Указываем кол-во кластеров, инициализацию, максимальное кол-во итераций и размер батча
model = MiniBatchKMeans(
    n_clusters=5,
    init='k-means++',
    max_iter=100,
    batch_size=1024,
    verbose=0
)

# Проходимся по каждому файлу
for im_name in tqdm(albumentations_with_mask_augs):
    # По каждой вариации изображения (оригинал и аугментации)
    for img in albumentations_with_mask_augs[im_name]:

        # Задача обучения "без учителя", поэтому обучаем алгоритм только на изображениях, без масок
        if img.endswith('_mask'):
            continue

        # Убедимся, что все векторы будут одной длины
        resized_img = cv2.resize(albumentations_with_mask_augs[im_name][img], (224, 224))

        # Представляем в формат для обучения
        pixels = resized_img.reshape(-1, 3)

        # Обучение модели
        model.partial_fit(pixels)


# ### Кластеризация областей

# In[47]:


# Сюда будем сохранять результаты кластеризации
clustering_data = {}

# Проходимся по каждому файлу
for im_name in tqdm(albumentations_with_mask_augs):
    # По каждой вариации изображения (оригинал и аугментации)
    for img in albumentations_with_mask_augs[im_name]:

        # Задача обучения "без учителя", поэтому инференс делаем только на изображениях, без масок
        if img.endswith('_mask'):
            continue

        # Убедимся, что все векторы будут одной длины
        resized_img = cv2.resize(albumentations_with_mask_augs[im_name][img], (224, 224))

        # Представляем в формат для инференса
        pixels = resized_img.reshape(-1, 3)

        # Предсказание кластеров после обучения
        labels = model.predict(pixels)

        # Возвращаем обратно 224х224
        kmeans_res = labels.reshape((224, 224)).astype('int8')

        # Сохраняем результат в словарь
        clustering_data.setdefault(im_name, {})[f'{img}_clustering'] = kmeans_res


# <a id=4-2></a>
# ## 4.2. Визуализация кластеризации.

# In[48]:


files2vis = ['image_image_01003_20210720_const_0016.jpg', 'image_image_01029_20210720_const_0153.jpg',
             'image_image_01003_20210720_const_0018.jpg', 'image_image_01029_20210720_const_0049.jpg', 
             'image_image_01029_20210720_const_0017.jpg', ]

for filename in files2vis:
    original_image = albumentations_with_mask_augs[filename]['original']
    clustering = clustering_data[filename]['original_clustering']

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(original_image)
    axs[0].set_title("Original")
    im = axs[1].imshow(clustering)
    axs[1].set_title("Clustering")


    plt.colorbar(im, ax=axs[1], ticks=np.unique(model.labels_))


# <a id=4-3></a>
# ## 4.3. Описание кластеров.
# 
# Исходя из визуализации кластеров можно сделать такой вывод:
# - Кластер 0. Основные поля.
# - Кластер 1. Облака.
# - Кластер 2. Поля с пшеницей.
# - Кластер 3. Темно-зеленная влажная почва.
# - Кластер 4. На текущих изображениях не выявлен. Возможно, это водоемы.

# #####

# <a id=4-4></a>
# ## 4.4. Расчет метрик.
# 
# Выполним расчет индексов ```IoU``` и ```Dice``` для оценки полученного кластеризатора.

# Метрику ```IoU``` импортируем из библиотеки ```segmentation_models_pytorch```. 
# 
# Метрику ```Dice``` получим из формулы ниже:

# ![image.png](attachment:4a986df2-912e-4542-8775-f820bf7240ed.png)

# Для расчета метрик будем оставлять только кластеры под номером ```0``` и ```2```, которые охарактеризовали как поле с пшеницей.

# In[52]:


# Сюда будем сохранять метрики по всем изображениям
cluster_metrics = {'iou': [],
                   'dice': []}

# Проходимся по каждому файлу
for filename in tqdm(clustering_data):
    # Считаем метрики для оригинала и аугментаций вместе
    for img in clustering_data[filename]:
        type_image = img[:-11]

        # Загружаем маску и приведем к одному размеру
        original_mask = albumentations_with_mask_augs[filename][f'{type_image}_mask']
        original_mask = cv2.resize(original_mask, (224, 224))
        # Маска в формате RGB, приведем в grayscale
        original_mask = cv2.cvtColor(original_mask, cv2.COLOR_RGB2GRAY)
        # Избавляемся от артефактов, бинаризуем маску
        original_mask =(original_mask < 155).astype('int8')

        # Загружаем результат кластеризации
        clustering_mask = clustering_data[filename][img]
        # Предобрабатываем маску, удаляя артефакты
        clustering_mask = morphology.closing(morphology.opening(clustering_mask))

        # Определяем, что поля наш целевой признак - они будут иметь значение 1, все остальные кластеры приравниваем к 0
        clustering_mask = np.where(
        np.isin(clustering_mask, [0, 2]),
        1, 0)

        # Переводим маски в тензоры, для работы библиотеки
        original_mask = torch.from_numpy(original_mask)[None, None, ...]
        clustering_mask = torch.from_numpy(clustering_mask)[None, None, ...]
        # Получаем статистики для получения будущих метрик
        tp, fp, fn, tn = metrics.get_stats(clustering_mask, original_mask, mode='binary', threshold=0.5)

        iou = metrics.functional.iou_score(tp, fp, fn, tn)
        dice = 2 * iou / (1 + iou)

        # Сохраняем метрики
        cluster_metrics['iou'].append(iou.item())
        cluster_metrics['dice'].append(dice.item())


# Итоговые индексы ```IoU``` и ```Dice``` для оценки полученного кластеризатора (основной
# набор данных, дополненный изображениями, полученными в результате аугментации) с
# точки зрения определения с/х угодий с использованием масок:

# In[53]:


# Считаем итоговую метрику как среднее по всем изображениям
total_iou = np.mean(cluster_metrics['iou'])
total_dice = np.mean(cluster_metrics['dice'])

print(f'Итоговый IoU: {total_iou:.2f}')
print(f'Итоговый Dice: {total_dice:.2f}')


# ##

# <a id=5></a>
# # 5. Формирование итогового набора расширенных и предобработанных данных.

# В процессе работы над данными мы имеем:
# * Основной набор данных, помечаемый ключом ```original```
# * Аугментированные изображения
# * Разметка для каждого изображения, включая аугментации

# Итоговый набор данных представлен в словаре ```albumentations_with_mask_augs```.

# Сохраним этот набор данных, добавляя разметку для всех изображений.

# In[54]:


# Сюда будет загружать изображения с масками
os.makedirs('preprocess_images/images', exist_ok=True)
os.makedirs('preprocess_images/labels', exist_ok=True)

# Проходимся по каждому изображению
for filename in tqdm(albumentations_with_mask_augs):
    filestem = filename[:-4]
    # Создаем отдельную папку под изображению, сюда будем сгружать оригинальное изображение с аугментациями
    os.makedirs(f'preprocess_images/images/{filestem}', exist_ok=True)
    os.makedirs(f'preprocess_images/labels/{filestem}', exist_ok=True)

    # По каждому типу изображения: оригинал, аугментация или маска
    for image_type in albumentations_with_mask_augs[filename]:

        image = albumentations_with_mask_augs[filename][image_type]
        # Маску загружаем в директорию labels
        if image_type.endswith('_mask'):
            # Новый путь файла
            new_path = f'preprocess_images/labels/{filestem}/{filestem}_{image_type}.jpg'
            cv2.imwrite(new_path, image)

        # Изображения загружаем в директорию images
        else:
            # Новый путь файла
            new_path = f'preprocess_images/images/{filestem}/{filestem}_{image_type}.jpg'
            cv2.imwrite(new_path, image)


# Итоговая структура **выгруженных данных**:
# 
# preprocess_images/<br>
# |--- images/<br>
# |------ image_image_00001_20210420_const_0004/<br>
# |--------- image_image_00001_20210420_const_0004_original.jpg<br>
# |--------- image_image_00001_20210420_const_0004_brightning.jpg<br>
# |--- labels/<br>
# |------ image_image_00001_20210420_const_0004/<br>
# |--------- image_image_00001_20210420_const_0004_original_mask.jpg<br>
# |--------- image_image_00001_20210420_const_0004_brightning_mask.jpg<br>
# 

# Также, сохраним набор данных в архив с расширением ```.zip```

# In[55]:


shutil.make_archive('preprocess_images', 'zip', 'preprocess_images')

