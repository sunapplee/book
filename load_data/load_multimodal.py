#!/usr/bin/env python
# coding: utf-8

# # Задание по лекции от 12.02.2026
# 
# Задание состоит за **2 частей**:
# - **RSA:** Работа со "сломанным" ```csv``` файлом и его последующий анализ
# - **AS2024:** Работа с изображениями и аудио разных форматов
# 
# ***Важно:*** ноутбук должен иметь содержание, а также выводы по каждой главе и графикам.

# ## Содержание
# * [Импорт библиотек](#0)
# * [Часть 1. RSA](#1)
#   * [1.1 Загрузка данных](#1-1)
#   * [1.2 Использование памяти](#1-2)
#   * [1.3 Оптимизация типов данных](#1-3)
#   * [1.4 Процент оптимизации](#1-4)
#   * [1.5 Разведочный анализ данных](#1-5)
#   * [1.6 Визуализация датасета](#1-6)
# * [Часть 2. AS2024](#2)
#   * [2.1 Загрузка изображений](#2-1)
#   * [2.2 Загрузка битого csv-файла](#2-2)
#   * [2.3 Расширение набора данных](#2-3)
#   * [2.4 Аннотация новых изображений](#2-4)
#   * [2.5 Загрузка npz-файла](#2-5)
#   * [2.6 Конвертация npz-файла](#2-6)
#   * [2.7 Чтение аудиофайлов](#2-7)
#   * [2.8 Визуализация звуковых волн](#2-8)

# #

# <a id='0'></a>
# 
# ## Импорт библиотек

# In[1]:


from pathlib import Path
import os
import pandas as pd
import numpy as np
import csv

from sklearn.impute import SimpleImputer
from matplotlib import pyplot as plt

from tqdm import tqdm

from PIL import Image
import cv2
import skimage
import torchvision

import librosa
import soundfile as sf
import torchaudio
import sounddevice as sd

import plotly.express as px


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# ####

# <a id='1'></a>
# 
# # Часть 1. RSA

# <a id='1-1'></a>
# 
# ## 1.1 Загрузка данных

# Данные представлены в виде 4 файлов в директории ```rsa```. Формат данных имеет кастомный вид, поэтому существующими методами автоматически его распарсить **невозможно**. Напишем скрипт, который парсит данные **построчно**.

# In[3]:


data_paths = Path('rsa')

all_data = {}

for path in data_paths.iterdir():
    file_data = []
    with open(path, encoding='cp1251') as f:
        lines = f.readlines()
        for line in lines:

            # Строка с названиями колонок
            if 'SignalsArray' in line:
                # Парсим строку
                columns = line.split('=')[1].strip().split(';')
                # Убеждаемся, что в список не попали пустые строки
                columns = [col for col in columns if col]
                # В списке представлены названия сигналов, однака
                # также в данных пристутсвует время записи, добавим еще одну колонку
                columns.insert(0, 'Time')

            # Строки с отдельными записями, это наши данные
            elif 'RsaData' in line:
                # Парсим строку
                split_lines = line.split('=')[1].strip().split(';')

                # В итоге получаем строку формата "2 785.7847 0",
                # Где 2 - индекс, 785.7847 - значение, 0 - константа
                # Для создания набора данных, нам нужно только значение
                data = [float(value.split()[1]) for value in split_lines if len(value.split()) == 3]

                # Также у нас есть 2 значения, которое не подчиняется этому правилу:
                # Время записи, и первая строка значений формата "2 1 789.0552 0"
                # Тоже добавим их в список
                data.insert(0, split_lines[0])
                data.insert(1, float(split_lines[1].split()[2]))

                # Добавим запись в единый список
                file_data.append(data)

        # Проверяем, что количество колонок совпадает с количеством значений в одной записи
        # Это поможет избежать ошибок
        if len(columns) == len(file_data[0]):
            print(f'Всего колонок в файлe {path}:', len(columns))
            # Создаем pd.DataFrame на основе данных из файла
            df = pd.DataFrame(file_data, columns=columns)
            all_data[str(path)] = df


# ### Обзор данных

# In[4]:


print(f'Размерность файла rsa\\N.rsa: ', all_data['rsa\\N.rsa'].shape)
all_data['rsa\\N.rsa'].head(3)


# In[5]:


print(f'Размерность файла rsa\\PTK-Z.rsa: ', all_data['rsa\\PTK-Z.rsa'].shape)
all_data['rsa\\PTK-Z.rsa'].head(3)


# In[6]:


print(f'Размерность файла rsa\\table2.rsa: ', all_data['rsa\\table2.rsa'].shape)
all_data['rsa\\table2.rsa'].head(3)


# In[7]:


print(f'Размерность файла rsa\\t_loop.rsa: ', all_data['rsa\\t_loop.rsa'].shape)
all_data['rsa\\t_loop.rsa'].head(3)


# Исходя из вывода видно, что файлы представляют собой **показания датчиков в определенный момент времени**. Все четыре таблицы имеют одинаковое количество записей, а также одни и те же временные интервалы. На основании этой информации можем объединить четыре таблицы в одну, тем самым получим **показания всех датчиков** в определенный момент времени. 

# In[8]:


# Выполняем слияние всех таблиц по ключу 'Time'
df_merged = pd.merge(all_data['rsa\\N.rsa'], all_data['rsa\\PTK-Z.rsa'], on='Time').merge(all_data['rsa\\table2.rsa'], on='Time').merge(all_data['rsa\\t_loop.rsa'], on='Time')


# В итоге получаем единую таблицу размерностью ```(1800, 192)```.

# In[9]:


df_merged.head(3)


# <a id='1-2'></a>
# 
# 
# ## 1.2 Использование памяти

# In[10]:


df_merged.info()


# Из описания видно, что 191 колонка имеет тип данных ```float64```, а одна (**"Time"**) - строковый тип данных.
# 
# При этой структуре данных, датасет занимает ```2.6 MB``` в памяти.

# <a id='1-3'></a>
# 
# 
# ## 1.3 Оптимизация типов данных

# ```float64``` используется в вычислениях, где важна точность. Например научные, финансовые, сложные математические расчеты.
# 
# ```float32``` используется в ситуациях, когда нужен баланс между скоростью, занимаемой памятью и точностью. Его используют в машинном обучении, компьютерной графике и тд.
# 
# Для нашего анализа данных, с учетом оптимизации датасета, поменяем тип данных на ```float32```.

# In[11]:


df_optimizied = df_merged.astype('float32', errors='ignore')

df_optimizied.info()


# Не сильно потеряв в точности, у нас получилось сократить использование памяти до ```1.3 MB```!

# In[12]:


df_optimizied.head(3)


# <a id='1-4'></a>
# 
# ## 1.4 Процент оптимизации

# In[13]:


print('Процент оптимизации после замены Float64 на Float32 составил: ', (2.6 - 1.3) / 2.6 * 100, '%')


# **Экономия памяти = 50%**

# <a id='1-5'></a>
# 
# 
# ## 1.5 Разведочный анализ данных

# Проанализируем датасет, выявим:
# - Мусорные признаки: константные или околоконстантные столбцы
# - Выбросы
# - Пропуски: столбцы с более чем 90% пропущенных значений

# ### Константные признаки

# In[14]:


# Если доля одного значения больше 95%, то считаем такую колонку константной
top_freq = df_optimizied.apply(
    lambda s: s.value_counts(normalize=True, dropna=False).iloc[0]
)

# Отсекаем константные признаки
df_optimizied = df_optimizied.loc[:, (top_freq < 0.95).tolist()]
df_optimizied.head(3)


# После удаления констатных признаков, у нас осталось 169 колонок.

# ### Выбросы

# <span style="color: red;">Как правильно проанализировать выбросы, без построения boxplot?</span>
# Через перцентили?
# Межквартильный размах?

# Из-за шумов, сбоев калибровки, скачков питания или передачи сигнала **возможны аномальные значения**, не отражающие реальное состояние системы. Точную причину каждого экстремального значения определить невозможно, однако их наличие может **негативно влиять** на статистический анализ и обучение моделей.

# In[15]:


numeric_cols = df_optimizied.select_dtypes('number')

# Определяем 25 и 75 квартиль
Q1 = numeric_cols.quantile(0.25)
Q3 = numeric_cols.quantile(0.75)
# Считаем IQR (Межквартильный размах)
IQR = Q3 - Q1

# Строим границы выбросов (покроет 99% данных)
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Маска выбросов
mask = (numeric_cols < lower_bound) | (numeric_cols > upper_bound)

# Заменяем выбросы на NaN
df_optimizied[numeric_cols.columns] = numeric_cols.mask(mask)


# ### Пропуски

# In[16]:


# Обзор пропусков
df_optimizied.isna().sum()


# Как видно из вывода, в данных есть пропуски, заменим их на **медианное значение** по признаку.

# In[17]:


# Создаем imputer
numeric_cols = df_optimizied.select_dtypes('number')

# Предобрабатываем данные
imputer = SimpleImputer(strategy='median')
numeric_cols = pd.DataFrame(imputer.fit_transform(numeric_cols), columns=numeric_cols.columns)

# Применяем изменения к основному датасету
df_optimizied.loc[:, numeric_cols.columns] = numeric_cols


# In[18]:


print('После предобработки осталось пропусков:', df_optimizied.isna().sum().sum())


# <a id='1-6'></a>
# 
# 
# ## 1.6 Визуализация датасета
# 
# Для первичного анализа данных используем **три типа графиков**:
# 
# - **Lineplot** — позволяет оценить изменение показателей во времени, выявить тренды, сезонность и резкие аномальные скачки.
# 
# - **Boxplot** — наглядно отображает медиану, разброс данных и помогает обнаружить потенциальные выбросы на основе межквартильного размаха.
# 
# - **Histplot** — показывает форму распределения признака, позволяя определить асимметрию, плотность значений и характер распределения данных.
# 

# ***Примечание: ввиду упрощения читаемости, для отработки навыков визуализации возьму 15 первых признаков. Иначе построение 3х169 графиков заняло бы значительную часть ноутбука.***

# In[19]:


df2vis = df_optimizied.iloc[:, :15]
col2vis = df2vis.select_dtypes('number').columns


# In[20]:


# Преобразуем текстовую колонку с временем в формат datetime
df2vis['Time'] = pd.to_datetime(df2vis['Time'])


# ### Изменение признаков во времени (Lineplot)

# In[21]:


fig, axs = plt.subplots(3, 5, figsize=(20, 10))

for feature, ax in zip(col2vis, axs.ravel()):
    ax.plot(df2vis['Time'], df2vis[feature])
    ax.set_title(feature)

fig.autofmt_xdate()


# Часть сигналов (в диапазоне 785–805) ведёт себя **стабильно** с плавными технологическими колебаниями. 
# 
# Другая группа (3150–3200) показывает более **выраженные циклы**.
# 
# А остальные сигналы имеют **резкие провалы и скачки**, что может указывать на нестабильность процесса или особенности измерения.

# ### Обнаружение выбросов (Boxplot)

# In[22]:


fig, axs = plt.subplots(3, 5, figsize=(20, 10))

for feature, ax in zip(col2vis, axs.ravel()):
    ax.boxplot(df2vis[feature])
    ax.set_title(feature)

fig.autofmt_xdate()


# Большая часть признаков (в т.ч. после нашей предобработки), не имеет выбросов. Однако признаки ```10LAB20FU001XQ01```, ```10LAB30FU001XQ01```, ```10LAB40FU001XQ01``` имеют большое количество значений, выходящих за пределы межквартильного размаха (IQR), что указывает на наличие частых выбросов или повышенной вариативности сигнала.

# ### Распредление признаков (Histplot)

# In[23]:


fig, axs = plt.subplots(3, 5, figsize=(20, 10))

for feature, ax in zip(col2vis, axs.ravel()):
    ax.hist(df2vis[feature], bins=100)
    ax.set_title(feature)

fig.autofmt_xdate()


# Исходя из графиков видно, что некоторые признаки имеют форму распределения, близкую к нормальному. Для формальной проверки можно было бы применить тест *Колмогорова–Смирнова*, однако это выходит за рамки текущей задачи. Остальные же признаки имеют выраженную асимметрию, поэтому их распределение существенно отклоняется от нормального. Что интересно, тяжёлые хвосты или множественные выбросы не обнаружены.

# #

# <a id='2'></a>
# 
# 
# # Часть 2. AS2024

# <a id='2-1'></a>
# 
# 
# ## 2.1 Загрузка изображений

# В директории ```image-audio/image``` находятся изображения.
# 
# Чтобы их загрузить, реализуем функции загрузки изображений. Используем различные библиотеки, такие как ```PIL```, ```cv2```, ```torchvision```, ```skimage```, а также загрузим изображения в байтах.

# In[24]:


img_paths = list(Path('image-audio/images').glob("*.png"))


# ### PIL

# In[25]:


# Функция загрузки изображений
def load_image_PIL(img_paths) -> dict:
    images = {}
    # Проходимся по каждому изображению
    for path in tqdm(img_paths, total=237):
        # Загружаем изображение с помощью PIL
        img = Image.open(path)
        images[str(path)] = img

    # Возвращаем словарь с изображеняими
    return images


# In[26]:


# Запуск функции
pill_images = load_image_PIL(img_paths)


# In[27]:


# Пример загруженного изображения
pill_images['image-audio\\images\\100.png'].resize((200, 200))


# ### cv2

# In[28]:


# Функция загрузки изображений
def load_image_cv2(img_paths) -> dict:
    images = {}
    # Проходимся по каждому изображению
    for path in tqdm(img_paths, total=237):
        # Загружаем изображение с помощью cv2
        img = cv2.imread(path)
        images[str(path)] = img

    # Возвращаем словарь с изображеняими
    return images


# In[29]:


# Запуск функции
cv2_images = load_image_cv2(img_paths)


# In[30]:


# Пример загруженного изображения
plt.imshow(cv2_images['image-audio\\images\\15.png'])
plt.axis('off');


# ### torchvision

# In[31]:


# Функция загрузки изображений
def load_image_torchvision(img_paths) -> dict:
    images = {}
    # Проходимся по каждому изображению
    for path in tqdm(img_paths, total=237):
        # Загружаем изображение с помощью torchvision
        img = torchvision.io.read_image(path)
        images[str(path)] = img

    # Возвращаем словарь с изображеняими
    return images


# In[32]:


# Запуск функции
tv_images = load_image_torchvision(img_paths)


# In[33]:


# Пример загруженного изображения
image_np = tv_images['image-audio\\images\\22.png'].cpu().permute(1, 2, 0).numpy()
plt.imshow(image_np)
plt.axis('off');


# ### skimage

# In[34]:


# Функция загрузки изображений
def load_image_skimage(img_paths) -> dict:
    images = {}
    # Проходимся по каждому изображению
    for path in tqdm(img_paths, total=237):
        # Загружаем изображение с помощью skimage
        img = skimage.io.imread(path)
        images[str(path)] = img

    # Возвращаем словарь с изображеняими
    return images


# In[35]:


# Запуск функции
skimage_images = load_image_skimage(img_paths)


# In[36]:


# Пример загруженного изображения
plt.imshow(skimage_images['image-audio\\images\\29.png'])
plt.axis('off');


# ### bytes

# In[37]:


def load_image_bytes(img_paths) -> dict:
    images = {}
    # Проходимся по каждому изображению
    for path in tqdm(img_paths, total=len(img_paths)):
        # Загружаем изображение в байтовом виде
        with open(path, 'rb') as f:
            img_bytes = f.read()

        # Загружаем изображение с помощью np
        img_array = np.frombuffer(img_bytes, np.uint8)
        images[str(path)] = img_array

    # Возвращаем словарь с изображеняими
    return images


# In[38]:


# Запуск функции
bytes_images = load_image_bytes(img_paths)


# In[39]:


# Пример загруженного изображения
plt.imshow(skimage_images['image-audio\\images\\78.png'])
plt.axis('off');


# Все функции справились со своей задачей, однако функция ```load_image_bytes``` загрузила изображения в разы быстрее!

# <a id='2-2'></a>
# 
# ## 2.2 Загрузка csv-файла

# Метаданные по изображениям представлены в файле table.csv.

# In[40]:


table_path = 'image-audio/table.csv'


# In[41]:


data = pd.read_csv(table_path)
data.head(3)


# Как видно из вывода, в таблице много прощенных значений, но нас интересует столбец ```"дефекты"```. Каждая запись представлена в виде строки, распарсим строку и получим **словарь**, с которым будем работать дальше.

# In[42]:


# Создадим функцию парсинга
def parse_deffects(x):
    # Проверяем, что работаем со строкой
    if type(x) == str and x != '0':
        # Базовая обработка оишбок
        try:
            # Процесс парсинга данных
            deffects = x.split('\n')
            deffects_names = [d.split(':')[0][:-2].strip() for d in deffects]
            deffects_pos = [eval(d.split(':')[1].replace('-', ',')) for d in deffects]
            # Возвращаем словарь в формате {дефект: позиция}
            return dict(zip(deffects_names, deffects_pos))
        except Exception as e:
            return {}
    else:
        return {}


# In[43]:


# Применяем ко всему датасету
data['parsed_deffects'] = data['дефекты'].map(parse_deffects)
data.head(3)


# Посмотрим, какие дефекты удалось найти.

# In[44]:


unique_defects = set()

for d in data['parsed_deffects']:
    if isinstance(d, dict):
        unique_defects.update(d.keys())

print(f'Уникальных дефектов всего: {len(unique_defects)}')
print('Дефекты:', unique_defects)


# <a id='2-3'></a>
# 
# ## 2.3 Расширение набора данных

# В результате парсинга таблицы получили ```13``` типов дефектов:
# 
# - Ассиметрия углового шва
# - Кратер
# - Непровар
# - Подрез
# - Скопление включений
# - Трещина поперечная
# - Прожог
# - Поры
# - Трещина
# - Шлаковые включения
# - Брызги
# - Наплыв
# - Трещина сварного соединения. Трещина
# 
# Для каждого типа дефекта было подобрано по ```5``` дополнительных изображений. Данное дополнение датасета поможет достичь лучших результатов при анализе данных и повысить качество последующих предсказаний

# <a id='2-4'></a>
# 
# ## 2.4 Аннотация новых изображений

# Изображения из дополнительного датасета были аннотированы.
# 
# Итоговый датасет представляет собой структуру, где каждый класс дефекта находится в отдельной папке, содержащей соответствующие изображения:
# 
# ```
# dataset/
# ├── Ассиметрия_углового_шва/
# ├── Кратер/
# ├── Непровар/
# ├── Подрез/
# ├── Скопление_включений/
# ├── Трещина_поперечная/
# ├── Прожог/
# ├── Поры/
# ├── Трещина/
# ├── Шлаковые_включения/
# ├── Брызги/
# ├── Наплыв/
# └── Трещина_сварного_соединения__Трещина/
# ```
# 
# Такая организация позволяет эффективно использовать данные для задач классификации или детекции дефектов с помощью моделей машинного обучения.

# <a id='2-5'></a>
# 
# ## 2.5 Загрузка npz-файлов

# В директории ```audio``` находтся файлы формата ```npz``` - это формат архива ```NumPy```.

# In[45]:


# Получаем пути файлов
npz_files = Path('image-audio/audio').glob('*.npz')


# In[46]:


audio_arrays = {}
# Проходимся по каждому файлу, достаем оттуда аудиоданные
for npz in npz_files:
    loaded = np.load(npz)['arr_0']
    audio_arrays[str(npz)] = loaded


# Пример файла:

# In[47]:


audio_arrays['image-audio\\audio\\Образец1 -б1.npz']


# <a id='2-6'></a>
# 
# ## 2.6 Конвертация npz-файла

# Сделаем конвертацию загруженного npz-файла в доступные форматы: ```wav```, ```mp3```, ```m4a```

# In[48]:


# Создаем директории для сохранения
os.makedirs("image-audio/audio/wav_files", exist_ok=True)
os.makedirs("image-audio/audio/mp3_files", exist_ok=True)
os.makedirs("image-audio/audio/m4a_files", exist_ok=True)

# Проходимся в цикле по каждому файлу
for name, arr in tqdm(audio_arrays.items()):
    base_name = Path(name).stem
    # Определяем пути сохранения
    new_path_wav = f"image-audio/audio/wav_files/{base_name}.wav"
    new_path_mp3 = f"image-audio/audio/mp3_files/{base_name}.mp3"
    new_path_m4a = f"image-audio/audio/m4a_files/{base_name}.m4a"

    # Сохраняем как wav-файл
    sf.write(new_path_wav, arr, samplerate=22050)

    # Сохраняем еще форматы из только что созданого wav-файла:
    # Сохраняем mp3 в соответствующую папку
    os.system(f'ffmpeg -i {new_path_wav} {new_path_mp3}')

    # Сохраняем m4a в соответствующую папку
    os.system(f'ffmpeg -i {new_path_wav} -c:a aac -b:a 192k {new_path_m4a}')


# <a id='2-7'></a>
# 
# ## 2.7 Чтение аудиофайлов

# Загрузим аудиофайлы с помощью следующих библиотек:
# - **librosa**
# - **torchaudio**
# - **soundfile**
# - **sounddevice**

# In[49]:


# Сохраняем пути файлов
wav_files = Path('image-audio/audio/wav_files/')
mp3_files = Path('image-audio/audio/mp3_files/')
m4a_files = Path('image-audio/audio/m4a_files/')


# Сортируем файлы по имени для корректного соответствия
wav_files = sorted(wav_files.iterdir())
mp3_files = sorted(mp3_files.iterdir())
m4a_files = sorted(m4a_files.iterdir())


# #####

# ### librosa

# Реализуем функцию ```get_audio_librosa``` для загрузки аудиофайлов с помощью соответсвующей библиотеки.

# In[50]:


def get_audio_librosa(wav_files, mp3_files, m4a_files):
    audios = {}
    # Проходимся по каждому файлу трех форматов (wav, mp3, m4a)
    for wav_file, mp3_file, m4a_file in tqdm(
                                            zip(
                                             wav_files,
                                             mp3_files,
                                             m4a_files
                                                ), total=30
                                            ):
        # Загружаем каждый файл
        wav, sr = librosa.load(wav_file)
        mp3, sr = librosa.load(mp3_file)
        m4a, sr = librosa.load(m4a_file)
        # Сохраняем аудио в словарь
        audios[wav_file.stem] = {
            'wav': wav,
            'mp3': mp3,
            'm4a': m4a
        }

    # Возвращаем готовый список
    return audios


# Запустим функцию и проверим ее работу.

# In[51]:


librosa_audios = get_audio_librosa(wav_files, mp3_files, m4a_files)

librosa_audios['Образец1 -б1']


# #####

# ### torchaudio

# Реализуем функцию ```get_audio_torchaudio``` для загрузки аудиофайлов с помощью соответсвующей библиотеки.

# In[52]:


def get_audio_torchaudio(wav_files, mp3_files, m4a_files):
    audios = {}
    # Проходимся по каждому файлу трех форматов (wav, mp3, m4a)
    for wav_file, mp3_file, m4a_file in tqdm(
                                            zip(
                                             wav_files,
                                             mp3_files,
                                             m4a_files
                                                ), total=30
                                            ):
        # Загружаем каждый файл
        wav, sr = torchaudio.load(wav_file)
        mp3, sr = torchaudio.load(mp3_file)
        m4a, sr = torchaudio.load(m4a_file)
        # Сохраняем аудио в словарь
        audios[wav_file.stem] = {
            'wav': wav,
            'mp3': mp3,
            'm4a': m4a
        }

    # Возвращаем готовый список
    return audios


# Запустим функцию и проверим ее работу.

# In[53]:


torchaudio_audios = get_audio_torchaudio(wav_files, mp3_files, m4a_files)

torchaudio_audios['Образец1 -б1']


# #####

# ### soundfile

# Реализуем функцию ```get_audio_soundfile``` для загрузки аудиофайлов с помощью соответсвующей библиотеки.

# In[54]:


def get_audio_soundfile(wav_files, mp3_files, m4a_files):
    audios = {}
    # Проходимся по каждому файлу трех форматов (wav, mp3, m4a)
    for wav_file, mp3_file, m4a_file in tqdm(
                                            zip(
                                             wav_files,
                                             mp3_files,
                                             m4a_files
                                                ), total=30
                                            ):
        # Загружаем каждый файл
        wav, sr = sf.read(wav_file)
        mp3, sr = sf.read(mp3_file)
        # soundfile не поддерживает m4a формат аудио
        # m4a, sr = sf.read(m4a_file)
        # Сохраняем аудио в словарь
        audios[wav_file.stem] = {
            'wav': wav,
            'mp3': mp3,
        }

    # Возвращаем готовый список
    return audios


# Запустим функцию и проверим ее работу.

# In[55]:


soundfile_audios = get_audio_soundfile(wav_files, mp3_files, m4a_files)

soundfile_audios['Образец1 -б1']


# #####

# ### sounddevice

# Слушаем аудио с помощью библиотеки ```sounddevice```.

# In[56]:


sd.play(soundfile_audios['Образец1 -б1']['wav'], 22050)


# <a id='2-8'></a>
# 
# ## 2.8 Визуализация звуковых волн

# Сделаем визуализацию звуковой волны аудиосигнала с помощью ```librosa```.

# In[57]:


plt.title('Визуализация звуковой волны аудиосигнала с помощью librosa')
librosa.display.waveshow(librosa_audios['Образец1-б2']['wav']);


# #####

# Сделаем визуализацию звуковой волны аудиосигнала с помощью ```librosa```.

# In[58]:


# Получаем данные
audio_data = librosa_audios['Образец1-б2']['wav']
sr = librosa_audios['Образец1-б2'].get('sr', 22050)

# Создаем временную ось
time = np.arange(len(audio_data)) / sr

# Строим график
fig = px.line(x=time, y=audio_data,
              title='Визуализация звуковой волны аудиосигнала с помощью Plotly',
              labels={'x': 'Время (секунды)', 'y': 'Амплитуда'})

fig.show()

