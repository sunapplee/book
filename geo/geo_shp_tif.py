#!/usr/bin/env python
# coding: utf-8

# # Работа с геоданными
# 
# Ноутбук для работы с векторными и растровыми геоданными, визуализации и анализа лесных участков.

# ## 1. Импорты библиотек

# In[3]:


import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt


# ## 2. Загрузка данных
# 

# In[4]:


# Загружаем необходимые слои
etalons = gpd.read_file('data/shp/wgs84-etalons.shp')
bounds = gpd.read_file('data/shp/wgs84-forestry-bounds.shp')
shortcuts = gpd.read_file('data/shp/wgs84-etalons-clearcuts.shp')


# ## 3. Базовая визуализация
# 

# In[5]:


# Отображение нескольких слоёв на одном графике
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
etalons.plot(color='blue', ax=ax)
shortcuts.plot(color='red', ax=ax)
plt.title('Etalons')
plt.show()


# ## 4. Анализ данных
# 

# Вывод ячеек разного цвета в зависимости от условий. Для примера
# 

# In[6]:


import pandas as pd

# Просмотр структуры данных
pd.set_option('display.max_columns', None)
etalons.sample(2)


# In[7]:


# Словарь для расшифровки пород
species_dict = {
    'С': 'Сосна обыкновенная',
    'Е': 'Ель европейская',
    'Б': 'Берёза повислая',
    'ОС': 'Осина',
    'Л': 'Лиственница сибирская',
    '000000': 'Не покрытые древесно-кустарниковой растительностью'
}

# Подсчёт распространения пород по признаку Mr1
species_counts = etalons['Mr1'].value_counts()
total_count = len(etalons)

results = []
for code, count in species_counts.items():
    percentage = (count / total_count) * 100
    description = species_dict.get(code, f"Неизвестная порода ({code})")
    results.append({
        'Сокращение': code,
        'Расшифровка': description,
        '% распространения': round(percentage, 2)
    })
pd.DataFrame(results)


# С помощью такого синтаксиса можно визуализировать любой признак.

# In[9]:


# Визуализация признака Mr1 с легендой
etalons.plot(column='Mr1', legend=True)
plt.title('Распространение пород деревьев (Mr1)')
plt.tight_layout()
plt.show()


# ## 5. Интерактивная визуализация

# Отрисуем границы участков на карте с подложкой в виде карт OpenStreetMap.

# In[14]:


# Интерактивная карта с подложкой OpenStreetMap
etalons.explore()


# Отрисуем границы участков на карте с подложкой в виде любых спутниковых карт. Добавим легенду, добавим туда признак H1.

# In[15]:


# Интерактивная карта со спутниковой подложкой и признаком H1
etalons.explore(
    column='H1',
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri Satellite",
    style_kwds={"fill": True, "weight": 1, 'color': 'white'}
)


# ## 6. Работа с растровыми данными
# 

# In[18]:


import rasterio
import os


def load_tif_files():
    """Загружает все TIF файлы и приводит их к единому размеру"""
    directory = 'data/tif/'
    files = os.listdir(directory)

    # Загрузка данных всех файлов
    datasets = []
    shapes = []
    for f in files:
        with rasterio.open(directory + f) as src:
            band = src.read(1)  # Первый канал
            datasets.append(band)
            shapes.append(band.shape)

    # Определение максимального размера для выравнивания
    max_height = max(shape[0] for shape in shapes)
    max_width = max(shape[1] for shape in shapes)

    # Дополнение всех изображений до максимального размера
    padded_list = []
    for dataset, shape in zip(datasets, shapes):
        padded_img = np.zeros((max_height, max_width), dtype=dataset.dtype)
        padded_img[:shape[0], :shape[1]] = dataset
        padded_list.append(padded_img)

    return padded_list


# ## 7. Аугментация данных
# 

# In[20]:


from PIL import Image

def augment(image, rotation_angle_deg=None, flip_code=None, scale_factor=None):
    """Применяет аугментации к изображению: отражение, поворот, масштабирование"""
    pil_image = Image.fromarray(image.astype(np.uint8))

    # Отражение по горизонтали/вертикали
    if flip_code == 0:
        pil_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
    elif flip_code == 1:
        pil_image = pil_image.transpose(Image.FLIP_TOP_BOTTOM)
    elif flip_code == 2:
        pil_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)

    # Поворот
    if rotation_angle_deg is not None:
        pil_image = pil_image.rotate(rotation_angle_deg, expand=False)

    # Масштабирование
    if scale_factor is not None:
        new_size = (int(pil_image.width * scale_factor), int(pil_image.height * scale_factor))
        pil_image = pil_image.resize(new_size, Image.BILINEAR)

    return pil_image


# In[21]:


# Применение аугментации ко всем загруженным изображениям
images = load_tif_files()

for im in images:
    for i in range(5):
        augment(im).save(f'aug_file_{i}.jpeg')


# In[22]:


images


# ## 8. Справочная информация
# 

# Для проведения NDVI и других спектральных анализов необходимо работать с каналами из TIF файлов.

# ### **Каналы Sentinel-2 (актуальные для NDVI, Moisture Index, SWIR):**
# 
# | Канал | Название | Назначение |
# |-------|----------|------------|
# | **B4** | Red (красный) | Используется в NDVI |
# | **B8** | NIR (ближний ИК) | Используется в NDVI, Moisture Index |
# | **B11** | SWIR (коротковолновый ИК) | Используется в Moisture Index, SWIR |
# | **B12** | SWIR (длинноволновый ИК) | Также может использоваться как SWIR |
# | **B6** | В ближнем ИК (NIR), но с меньшим спектральным разрешением | Используется для оценки растительности (менее точно, чем B8) |
# 
# ---
# 
# ### **Расчёт по каналам:**
# 
# #### 1. **NDVI**:
# - Использует **B8 (NIR)** и **B4 (Red)**  
# - Формула: `(B8 - B4) / (B8 + B4)`
# 
# #### 2. **Moisture Index**:
# - Использует **B11 (SWIR)** и **B8 (NIR)**  
# - Формула: `(B11 - B8) / (B11 + B8)`
# 
# #### 3. **SWIR**:
# - Можно использовать **B11** или **B12** как отдельный канал (без деления).  
# - Или как разность/отношение с другими каналами, например:  
#   - `(B11 - B8) / (B11 + B8)` — то же, что Moisture Index  
#   - `B11` — как чистое значение отражения в ИК-диапазоне
