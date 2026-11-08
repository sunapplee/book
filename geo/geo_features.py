#!/usr/bin/env python
# coding: utf-8

# # Занятие от 15.02.2026
# 
# #### Тема занятия - геопространственные признаки. Разбор 1 и 2 модуля DIGITAL SKILLS 2023.

# ## Содержание
# 
#   * [Импорт библиотек](#0)
# * [1. H3 — преобразование координат в категорию](#1)
# * [2. Haversine — расстояние на сфере](#2)
# * [3. Shapely — работа с геометрией](#3)
# * [4. Разбиение данных](#4)

# ####

# <a id=0></a>
# ### Импорт библиотек

# In[9]:


import h3
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances
from sklearn.model_selection import train_test_split
import numpy as np
from shapely.geometry import Point, Polygon


# ####

# <a id=1></a>
# ## 1. H3 — преобразование координат в категорию
# 
# H3 — система **пространственной индексации** от Uber.
# Она позволяет заменить точные координаты на индекс гексагональной ячейки.

# ### Преобразование координат в H3-индекс

# In[10]:


# пример данных
df = pd.DataFrame({
    "lat": [61.1699, 60.1705],
    "lon": [24.9384, 24.9410]
})

# resolution 8–10 обычно используется для городских задач
resolution = 9

df["h3_cell"] = df.apply(
    lambda row: h3.latlng_to_cell(row["lat"], row["lon"], resolution),
    axis=1
)

df.head()


# Теперь вместо непрерывных координат у нас категориальный признак `h3_cell`.
# 
# Можно строить агрегаты:

# In[11]:


# плотность объектов в ячейке
counts = df.groupby("h3_cell").size().rename("cell_density")
df = df.join(counts, on="h3_cell")

df


# Это уже полноценная ML-фича.

# ####

# <a id=2></a>
# ## 2. Haversine — расстояние на сфере

# Обычная евклидова формула для широты и долготы даёт погрешность.
# Используем Haversine — расстояние по поверхности Земли.

# В ```scikit-learn``` есть готовая реализация —
# ```sklearn.metrics.pairwise.haversine_distances```

# In[4]:


city_center = np.array([[60.1699, 24.9384]])

coords_rad = np.radians(df[["lat", "lon"]].values)
center_rad = np.radians(city_center)

dist = haversine_distances(coords_rad, center_rad)

df["distance_to_center_km"] = dist.flatten() * 6371

df


# Получаем числовой признак, который часто сильно улучшает модель.

# ####

# <a id=3></a>
# 
# ## 3. Shapely — работа с геометрией
# 
# 
# Shapely позволяет выполнять пространственную логику: проверять принадлежность, пересечения, считать площадь.

# In[5]:


# создаём полигон (условный район)
polygon = Polygon([
    (24.93, 60.16),
    (24.95, 60.16),
    (24.95, 60.18),
    (24.93, 60.18)
])

# проверка для строки датафрейма
def is_inside(lat, lon):
    point = Point(lon, lat)
    return polygon.contains(point)

df["inside_area"] = df.apply(
    lambda row: is_inside(row["lat"], row["lon"]),
    axis=1
)

df


# Теперь у нас бинарная пространственная фича.

# ### Пример: буфер (радиус вокруг точки)

# In[6]:


point = Point(24.9384, 60.1699)

buffer_zone = point.buffer(0.01)  # условный радиус


# Буферы используются для создания зон доступности.

# ####

# <a id=4></a>
# 
# ## 4. Разбиение данных 

# In[ ]:


X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

