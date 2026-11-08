#!/usr/bin/env python
# coding: utf-8

# # Модуль 3 с геоданными.
# 
# В этом модуле возможна классификация/регрессия для геоданных. Принцип такой же как у табличных данных, за исключением визуализации.
# 
# Для примера возьмем классификацию, для регрессии этот же модуль тоже подойдет.

# # Импорт библиотек

# In[1]:


import geopandas as gpd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt


# In[2]:


import warnings
warnings.simplefilter(action='ignore')


# # Импорт данных

# In[3]:


gdf = gpd.read_file('data/wgs84-etalons.shp')


# In[4]:


X = gdf.select_dtypes('number').drop(['Kf2'], axis=1).iloc[:, 8:13]
y = gdf['Kf2']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train.head(3)


# # Обучение модели и инфренс

# In[5]:


model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# # Визуализация прогноза

# In[6]:


test_gdf = gdf.iloc[X_test.index]
train_gdf = gdf.iloc[X_train.index]
test_gdf['y_pred'] = y_pred

# чтобы на графике были категориальные признаки,
# если визуализация регрессии - удаляем строку
test_gdf["y_pred"] = test_gdf["y_pred"].astype("category")

fig, ax = plt.subplots(figsize=(10, 8))

test_gdf.plot(
    ax=ax,
    column="y_pred",
    edgecolor="black",
    linewidth=0.5,
    alpha=0.8,
    legend=True,
)

train_gdf.plot(
    ax=ax,
    linewidth=0.5,
    edgecolor="black",
    color='white'
)

ax.set_title("Карта предсказаний", fontsize=16)
ax.set_axis_off()
plt.tight_layout()
plt.show()


# ##

# ## Сохранение модели

# In[7]:


import joblib 

joblib.dump(model, 'geo_model.pkl')

