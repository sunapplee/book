#!/usr/bin/env python
# coding: utf-8

# # Анализ признаков и их значимости

# ## Импорт библиотек

# In[3]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from sklearn.inspection import permutation_importance

import shap


# ## Импорт данных

# In[4]:


### Для работы с BIGDATA
for chunk in pd.read_csv("big.csv", chunksize=200_000):
    chunk.to_parquet("data.parquet", engine="pyarrow", append=True)


df = pd.read_csv('data/Air_Quality.csv')

# Добавим случайные признаки, для понимания, какие данные могут быть в датасете.
df["random_feature1"] = np.random.rand(len(df)) * 10
df["random_feature2"] = np.random.rand(len(df)) * 100
df["random_feature3"] = np.random.rand(len(df))

# Для анализа берем только числовые признаки
numeric_features = df.select_dtypes('number').columns

df = df[numeric_features]
df.head(3)


# ## Анализ данных

# ### Построение графиков распределений признаков

# In[5]:


plt.figure(figsize=(15, 10))

for i, col in enumerate(numeric_features, 1):
    plt.subplot(3, 4, i)
    sns.histplot(df[col], kde=True)
    plt.title(col)

plt.tight_layout()
plt.show()


# Видны странные признаки, которые имеют сплошной график распределения.

# ## Оценка важности признаков
# 
# После анализа признаков займемся оценкой важности и фильтрацией признаков.
# 
# Ниже будут разные подходы к оценке признаков. В работе лучше использовать несколько подходов, так результат будет объективнее.

# ### Корреляционный анализ

# - Сильно коррелированные признаки могут приводить к мультиколлинеарности
# 
# - Такие признаки можно удалить или объединить

# In[6]:


corr_matrix = df[numeric_features].corr()

corr_matrix


# In[7]:


plt.figure(figsize=(9, 6))
sns.heatmap(corr_matrix, cmap="coolwarm", annot=False)
plt.title("Корреляционная матрица")
plt.show()


# In[8]:


corr_with_target = corr_matrix["AQI"].sort_values(ascending=False)
corr_with_target


# Для 3 последних признаков корреляция с целевой переменной нулевая.

# ### Дисперсионный анализ (ANOVA)
# 
# Используется для оценки зависимости числовых признаков от целевой переменной.
# 
# ANOVA не применима для анализа непрерывной целевой переменной без категориального фактора. В связи с этим целевая переменная AQI будем дискретизировать на классы, после чего проведем однофакторный дисперсионный анализ для оценки статистической значимости признаков.

# In[9]:


# Сначала дискретизируем AQI в классы

df["AQI_class"] = pd.cut(
    df["AQI"],
    bins=[0, 20, 25, 45, 70, np.inf],
    labels=[0, 1, 2, 3, 4]
)

df['AQI_class'].value_counts()


# In[10]:


anova_results = {}

for col in numeric_features:
  groups = []
  for name, group in df.groupby("AQI_class", observed=False):
      groups.append(group[col].values)  # добавляем массив значений каждого класса

  f_stat, p_value = stats.f_oneway(*groups)
  anova_results[col] = p_value


pd.options.display.float_format = '{:.4f}'.format
anova_results = pd.Series(anova_results).sort_values()
anova_results


# 📌 Интерпретация:
# - p-value < 0.05 — признак статистически значим
# - p-value > 0.05 — влияние не подтверждено
# 
# 
# Исходя из дисперсионного анализа все признаки статистически значимы для целевой переменной (AQI), кроме случайных. Аналогично случайным могут быть реальные мусорные признаки.

# In[11]:


df


# In[12]:


df = df.drop('AQI_class', errors='ignore', axis=1)


# ## Важно!
# 
# Для дальнейшнего анализа мы будем оценивать взаимосвязь признаков по инфреренсу модели, поэтому обучим простую модель RandomForest для регрессии.

# In[13]:


target = "AQI"

# Для ускорения анализа берем лишь часть данных
sample_indx = df.sample(5000).index

X = df.drop(columns=[target]).iloc[sample_indx]
y = df[target].iloc[sample_indx]


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# In[15]:


model = RandomForestRegressor(
    n_jobs=-1
)

model.fit(X_train, y_train)


# ### SHAP-анализ

# In[16]:


explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)


# In[17]:


shap.summary_plot(shap_values, X_test)


# In[18]:


mean_abs_shap = np.abs(shap_values).mean(axis=0)

# Создаём таблицу с признаками
shap_df = pd.DataFrame({
    "Feature": X_test.columns,        # названия признаков
    "MeanAbsSHAP": mean_abs_shap       # среднее абсолютное SHAP
})

# Сортируем по важности
shap_df = shap_df.sort_values(by="MeanAbsSHAP", ascending=False)
shap_df


# По SHAP анализу из графика и числовых значений важности, самые нерелеватные признаки:
# - random_feature1
# - random_feature2
# - random_feature3

# ### Permutation Importance

# In[21]:


perm_importance = permutation_importance(
    model, X_test, y_test, n_repeats=10
)

importances = pd.Series(
    perm_importance.importances_mean,
    index=X.columns
).sort_values(ascending=False)

importances


# In[22]:


plt.figure(figsize=(10, 6))
importances.plot(kind="bar")
plt.title("Permutation Importance")
plt.show()


# По анализу Permutation Importance - случайные признаки также неважны для целевой переменной.

# ## Исключение наименее значимых признаков
# 
# 
# Тут уже исходя из проведенных анализов попросят выкинуть неважные признаки, берем несколько неважных (например, случайных), и выбрасываем из основго датасета.
# 
# Важно, аргументировать почему мы это делаем. Также нужно аргументировать, почему некоторые неважные колонки все-таки нужно оставить: айди, метки и тд.

# In[23]:


df_total = df.drop(['random_feature1', 'random_feature2', 'random_feature3'], axis=1)
df_total.head()


# ## Оптимизация типов данных
# 
#  — это процесс приведения столбцов датафрейма или переменных программы к таким типам данных, которые занимают меньше памяти, работают быстрее и при этом не теряют нужную точность.

# In[24]:


df_total.info()


# memory usage: 4.0 MB
# 
# Это значит что сейчас в памяти датасет занимает такое количество оперативной памяти.
# 
# 
# Попробуем для всех признаков сменить тип на float32. Таким образом, мы не потеряем данных, но обеспечим более умеренный размер данных. 

# In[25]:


df_total = df_total.astype('float32')

df_total.info()


# memory usage: 1.4 MB
# 
# 
# А это значит, что мы оптимизировали датасет в 2 раза.

# ## Сохранение итогового набора данных с максимальным сжатием
# 
# Сохраним набор данных в формате parquet с помощью сжатия zstd. Это один из самых эффективных современных алгоритмов,
# сочетает хорошее сжатие и высокую скорость.

# In[26]:


df_total.to_parquet(
    'data/data.parquet',
    compression='zstd'
)

