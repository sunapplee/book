#!/usr/bin/env python
# coding: utf-8

# # Импорт библиотек

# In[1]:


import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, HDBSCAN, Birch
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from matplotlib import pyplot as plt
import seaborn as sns


# # Импорт данных

# In[2]:


df = pd.read_csv('data/Air_Quality.csv')

df.head(3)


# Для кластеризации уберем все лишние признаки, оставим только числовые. Их отмасштабируем для лучшей производительности.

# In[3]:


X = df.select_dtypes(include='number')

scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X.sample(3)


# # Оптимальное количество кластеров
# 
# Для определения оптимального количества кластеров, тк не все алгоритмы их определяют, используем метод локтя.
# 
# Для реализации метода берем KMeans, тут нам важно просто выбить критерий, ну и определить кол-во кластеров. Для других моделей (AgglomerativeClustering, BIRCH и тд) используем то же кол-во.
# 
# [Все придумали за нас](https://stackoverflow.com/questions/41540751/sklearn-kmeans-equivalent-of-elbow-method)

# In[4]:


distorsions = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    distorsions.append(kmeans.inertia_)

fig = plt.figure(figsize=(15, 5))
plt.plot(range(2, 11), distorsions)
plt.grid(True)
plt.title('Elbow curve');


# Данные не очень хорошо кластеризуются. Такое поведение вполне возможно на чемпионате. Поэтому важно правильно интерпретировать и аргументировать эти результаты.

# 
# > Исходя из графика, мы видим, что внутрекластерное расстояние постепенно падает, но главное искривление происходит на 5 кластерах.
# 
# n_clusters=5

# # Модели кластеризации
# 
# Эти модели работают безотказно. Советую всегда брать их, когда просят использовать 3 разные модели. Разные архитектуры в конечном счете приведут к разным интерпретациям, что даст больше объективности.
# 
# 
# [Документация](https://scikit-learn.org/stable/modules/clustering.html)
# 
# 
# 

# ## KMeans

# In[5]:


kmeans_df = df.copy()

kmeans = KMeans(n_clusters=5)
kmeans.fit(X)
kmeans_df['label'] = kmeans.predict(X)


# ## HDBSCAN
# 
# Тут количество кластеров указывать не нужно.

# In[6]:


hdbcan_df = df.copy()

hdbcan = HDBSCAN()
hdbcan_df['label'] = hdbcan.fit_predict(X)


# ## BIRCH

# In[7]:


birch_df = df.copy()

birch = Birch(n_clusters=5)
birch.fit(X)
birch_df['label'] = birch.predict(X)


# ## Анализ кластеризации и дисбаланс классов
# 
# Очевидно, что идеальную кластеризацию провести в этими данными тяжело. Некоторые модели могут очень плохо отработать. Поэтому мы кластеризуем тремя моделями.

# In[8]:


kmeans_df['label'].value_counts()


# In[9]:


hdbcan_df['label'].value_counts()


# HDBSCAN не смог найти закономерности в данных и разделить их на кластеры.

# In[10]:


birch_df['label'].value_counts()


# # Визуализация кластеров

# Для визуализации кластеров используем методы понижения размерности - PCA (метод главных компонент)

# In[11]:


pca = PCA(n_components=2)
X_cpa = pca.fit_transform(X)
df_pca = pd.DataFrame(X_cpa, columns=['x', 'y'])
df_pca
df_pca.head(3)


# Функция scatterplot из seaborn может долго работать на большом наборе данных. Для ускорения работы мы можем уменьшить количество данных на графике, это не сильно повлияет на саму визуализацию.

# In[12]:


sample_indx = df_pca.sample(10000).index


# In[13]:


fig, axs = plt.subplots(1, 3, figsize=(16, 4))

sns.scatterplot(df_pca.iloc[sample_indx],
                x='x', y='y',
                hue=kmeans_df.iloc[sample_indx]['label'],
                ax=axs[0])
axs[0].set_title('Визуализация KMeans')


sns.scatterplot(df_pca.iloc[sample_indx],
                x='x', y='y',
                hue=hdbcan_df.iloc[sample_indx]['label'],
                ax=axs[1])
axs[1].set_title('Визуализация HDBSCAN')


sns.scatterplot(df_pca.iloc[sample_indx],
                x='x', y='y',
                hue=birch_df.iloc[sample_indx]['label'],
                ax=axs[2])
axs[2].set_title('Визуализация BIRCH');


# # Метрики качества кластеризации

# ## Silhouette Score
# 
# Чем выше метрика, тем лучше.

# In[14]:


print(f'''
Индекс Silhouette Score
KMeans: {silhouette_score(X, kmeans_df['label']):.2f}
HDBSCAN: {silhouette_score(X, hdbcan_df['label']):.2f}
BIRCH: {silhouette_score(X, birch_df['label']):.2f}
''')


# ## Calinski–Harabasz Index
# 
# 
# Чем больше, тем лучше.

# In[15]:


print(f'''
Индекс Calinski–Harabasz
KMeans: {calinski_harabasz_score(X, kmeans_df['label']):.2f}
HDBSCAN: {calinski_harabasz_score(X, hdbcan_df['label']):.2f}
BIRCH: {calinski_harabasz_score(X, birch_df['label']):.2f}
''')


# ## Davies–Bouldin Index
# 
# Чем меньше, тем лучше

# In[16]:


print(f'''
Индекс Дэвиcа-Болдуина
KMeans: {davies_bouldin_score(X, kmeans_df['label']):.2f}
HDBSCAN: {davies_bouldin_score(X, hdbcan_df['label']):.2f}
BIRCH: {davies_bouldin_score(X, birch_df['label']):.2f}
''')


# Тут возможны неоднозначные ситуации как сейчас.
# 
# На чемпионате важно оставить метки кластеров для дальнейшней работы. HDBSCAN выбил лучшие метрики, однако он предсказал 1 кластер. Это верно, однако для нас лучше визуально распределенные кластеры, как KMeans.
# 
# 
# В таком случае, лучше написать:
# 
# 
# 
# > Исходя из анализа дисбаланса классов, визулизации и оценки метрик, лучшим алгоритмом стал KMeans. Эта модель показала лучшее распределение данных...
# 
# 

# # Обзор кластеров

# Для модели KMeans проведем обзор кластеров. Важно использовать изначальные данные, а не отмасштабированные.

# In[17]:


df[['CO',	'NO2', 'SO2', 'O3', 'PM2.5', 'PM10', 'AQI']].groupby(kmeans_df['label']).mean()


# На основе этих данных можно описать каждый кластер.
