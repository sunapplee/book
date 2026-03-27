#!/usr/bin/env python
# coding: utf-8

# # Анализ текстовых данных и маршрутизация вопросов
# 
# В данном ноутбуке проводится анализ текстовых вопросов из датасета **SberQuAD**.  
# Цель работы — выделить тематические кластеры вопросов и обучить модель для автоматической маршрутизации текста.
# 
# В качестве итоговой модели используется **ruBERT**, дообученный для задачи **многоклассовой классификации вопросов по тематическим кластерам**.

# ## Содержание
# * [1.Загрузка данных](#1)
# * [2. Кластеризация вопросов](#2)
#     * [2.1 Подготовка данных](#2-1)
#     * [2.2 KMeans](#2-2)
#     * [2.3 Hierarchical clustering](#2-3)
#     * [2.4 BERTopic](#2-4)
#     * [2.5 Лучший алгоритм](#2-5)
#     
# * [3. Маршрутизация текста](#3)
# * [4. Анализ частотности слов](#4)
# * [5. Облако слов](#5)

# ###

# ## Импорт библиотек

# In[1]:


import pandas as pd
import numpy as np
from tqdm import tqdm

import ollama

from sklearn.decomposition import PCA
import umap

import string
import regex

from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
from kneed import KneeLocator
from scipy.cluster.hierarchy import linkage, dendrogram
from bertopic import BERTopic
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from pymorphy3 import MorphAnalyzer

import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
import cv2
from wordcloud import WordCloud

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from transformers import get_linear_schedule_with_warmup


# ###

# <a id=1></a>
# # 1. Загрузка данных
# 
# Для анализа данных возьмем набор вопросов из датасета **SberQuAD**.

# In[2]:


questions_list = pd.read_parquet('data/train-00000-of-00001.parquet')['question'].tolist()
print('Количество вопросов:', len(questions_list))
questions_list[:4]


# ###

# <a id=2></a>
# ## 2. Кластеризация вопросов
# 
# Для кластеризации вопросов рассмотрим три алгоритма. 
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
# #### Алгоритмы кластеризации:
# 
# 1. **MiniBatch K-means**
# * Классический метод кластеризации общего назначения, поэтому подойдет для кластеризации текста.
# * Алгоритм, работающий на батчах, идеально подходит для текста.
# * Эффективно работает, когда нужно небольшое число кластеров (например, 5).
# 2. **Иерархическая кластеризация**
# * Не масштабируется большие датасеты.
# * Сложность делает метод непрактичным.
# 3. **BERTopic**
# * Специализированный алгоритм для тематического моделирования текстов.
# * Автоматически выделяет темы (topics) и ключевые слова в документах.
# * Может выделять слишком много мелких тем, если данные неоднородны.
# 
# Для реализации возьмем все 3 алгоритма, проверим метрику 

# ###

# <a id=2-1></a>
# ## 2.1 Подготовка данных
# Для алгоритмов `MiniBatch KMeans` и `Hierarchical clustering` необходимо преобразовать текстовые вопросы в числовые представления (эмбеддинги). Для этого используется модель векторизации текста.
# 
# Данная модель преобразует текст в плотные семантические векторы, которые отражают смысл предложения и позволяют применять алгоритмы кластеризации, основанные на расстояниях между объектами. 
# 
# Модель `Qwen3 embeddings` выбрана благодаря современной архитектуре, компактному размеру и высокой скорости вычисления эмбеддингов.

# ### Векторизация текста

# In[3]:


def get_embed(questions: list):
    # Прогоняем текст через модель
    batch = ollama.embed(
        model='qwen3-embedding:0.6b',
        input=questions,
    )
    # Возвращаем список эмбеддингов
    return batch['embeddings']


# In[4]:


# Ограничиваем выборку
QUESTION_COUNT = 30000


# In[5]:


get_ipython().run_cell_magic('time', '', "embed_list = get_embed(questions_list[:QUESTION_COUNT])\nembed_df = pd.DataFrame(embed_list)\nprint('Размерность датасета:', embed_df.shape)\nembed_df.head()\n")


# Итого получили `1024` признака для каждого вопроса.

# ### Уменьшение размерности
# 
# Эмбеддинги имеют высокую размерность, что может ухудшать работу алгоритмов кластеризации и увеличивать время вычислений. Поэтому перед применением алгоритмов кластеризации выполним снижение размерности.
# 
# Для этого используем следующие методы:
# 
# - **PCA** — линейный метод уменьшения размерности, сохраняющий максимальную дисперсию данных.
# - **UMAP** — нелинейный метод, позволяющий сохранять локальную структуру данных и часто используемый при работе с текстовыми эмбеддингами.
# 
# Снижение размерности позволяет уменьшить шум в данных, ускорить работу алгоритмов кластеризации и улучшить разделимость кластеров.

# #### PCA
# 
# Для `UMAP` оптимально будет снижение до `100` признаков.

# In[6]:


# Инициализируем метод
pca_reducer = PCA(n_components=100)


# In[7]:


# Применяем к нашим данным
pca_embedding = pca_reducer.fit_transform(embed_df)
pca_embedding_df = pd.DataFrame(pca_embedding)
print('Размер после снижения размерности:', pca_embedding_df.shape)
pca_embedding_df.head()


# #### UMAP
# 
# Для `UMAP` оптимально будет снижение до `15` признаков.

# In[8]:


# Инициализируем метод
umap_reducer = umap.UMAP(n_components=15)


# In[9]:


# Применяем к нашим данным
umap_embedding = umap_reducer.fit_transform(embed_df)
umap_embedding_df = pd.DataFrame(umap_embedding)
print('Размер после снижения размерности:', umap_embedding_df.shape)
umap_embedding_df.head()


# #### PCA для визуализации
# 
# Представим данные в **двух размерностях**, чтобы можно было выполнить визуализацию результатов кластеризации.

# In[10]:


# Инициализируем метод
pca_reducer = PCA(n_components=2)


# In[11]:


# Применяем к нашим данным
pca2_embedding = pca_reducer.fit_transform(embed_df)
pca2_embedding_df = pd.DataFrame(pca2_embedding)
print('Размер после снижения размерности:', pca2_embedding_df.shape)
pca2_embedding_df.head()


# После подготовки данных, можем перейти к обучению алгоримтов кластеризации!

# <a id=2-2></a>
# ## 2.2 MiniBatch KMeans

# ### Количество кластеров

# Определим количество кластеров с помощью **метода локтя**, реализиованного в библиотеке `kneed`.

# In[12]:


# Сюда сохраняем метрику
inertia = []
# Количество кластеров
k_values = range(2, 20)

# Проходимся по n кластерам
for k in k_values:
    # Обучаем модель
    model = MiniBatchKMeans(n_clusters=k, random_state=42)
    model.fit(pca_embedding_df)
    # Сохраняем метрику
    inertia.append(model.inertia_)


# In[13]:


# Выделяем лучший кластер
kneedle = KneeLocator(
    k_values,
    inertia,
    curve="convex",
    direction="decreasing"
)
optimal_k = kneedle.elbow
kneedle.plot_knee()
plt.title(f'Оптимальное количество кластеров: {optimal_k}');


# ### Обучение алгоритма

# In[14]:


get_ipython().run_cell_magic('time', '', '# Обучаем модель\nmodel_kmeans = MiniBatchKMeans(n_clusters=optimal_k, random_state=42)\nmodel_kmeans.fit(pca_embedding_df)\nlabels_kmeans = model_kmeans.predict(pca_embedding_df)\n')


# In[15]:


print('Распределение кластеров:')
pd.Series(labels_kmeans).value_counts()


# ### Оценка качества кластеризации
# 
# Для оценки качества кластеризации используются следующие метрики:
# 
# - **Silhouette score** — показывает, насколько объект близок к своему кластеру по сравнению с другими. Значения лежат в диапазоне от -1 до 1. Чем выше значение, тем лучше кластеризация.
# 
# - **Davies–Bouldin Index** — оценивает сходство между кластерами. Чем меньше значение, тем лучше разделены кластеры.
# 
# - **Calinski–Harabasz Index** — измеряет соотношение межкластерной и внутрикластерной дисперсии. Чем выше значение, тем лучше качество кластеризации.

# In[16]:


get_ipython().run_cell_magic('time', '', "print(f'Silhouette score для MiniBatchKMeans: {silhouette_score(pca_embedding_df, labels_kmeans):.3f}\\n')\nprint(f'Davies–Bouldin Index для MiniBatchKMeans: {davies_bouldin_score(pca_embedding_df, labels_kmeans):.3f}\\n')\nprint(f'Calinski–Harabasz Index для MiniBatchKMeans: {calinski_harabasz_score(pca_embedding_df, labels_kmeans):.3f}\\n')\n")


# ### Визуализация кластеров

# In[17]:


plt.title('Визуализация кластеризации для MiniBatchKMeans')
sns.scatterplot(data=pca2_embedding_df, x=0, y=1, hue=labels_kmeans, palette="Spectral");


# <a id=2-3></a>
# ## 2.3 Hierarchical clustering

# ### Количество кластеров

# Определим количество кластеров с помощью **дендрограммы**.

# In[18]:


# Построим дерево объединения кластеров
Z = linkage(umap_embedding_df, method='ward')

plt.figure(figsize=(10,5))
dendrogram(Z)
plt.show()


# Число кластеров было выбрано на основе анализа дендрограммы. Наибольший разрыв между уровнями объединения наблюдается в средней части дерева, что указывает на наличие `5` устойчивых кластеров.

# ### Обучение алгоритма

# In[19]:


get_ipython().run_cell_magic('time', '', '# Обучаем модель\nmodel_aggl = AgglomerativeClustering(n_clusters=5)\nmodel_aggl.fit(umap_embedding_df)\nlabels_aggl = model_aggl.labels_\n')


# In[20]:


print('Распределение кластеров:')
pd.Series(labels_aggl).value_counts()


# ### Оценка качества кластеризации
# 
# Для оценки качества кластеризации используются следующие метрики:
# 
# - **Silhouette score** — показывает, насколько объект близок к своему кластеру по сравнению с другими. Значения лежат в диапазоне от -1 до 1. Чем выше значение, тем лучше кластеризация.
# 
# - **Davies–Bouldin Index** — оценивает сходство между кластерами. Чем меньше значение, тем лучше разделены кластеры.
# 
# - **Calinski–Harabasz Index** — измеряет соотношение межкластерной и внутрикластерной дисперсии. Чем выше значение, тем лучше качество кластеризации.

# In[21]:


get_ipython().run_cell_magic('time', '', "print(f'Silhouette score для AgglomerativeClustering: {silhouette_score(umap_embedding_df, labels_aggl):.3f}\\n')\nprint(f'Davies–Bouldin Index для AgglomerativeClustering: {davies_bouldin_score(umap_embedding_df, labels_aggl):.3f}\\n')\nprint(f'Calinski–Harabasz Index для AgglomerativeClustering: {calinski_harabasz_score(umap_embedding_df, labels_aggl):.3f}\\n')\n")


# ### Визуализация кластеров

# In[22]:


plt.title('Визуализация кластеризации для AgglomerativeClustering')
sns.scatterplot(data=pca2_embedding_df, x=0, y=1, hue=labels_aggl, palette="Spectral");


# <a id=2-4></a>
# ## 2.4 BERTopic

# ### Обучение алгоритма

# In[23]:


# Обучаем алгоритм
bertopic_model = BERTopic(language="russian",)
topics, probs = bertopic_model.fit_transform(questions_list[:QUESTION_COUNT])
print('Количество кластеров:', len(set(topics)))


# Рассмотрим темы:

# In[24]:


topic_info = bertopic_model.get_topic_info()
topic_info[['Topic', 'Name', 'Count']]


# ### Оценка качества кластеризации
# 
# Для оценки качества кластеризации используются следующие метрики:
# 
# - **Silhouette score** — показывает, насколько объект близок к своему кластеру по сравнению с другими. Значения лежат в диапазоне от -1 до 1. Чем выше значение, тем лучше кластеризация.
# 
# - **Davies–Bouldin Index** — оценивает сходство между кластерами. Чем меньше значение, тем лучше разделены кластеры.
# 
# - **Calinski–Harabasz Index** — измеряет соотношение межкластерной и внутрикластерной дисперсии. Чем выше значение, тем лучше качество кластеризации.

# In[25]:


get_ipython().run_cell_magic('time', '', "print(f'Silhouette score для Bertopic: {silhouette_score(umap_embedding_df, topics):.3f}\\n')\nprint(f'Davies–Bouldin Index для Bertopic: {davies_bouldin_score(umap_embedding_df, topics):.3f}\\n')\nprint(f'Calinski–Harabasz Index для Bertopic: {calinski_harabasz_score(umap_embedding_df, topics):.3f}\\n')\n")


# ### Визуализация кластеров

# In[26]:


plt.title('Визуализация кластеризации для Bertopic')
sns.scatterplot(data=pca2_embedding_df, x=0, y=1, hue=topics, palette="Spectral");


# ### Визуализация тем

# In[27]:


bertopic_model.visualize_barchart()


# <a id=2-5></a>
# ## 2.5 Лучший алгоритм
# 
# Итак, лучшую метрику показал алгоритм `Иерархической кластеризации` на **5** кластерах. Также на визуализации кластеры получились наиболее отчетливыми.
# 
# Однако, мы имеет только метки кластера. Получим названия кластеров (тем), с помощью `TF-IDF`.

# In[28]:


# DataFrame
aggl_df = pd.DataFrame({
    "question": questions_list[:QUESTION_COUNT],
    "cluster": labels_aggl
})

# Будем удалять стоп-слова из топа
russian_stopwords = stopwords.words("russian")
russian_stopwords += ['каких', 'каком', 'какие', 'какую', 'каким']

# Инициализируем TF-IDF
vectorizer = TfidfVectorizer(stop_words=russian_stopwords)
X = vectorizer.fit_transform(aggl_df["question"])
# Получаем слова
terms = vectorizer.get_feature_names_out()

# Получаем топ слов
cluster_names = {}
for cluster in sorted(aggl_df.cluster.unique()):
    idx = (aggl_df.cluster == cluster).values
    cluster_tfidf = X[idx].mean(axis=0)
    top_words = np.array(terms)[np.argsort(cluster_tfidf.A1)[-5:]]
    cluster_names[cluster] = ", ".join(top_words)

cluster_names


# Получаем осмысленное описание тем наших вопросов!

# ###

# <a id=3></a>
# # 3. Маршрутизация текста
# 
# 

# ## Подготовка данных
# 
# Перед тем, как обучать модель маршрутизации, предобработаем наши вопросы.

# ### Анализ дубликатов

# In[29]:


print('Дубликатов:', aggl_df['question'].duplicated().sum())


# ### Смесь латиницы и кириллицы 
# Проведем анализ сэмплов, красный символ означает латиницу.

# In[30]:


texts = aggl_df.sample(10)['question'].tolist()
res = ''
for text in texts:
    res = ''
    for char in text:
        if char in string.ascii_lowercase + string.ascii_uppercase:
            res += f"\x1b[0;30;41m{char}\x1b[0m"
        else:
            res += char
    print(res + '\n')


# Как видно из вывода, вопросы на русском написаны кириллицей полностью, а английские слова если и есть, то латиницей. **Предобработки не требуется.**

# ### Эмодзи, символы и спецсимволы
# 
# В тексте могут встречаться **эмодзи**. Зачастую они не несут никакой смысловой нагрузки, а являются только *фактором привлечения внимания*. Для корректной работы модели в будущем *избавимся от них*.
# 
# **Спецсимволы** встречаются редко, однако тоже не несут смысла и будут вредно сказываться на качестве данных.
# 
# **Пунктуация** в задаче маршрутизации также будет только во вред.
# 
# Оставляем только буквы, цифры, пробелы. `Пунктуацию`, ```Спецсимволы```, ```Unicode-символы``` и ```эмодзи``` будем удалять, что явялется стандартной практикой.

# In[31]:


def keep_letters_numbers(text):
    # Оставляем только буквы, цифры, пробелы
    return regex.sub(r"[^\p{L}\p{N} ]", "", text)

for text in texts[:5]:
    print(keep_letters_numbers(text) + '\n')


# Применим преобразование на весь датасет.

# In[32]:


aggl_df['question'] = aggl_df['question'].apply(keep_letters_numbers)


# In[33]:


aggl_df.sample(4)


# ### Удаление стоп-слов

# In[34]:


# Переводим вопросы в нижний регистр
aggl_df['question'] = aggl_df['question'].apply(lambda x: x.lower())
aggl_df.sample(3)


# In[35]:


# Функция удаления стоп-слов
stop_words = set(stopwords.words('russian'))

def remove_stopwords(text):
    tokens = word_tokenize(text)
    return ' '.join([word for word in tokens if word not in stop_words])


# Удалим стоп-слова во всем датасете.

# In[36]:


aggl_df['question'] = aggl_df['question'].apply(remove_stopwords)
aggl_df.sample(3)


# ### Лемматизация 
# Для приведения слов к нормальной форме используем библиотеку `pymorphy3`, это поможет уменьшить количество различных форм одного и того же слова и улучшить результаты текстового анализа.

# In[37]:


# Реалиазция лемматизации
morph = MorphAnalyzer()

def lemmatize(text):
    tokens = text.split()
    return ' '.join([morph.parse(word)[0].normal_form for word in tokens])


# Лемматизируем весь набор вопросов.

# In[38]:


get_ipython().run_cell_magic('time', '', "aggl_df['question'] = aggl_df['question'].apply(lemmatize)\naggl_df.sample(3)\n")


# ### Разделение выборки
# 
# В качестве обучающей и тестовой выборок используем **5000 вопросов**, заранее проаннотированных с помощью кластеризации.
# Разделим данные на три части: **70% / 15% / 15%**.
# 
# * **70%** — обучающая выборка. Используется непосредственно для дообучения модели и изменения её весов.
# * **15%** — валидационная выборка. Примеры из этого набора не участвуют в обучении, но используются для контроля процесса: по валидационному *loss* и метрикам сохраняется лучшая модель.
# * **15%** — тестовая выборка. Полностью откладывается до конца эксперимента и используется только для финальной, объективной оценки качества модели на данных, которые она никогда не видела.
# 
# Такое разбиение позволяет одновременно эффективно обучить модель, своевременно отслеживать переобучение и провести честную итоговую проверку качества.

# In[39]:


# 1) Train = 70%, Temp = 30%
train_df, temp_df = train_test_split(
    aggl_df,
    test_size=0.30,
    stratify=aggl_df["cluster"],
    random_state=42
)

# 2) Temp делим пополам: 15% вал, 15% тест
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,   # 50% от 30% → 15% от всего набора
    stratify=temp_df["cluster"],
    random_state=42
)

print('Размеры выборок:')
train_df.shape, val_df.shape, test_df.shape


# ### Создание класса `Dataset`

# In[40]:


# Инициализация токенизатора
tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")


# In[41]:


class RouterDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)                   # количество строк

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # вопрос и таргет
        question = row["question"]
        label = row['cluster']

        # токенизация пары текстов
        encoded = self.tokenizer(
            question,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(),             # тензор токенов
            "attention_mask": encoded["attention_mask"].squeeze(),   # маска
            "token_type_ids": encoded["token_type_ids"].squeeze(),   # тип сегмента
            "label": torch.tensor(label)                             # таргет
        }


# ### Создание `Dataloader`

# In[42]:


# создаём PyTorch Dataset
train_dataset = RouterDataset(train_df, tokenizer, max_len=64)
val_dataset = RouterDataset(val_df, tokenizer, max_len=64)
test_dataset = RouterDataset(test_df, tokenizer, max_len=64)

# создаём DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,  pin_memory=True)
val_loader = DataLoader(val_dataset,   batch_size=16, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset,  batch_size=16, shuffle=False, pin_memory=True)


# ## Инициализация модели

# In[43]:


# Модель: ruBERT как энкодер + линейная голова для предсказания кластера
class BertForMatching(nn.Module):
    def __init__(self, model_name="DeepPavlov/rubert-base-cased"):
        super().__init__()

        # загружаем RuBERT (все слои, без классификатора)
        self.bert = AutoModel.from_pretrained(model_name)

        # Linear-слой: 768 → кол-во кластеров (прогноз темы)
        self.fc = nn.Linear(self.bert.config.hidden_size, len(cluster_names))

    def forward(self, input_ids, attention_mask, token_type_ids):
        # прогоняем BERT на входных токенах
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # [CLS] токен — обобщённое представление пары текстов
        cls = output.last_hidden_state[:, 0]

        # логит
        logits = self.fc(cls).squeeze(-1)

        return logits


# ## Обучение модели

# Объявляем функцию для обучения.

# In[44]:


# Основной цикл обучения
def train(model, loader, epochs=3, lr=2e-5):
    model = model.cuda()  # переносим модель на GPU
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # лосс для классификации
    criterion = nn.CrossEntropyLoss()

    # количество шагов для scheduler
    total_steps = len(loader) * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    best_val_loss = float("inf")   # для сохранения лучшей модели
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for batch in tqdm(loader):
            # перенос батча на GPU
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            token_type_ids = batch["token_type_ids"].cuda()
            labels = batch["label"].cuda()
            optimizer.zero_grad()

            # предсказания модели
            logits = model(input_ids, attention_mask, token_type_ids)

            # считаем ошибку
            loss = criterion(logits, labels)
            loss.backward()
            # шаг обновления весов
            optimizer.step()
            # шаг scheduler — меняем learning rate
            scheduler.step()
            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_loader)
        print(f"\nEpoch {epoch+1}: Train loss = {train_loss:.4f}")

        # Валидация
        val_loss, acc, f1 = evaluate(model, val_loader, criterion)
        print(f"Validation: Loss={val_loss:.4f} | Acc={acc:.4f} | F1={f1:.4f}")

        # Сохраняем лучшую модель
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.bin")
            print("✓ Лучшая модель сохранена → best_model.bin")

    return model


# Напишем функцию для оценки качества модели (для валидации).

# In[45]:


def evaluate(model, loader, criterion):

    model.eval()
    total_loss = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:

            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            token_type_ids = batch["token_type_ids"].cuda()
            labels = batch["label"].cuda()

            logits = model(input_ids, attention_mask, token_type_ids)

            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = total_loss / len(loader)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    return val_loss, acc, f1


# ### Запуск обучения

# In[46]:


model = BertForMatching()
model = train(model, train_loader, epochs=10)


# ## Инференс
# 
# Посмотрим на результаты модели.

# In[47]:


### Загрузка лучшей модели
best_model = BertForMatching()
best_model.load_state_dict(torch.load('best_model.bin', weights_only=True))
best_model.eval()
best_model = best_model.cuda()


# In[48]:


# Выбираем случайный вопрос
for idx in np.random.randint(1, len(test_df), size=5):
    print(f"ВОПРОС: {test_df.iloc[idx]['question']}")

    # Инференс модели
    data = test_dataset[idx]
    input_ids = data["input_ids"].unsqueeze(0).cuda()
    attention_mask = data["attention_mask"].unsqueeze(0).cuda()
    token_type_ids = data["token_type_ids"].unsqueeze(0).cuda()
    labels = data["label"].unsqueeze(0).cuda()

    # Прогон модели
    logits = best_model(input_ids, attention_mask, token_type_ids)
    # Получаем ответ
    preds = torch.argmax(logits, dim=1)

    print(f'КАТЕГОРИЯ: {cluster_names[preds.item()]}')
    print()


# ## Оценка качества модели

# In[49]:


# лосс для классификации
criterion = nn.CrossEntropyLoss()
# Считаем метрики
metrics = evaluate(best_model, test_loader, criterion)

print(f'Лосс итоговой модели: {metrics[0]:.3f}')
print(f'Accuracy итоговой модели: {metrics[1]:.3f}')
print(f'F1 итоговой модели: {metrics[2]:.3f}')


# Отличные метрики для задачи множественной классификации!

# ##

# <a id=4></a>
# # 4. Анализ частотности слов

# In[50]:


# инициализиуем счетчик
word_counter = Counter()

# Проходися по каждому вопросу
for i, row in aggl_df.iterrows():
    words = row["question"].lower().split()
    word_counter.update(words)

# Топ 10
result = word_counter.most_common(10)
pd.DataFrame(result, columns=['Слово', 'Количество в вопросах'])


# Из анализа понятно, что самые частые слова - **вопросообразующие слова**. Например: *"какой", "сколько", "кто"*.

# ###

# <a id=5></a>
# # 5. Облако слов

# In[51]:


# Подготавливаем изображение для формы облака
img = cv2.imread('rosatom.jpg', cv2.IMREAD_GRAYSCALE)
img = np.where(img > 155, 255, 0)


# In[52]:


# Сохраняем все вопросы в 1 строчке
all_questions = ' '.join([row['question'] for i, row in aggl_df.iterrows()])

# Отрисовываем облако слов
plt.figure(figsize=(10, 100))
wc = WordCloud(background_color='white', mask=img,
               width=800, height=800,
               colormap='Blues_r').generate(all_questions)
plt.imshow(wc)
plt.axis("off");


# ###

# # Вывод
# 
# В ходе работы был проведён анализ текстовых вопросов из датасета **SberQuAD**.  
# Текст был преобразован в семантические эмбеддинги, после чего применены методы **кластеризации** для выделения тематических групп вопросов.
# 
# Были протестированы алгоритмы **MiniBatch K-Means**, **Hierarchical Clustering** и **BERTopic**.  
# Наилучшие результаты показала **иерархическая кластеризация**, позволившая выделить **5 устойчивых кластеров**.
# 
# Полученные кластерные метки использовались для обучения модели **ruBERT**, предназначенной для **маршрутизации вопросов по темам**.
# 
# На тестовой выборке модель показала:
# 
# - **Accuracy:** 0.817  
# - **F1-score:** 0.810  
# 
# Дополнительно был проведён **анализ частотности слов** и построено **облако слов** для визуализации наиболее распространённых терминов.
# 
# Таким образом, построенный пайплайн позволяет автоматически **кластеризовать и классифицировать текстовые вопросы**, что может быть использовано в системах обработки пользовательских запросов.
