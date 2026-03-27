#!/usr/bin/env python
# coding: utf-8

# # Matching объявлений — полный NLP-пайплайн
# Ноутбук реализует полный пайплайн задачи **matching объявлений**: включает загрузку и первичный анализ данных, предобработку текстов (исправление смешения латиницы и кириллицы, удаление emoji, спецсимволов, стоп-слов и стемминг), вычисление метрик схожести (символьных, множественных, префиксных и эмбеддинговых на базе **LaBSE**, **RuBERT-tiny2**, **FRIDA**, **USER**), векторизацию текстов с помощью **CountVectorizer** и **TF-IDF**, а также дообучение модели **RuBERT** для бинарной классификации пар текстов и получение итоговых метрик качества.
# 

# # Содержание
# 
# * [Импорт библиотек](#0)
# * [1. Загрузка данных](#1)
# * [2. Предобработка данных](#2)
# * [3. Метрики схожести](#3)
# * [4. Косинусная схожесть](#4)
# * [5. Классические методы векторизации текста](#5)
#     * [5.1 CountVectorizer](#5-1)
#     * [5.2 TF-IDF](#5-2)
# * [6. Дообучение BERT](#6)

# ###

# <a id=0></a>
# ## Импорт библиотек

# In[ ]:


import pandas as pd
import numpy as np
import regex as msno

import string
import regex as re
import matplotlib.pyplot as plt

import emoji

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer

from joblib import Parallel, delayed

from pymorphy3 import MorphAnalyzer

import textdistance

from sentence_transformers import SentenceTransformer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from transformers import get_linear_schedule_with_warmup

from tqdm import tqdm
import gc

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score


# ##

# <a id=1></a>
# # 1. Загрузка данных

# In[2]:


df = pd.read_parquet('train_part_0004.snappy.parquet')
print('Размер датасета:', df.shape)
df.head(3)


# #

# <a id=2></a>
# # 2. Предобработка данных

# ### Анализ пропусков

# In[3]:


msno.matrix(df, color=(0., .3, 0.), figsize=(15, 6))


# Из графика видно, что в основном пропусков в колонках нет, за исключением ```param2```. Скорее всего, в основном, люди ограничиваются указанием **параметра 1**.

# ### Анализ выбросов

# In[4]:


print('Дубликатов:', df.duplicated().sum())


# ### Смесь латиницы и кириллицы

# Проведем анализ одного сэмпла, красный символ означает латиницу.

# In[5]:


text = df['base_description'].iloc[0]
res = ''
for char in text:
    if char in string.ascii_lowercase:
        res += f"\x1b[0;30;41m{char}\x1b[0m"
    else:
        res += char
print(res)


# Из текста видно, что происходит смешение символов, что может сказаться плохо на итоговом качестве данных. Заменим латинские символы на кириллические.

# In[6]:


# Словарь из символов
latin_to_cyr = {
    "a": "а", "A": "А",
    "c": "с", "C": "С",
    "e": "е", "E": "Е",
    "o": "о", "O": "О",
    "p": "р", "P": "Р",
    "x": "х", "X": "Х",
    "y": "у", "Y": "У",
    "k": "к", "K": "К",
    "b": "в", "B": "В",
    "m": "м", "M": "М",
    "t": "т", "T": "Т",
}


# In[7]:


# Чтобы ошибочно не изменить реальные слова на латинице, создадим простой классификатор
def is_english_word(word):
    cond = all("a" <= ch.lower() <= "z" for ch in word)
    return cond and len(word) > 1


# In[8]:


# Функция для номрмализации
def normalize_words(sentence):
    new_sentence = []
    words = sentence.split()
    for word in words:
        # если английское слово — не трогаем
        if is_english_word(word):
            new_sentence.append(word)
        # иначе: считаем русским → нормализуем гомоглифы
        new_sentence.append(''.join(latin_to_cyr.get(ch, ch) for ch in word))
    return ' '.join(new_sentence)


# Проверим алгоритм на том же описании.

# In[9]:


normalize_text = normalize_words(text)

res = ''
for char in normalize_text:
    if char in string.ascii_lowercase:
        res += f"\x1b[0;30;41m{char}\x1b[0m"
    else:
        res += char
print(res)


# Как видим, алгоритм справился отлично! Предобработаем весь датасет.

# Сравним скорость работы `joblib.Parallel` с созданием отдельного экземпляра Parallel для всех колонок, и единого.

# In[10]:


get_ipython().run_cell_magic('time', '', "\n# Создаем отдельный экземпляра Parallel для всех колонок\n\ndf.loc[:, 'base_title'] = Parallel(n_jobs=-1)(\n    delayed(normalize_words)(text) for text in df['base_title']\n)\ndf.loc[:, 'cand_title'] = Parallel(n_jobs=-1)(\n    delayed(normalize_words)(text) for text in df['cand_title']\n)\ndf.loc[:, 'base_description'] = Parallel(n_jobs=-1)(\n    delayed(normalize_words)(text) for text in df['base_description']\n)\ndf.loc[:, 'cand_description'] = Parallel(n_jobs=-1)(\n    delayed(normalize_words)(text) for text in df['cand_description']\n)\n")


# In[11]:


get_ipython().run_cell_magic('time', '', "\n# Колонки для предобработки\ncols = ['base_title', 'cand_title', 'base_description', 'cand_description']\n\n# Используем один экземпляр Parallel для всех колонок\nwith Parallel(n_jobs=-1) as parallel:\n    for col in cols:\n        df[col] = parallel(\n        delayed(normalize_words)(text) for text in df[col]\n    )\n")


# 2 вариант отработал потчи в **2 раза быстрее**!

# In[12]:


text = df['base_description'].iloc[1811]
res = ''
for char in text:
    if char in string.ascii_lowercase:
        res += f"\x1b[0;30;41m{char}\x1b[0m"
    else:
        res += char
print(res)


# ### Эмодзи и спецсимволы
# 
# 
# В тексте часто встречаются **эмодзи**. Зачастую они не несут никакой смысловой нагрузки, а являются только *фактором привлечения внимания*. Для корректной работы модели в будущем *избавимся от них*.
# 
# **Спецсимволы** встречаются редко, однако тоже не несут смысла и будут вредно сказываться на качестве данных.
# 
# **Пунктуация** в задаче мэтчинга также будет только во вред.
# 
# Оставляем только буквы, цифры, пробелы. `Пунктуацию`, ```Спецсимволы```, ```Unicode-символы``` и ```эмодзи``` будем удалять, что явялется стандартной практикой.

# In[13]:


df['base_description'].iloc[111]


# In[14]:


def keep_letters_numbers(text):
    # Оставляем только буквы, цифры, пробелы
    return re.sub(r"[^\p{L}\p{N}\s]", "", text)

keep_letters_numbers(df['base_description'].iloc[111])


# Функция работает! Распространим на весь датасет

# In[15]:


get_ipython().run_cell_magic('time', '', "df.loc[:, 'base_title'] = df['base_title'].apply(keep_letters_numbers)\ndf.loc[:, 'cand_title'] = df['cand_title'].apply(keep_letters_numbers)\ndf.loc[:, 'base_description'] = df['base_description'].apply(keep_letters_numbers)\ndf.loc[:, 'cand_description'] = df['cand_description'].apply(keep_letters_numbers)\n")


# In[16]:


df['base_description'].iloc[111]


# ### Удаление стоп-слов

# In[17]:


# Приводим в нижний регистр
df.loc[:, 'base_title'] = df['base_title'].str.lower()
df.loc[:, 'cand_title'] = df['cand_title'].str.lower()
df.loc[:, 'base_description'] = df['base_description'].str.lower()
df.loc[:, 'cand_description'] = df['cand_description'].str.lower()


# In[18]:


stop_words = set(stopwords.words('russian'))
def remove_stopwords(text):
    tokens = word_tokenize(text)
    return ' '.join([word for word in tokens if word not in stop_words])


# In[19]:


get_ipython().run_cell_magic('time', '', "# Будет обрабатывать каждую колонку\ncols = ['base_title', 'cand_title', 'base_description', 'cand_description']\n\nwith Parallel(n_jobs=-1) as parallel:\n    for col in cols:\n        df[col] = parallel(\n            delayed(remove_stopwords)(text) for text in df[col]\n        )\n")


# In[20]:


df.sample(3)


# ### Cтемминг и лемматизация
# 
# Для лемматизации могли бы использовать `pymorphy3`, однако из-за большого количества данных этот подход оказывается слишком медленным. Стемминг работает значительно быстрее, так как не требует словарного разбора, а лишь обрезает окончания по правилам. Поэтому будем использовать `SnowballStemmer`, который поддерживает русский язык и хорошо справляется с задачей нормализации текста.

# In[21]:


# Реалиазция лемматизации
morph = MorphAnalyzer()

def lemmatize(text):
    tokens = text.split()
    return ' '.join([morph.parse(word)[0].normal_form for word in tokens])


# In[22]:


# Реализация стемминга
stemmer = SnowballStemmer("russian")

def stem(text):
    tokens = text.split()
    return ' '.join([stemmer.stem(word) for word in tokens])


# In[23]:


get_ipython().run_cell_magic('time', '', "# Будет обрабатывать каждую колонку\ncols = ['base_title', 'cand_title', 'base_description', 'cand_description']\n\nwith Parallel(n_jobs=-1, batch_size=32) as parallel:\n    for col in cols:\n        df.loc[:200, col] = parallel(\n            delayed(stem)(text) for text in df.loc[:200, col]\n        )\n")


# ### Сбор датасета

# Итоговый датасет имеет такой вид.

# In[24]:


X = df[['base_title', 'cand_title', 'base_description', 'cand_description', 'is_double']]
X.to_parquet('data_preprocessed.parquet', index=False)
X.sample(4)


# #

# <a id=3></a>
# # 3. Метрики схожести
# 
# Рассмотрим метрики схожести строк. Для их вычисления будем использовать библиотеку `textdistance`.

# In[25]:


X_metrics = X.sample(1_000)


# <a id=3-1></a>
# ## 3.1 Посимвольные

# ### Hamming 

# In[26]:


get_ipython().run_cell_magic('time', '', "X_metrics.loc[:, 'hamming'] = Parallel(n_jobs=-1, batch_size=32)(\n    delayed(textdistance.hamming)(row['base_description'], row['cand_description'])\n    for _, row in X_metrics.iterrows()\n)\nX_metrics.sample(3)\n")


# ### Jaro-Winkler

# In[27]:


get_ipython().run_cell_magic('time', '', "X_metrics.loc[:, 'jaro_winkler'] = Parallel(n_jobs=-1, batch_size=32)(\n    delayed(textdistance.jaro_winkler)(row['base_description'], row['cand_description'])\n    for _, row in X_metrics.iterrows()\n)\nX_metrics.sample(3)\n")


# ### Levenshtein

# In[28]:


get_ipython().run_cell_magic('time', '', "X_metrics.loc[:, 'levenshtein'] = Parallel(n_jobs=-1, batch_size=32)(\n    delayed(textdistance.levenshtein)(row['base_description'], row['cand_description'])\n    for _, row in X_metrics.iterrows()\n)\nX_metrics.sample(3)\n")


# <a id=3-2></a>
# ## 3.2 Множественные 

# ### Jaccard Similarity

# In[29]:


get_ipython().run_cell_magic('time', '', "X_metrics.loc[:, 'jaccard'] = Parallel(n_jobs=-1, batch_size=32)(\n    delayed(textdistance.jaccard)(row['base_description'], row['cand_description'])\n    for _, row in X_metrics.iterrows()\n)\nX_metrics.sample(3)\n")


# ### Tanimoto Distance

# In[30]:


get_ipython().run_cell_magic('time', '', "X_metrics.loc[:, 'tanimoto'] = Parallel(n_jobs=-1, batch_size=32)(\n    delayed(textdistance.tanimoto)(row['base_description'], row['cand_description'])\n    for _, row in X_metrics.iterrows()\n)\nX_metrics.sample(3)\n")


# <a id=3-3></a>
# ## 3.3 Префиксные/суффиксные

# ### Prefix similarity

# In[32]:


get_ipython().run_cell_magic('time', '', "X_metrics.loc[:, 'prefix'] = Parallel(n_jobs=-1, batch_size=32)(\n    delayed(textdistance.prefix.normalized_similarity)(row['base_description'], row['cand_description'])\n    for _, row in X_metrics.iterrows()\n)\nX_metrics.sample(3)\n")


# ### Postfix similarity

# In[33]:


get_ipython().run_cell_magic('time', '', "X_metrics.loc[:, 'postfix'] = Parallel(n_jobs=-1, batch_size=32)(\n    delayed(textdistance.postfix.normalized_similarity)(row['base_description'], row['cand_description'])\n    for _, row in X_metrics.iterrows()\n)\nX_metrics.sample(3)\n")


# ### 

# <a id=4></a>
# # 4. Косинусная схожесть
# 
# Чтобы оценить, насколько два описания похожи по смыслу, мы вычисляем **косинусную близость их эмбеддингов**. Значение ближе к **1.0** означает более высокую семантическую схожесть.
# 
# Мы используем 5 моделей: **LaBSE**, **RuBERT-tiny2**, **FRIDA** и **USER**.
# Каждая из них кодирует текст по-разному, поэтому сравнение даёт более объективную картину: если большинство моделей дают высокую близость — тексты действительно похожи; если оценки сильно отличаются — модели улавливают разные смысловые признаки.

# In[34]:


# Функция подсчета метрики
def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ### LaBSE

# Загружаем модель.

# In[35]:


model_labse = SentenceTransformer("sentence-transformers/LaBSE")


# In[36]:


model_labse.device


# Функция для векторизации и подсчета метрики.

# In[37]:


def cousine_labse(row):
    t1 = model_labse.encode(row['base_description'])
    t2 = model_labse.encode(row['cand_description'])

    return cosine(t1, t2)


# Применяем ко всему датасету.

# In[38]:


get_ipython().run_cell_magic('time', '', "X_metrics.loc[:, 'cousine_labse'] = X_metrics.apply(cousine_labse, axis=1)\nX_metrics.sample(3)\n")


# ### cointegrated/rubert-tiny2

# Загружаем модель.

# In[39]:


model_rubert = SentenceTransformer("cointegrated/rubert-tiny2")


# In[40]:


model_rubert.device


# Функция для векторизации и подсчета метрики.

# In[41]:


def cousine_rubert(row):
    t1 = model_rubert.encode(row['base_description'])
    t2 = model_rubert.encode(row['cand_description'])

    return cosine(t1, t2)


# Применяем ко всему датасету.

# In[42]:


get_ipython().run_cell_magic('time', '', "X_metrics.loc[:, 'cousine_rubert'] = X_metrics.apply(cousine_rubert, axis=1)\nX_metrics.sample(3)\n")


# ### FRIDA

# Загружаем модель.

# In[43]:


model_frida = SentenceTransformer("ai-forever/FRIDA")


# In[44]:


model_frida.device


# Функция для векторизации и подсчета метрики.

# In[45]:


def cousine_frida(row):
    t1 = model_frida.encode(row['base_description'])
    t2 = model_frida.encode(row['cand_description'])

    return cosine(t1, t2)


# Применяем ко всему датасету.

# In[46]:


get_ipython().run_cell_magic('time', '', "X_metrics.loc[:, 'cousine_frida'] = X_metrics.apply(cousine_frida, axis=1)\nX_metrics.sample(3)\n")


# ### USER

# Загружаем модель.

# In[47]:


model_user = SentenceTransformer("deepvk/USER-base")


# In[48]:


model_user.device


# Функция для векторизации и подсчета метрики.

# In[49]:


def cousine_user(row):
    t1 = model_user.encode(row['base_description'])
    t2 = model_user.encode(row['cand_description'])

    return cosine(t1, t2)


# Применяем ко всему датасету.

# In[50]:


get_ipython().run_cell_magic('time', '', "X_metrics.loc[:, 'cousine_user'] = X_metrics.apply(cousine_user, axis=1)\nX_metrics.sample(3)\n")


# ####

# 
# Таким образом, мы получили **новые признаки**, отражающие **семантическую схожесть описаний**, рассчитанную с помощью **четырёх различных моделей эмбеддингов**. Эти признаки фиксируют, насколько близки тексты по смыслу с точки зрения разных архитектур, что повышает информативность итогового набора данных и позволяет модели лучше учитывать скрытые смысловые связи между описаниями.
# 

# ##

# <a id=5></a>
# # 5. Классические методы векторизации текста.

# Рассмотрим два **классических метода векторизации текста**, основанных на модели *Bag-of-Words*:
# 
# 1. **CountVectorizer** — преобразует текст в вектор частот слов. Каждое измерение соответствует слову из словаря, а значение — количеству его появлений в документе.
# 
# 2. **TF-IDF Vectorizer** — улучшенная версия CountVectorizer: учитывает не только частоту слова в документе, но и его редкость в корпусе. Таким образом, TF-IDF повышает вес информативных слов и снижает вес часто встречающихся.
# 
# Эти методы дают разреженные векторы большой размерности и отражают статистику слов, но не учитывают порядок и семантику текста.

# На примере колонок `base_title` и `cand_title` разберем работу методов.

# <a id=5-1></a>
# ## 5.1 CountVectorizer

# In[51]:


vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X_metrics['base_title'])

countvectorizer_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

print('Размер датасета:', countvectorizer_df.shape)
countvectorizer_df.sample(3)


# <a id=5-2></a>
# ## 5.2 TF-IDF

# In[52]:


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X_metrics['cand_title'])

tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

print('Размер датасета:', tfidf_df.shape)
tfidf_df.sample(3)


# ## 

# <a id=6></a>
# ## 6. Дообучение BERT 
# 
# Для решения задачи **matching объявлений** используется модель **BERT**, которую мы дообучаем на парах текстов вида *(base, candidate)*. Каждая пара размечена бинарным признаком **`is_double`**, отражающим факт дублирования объявления:
# 
# * `1` — объявления относятся к одному и тому же объекту,
# * `0` — объявления различны.
# 
# Для демонстрации процесса и снижения вычислительной нагрузки мы обучаем модель **только на текстовых названиях объявлений** (`base_title`, `cand_title`). Это упрощённый, но наглядный сценарий, позволяющий увидеть, как BERT учится различать дубли на основе лексического сходства.
# 
# В качестве обучающей и тестовой выборок используем **25 000 пар объявлений**, заранее отобранных в **сбалансированном соотношении 50/50** между дублями и недублями. Это обеспечивает корректное обучение модели и предотвращает смещение в сторону преобладающего класса.
# 
# Разделим данные на три части: **70% / 15% / 15%**.
# 
# * **70%** — обучающая выборка. Используется непосредственно для дообучения модели и изменения её весов.
# * **15%** — валидационная выборка. Примеры из этого набора не участвуют в обучении, но используются для контроля процесса: по валидационному *loss* и метрикам сохраняется лучшая модель.
# * **15%** — тестовая выборка. Полностью откладывается до конца эксперимента и используется только для финальной, объективной оценки качества модели на данных, которые она никогда не видела.
# 
# Такое разбиение позволяет одновременно эффективно обучить модель, своевременно отслеживать переобучение и провести честную итоговую проверку качества.
# 

# In[53]:


# загружаем данные
X = pd.read_parquet('data_preprocessed.parquet')
X.head(2)


# In[54]:


# 5k записей для каждого класса
false_data = X[X['is_double'] == 0][['base_title', 'cand_title', 'is_double']].sample(12500)
true_data = X[X['is_double'] == 1][['base_title', 'cand_title', 'is_double']].sample(12500)

data = pd.concat([true_data, false_data]).reset_index(drop=True)


# In[55]:


# 1) Train = 70%, Temp = 30%
train_df, temp_df = train_test_split(
    data,
    test_size=0.30,
    stratify=data["is_double"],
    random_state=42
)

# 2) Temp делим пополам: 15% вал, 15% тест
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,   # 50% от 30% → 15% от всего набора
    stratify=temp_df["is_double"],
    random_state=42
)

train_df.shape, val_df.shape, test_df.shape


# #### Создаем `Dataset`

# In[56]:


class MatchDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)                   # количество строк

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # два текста и таргет
        text1 = row["base_title"]
        text2 = row["cand_title"]
        label = float(row["is_double"])

        # токенизация пары текстов
        encoded = self.tokenizer(
            text1,
            text2,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(),             # тензор токенов
            "attention_mask": encoded["attention_mask"].squeeze(),   # маска
            "token_type_ids": encoded["token_type_ids"].squeeze(),   # тип сегмента
            "label": torch.tensor(label)                             # 0/1
        }


# #### Инициализация `Dataloader`

# In[57]:


# загружаем токенизатор той же модели, что и BERT
tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")

# создаём PyTorch Dataset
train_dataset = MatchDataset(train_df, tokenizer, max_len=64)
val_dataset   = MatchDataset(val_df, tokenizer, max_len=64)
test_dataset  = MatchDataset(test_df, tokenizer, max_len=64)

# создаём DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,  pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=16, shuffle=False, pin_memory=True)


# #### Инициализация модели

# In[58]:


# Модель: BERT как энкодер + линейная голова для предсказания похожести
class BertForMatching(nn.Module):
    def __init__(self, model_name="DeepPavlov/rubert-base-cased"):
        super().__init__()

        # загружаем RuBERT (все слои, без классификатора)
        self.bert = AutoModel.from_pretrained(model_name)

        # Linear-слой: 768 → 1 (прогноз совпадения)
        self.fc = nn.Linear(self.bert.config.hidden_size, 1)

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


# #### Обучение модели

# Объявляем функцию для обучения.

# In[59]:


# Основной цикл обучения
def train(model, loader, epochs=3, lr=2e-5):
    model = model.cuda()  # переносим модель на GPU

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # лосс для бинарной классификации
    criterion = nn.BCEWithLogitsLoss()

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
        val_loss, acc, f1, auc = evaluate(model, val_loader, criterion)
        print(f"Validation: Loss={val_loss:.4f} | Acc={acc:.4f} | F1={f1:.4f} | AUC={auc:.4f}")

        # Сохраняем лучшую модель
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.bin")
            print("✓ Лучшая модель сохранена → best_model.bin")

    return model


# Напишем функцию для оценки качества модели (для валидации)

# In[60]:


def evaluate(model, loader, criterion):
    """Возвращает средний val loss + accuracy + F1 + AUC."""
    model.eval()
    total_loss = 0

    all_labels = []
    all_scores = []

    with torch.no_grad():
        for batch in tqdm(loader):
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            token_type_ids = batch["token_type_ids"].cuda()
            labels = batch["label"].cuda()

            logits = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            scores = torch.sigmoid(logits).cpu().numpy()
            all_scores.extend(scores)
            all_labels.extend(labels.cpu().numpy())

    val_loss = total_loss / len(loader)

    preds = (torch.tensor(all_scores) > 0.5).numpy()

    acc = accuracy_score(all_labels, preds)
    f1  = f1_score(all_labels, preds)
    auc = roc_auc_score(all_labels, all_scores)

    return val_loss, acc, f1, auc


# Запустим обучение

# In[61]:


model = BertForMatching()
model = train(model, train_loader, epochs=10)


# #### Сохранение токенизатора

# In[62]:


tokenizer.save_pretrained("matching_model_tokenizer")


# #### Инференс модели

# Загрузим нашу модель.

# In[63]:


# загружаем tokenizer
tokenizer = AutoTokenizer.from_pretrained("matching_model_tokenizer")

# создаём модель и грузим веса
model = BertForMatching()
model.load_state_dict(torch.load("best_model.bin", map_location="cpu"))
model = model.cuda()
model.eval()   # режим инференса


# ### Пример работы модели

# In[68]:


# два текста
reference = 'Лабубу серый редкий в костюме Labubu'
candidate = 'Labubu серый игрушка в костюме ЗАяц'

# токенизация пары текстов
encoded = tokenizer(
    reference,
    candidate,
    padding="max_length",
    truncation=True,
    max_length=64,
    return_tensors="pt"
)

with torch.no_grad():
    logits = model(encoded['input_ids'].cuda(), encoded['attention_mask'].cuda(), encoded['token_type_ids'].cuda())
    score = torch.sigmoid(logits).cpu().numpy()

print('Схожесть 2-х текстов:', score.round(3)[0])


# #### Проверим качество на тестовом датасете.

# In[65]:


loss, acc, f1, auc = evaluate(model, test_loader, nn.BCEWithLogitsLoss())
print(f"Тестовые метрики: ACC={acc:.4f} | F1={f1:.4f} | AUC={auc:.4f}")


# Мы получили отличную `matching-модель`, которая умеет сравнивать два текстовых названия объявлений и выдавать непрерывный similarity-score от **0** до **1**.
