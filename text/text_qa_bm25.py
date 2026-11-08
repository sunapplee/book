#!/usr/bin/env python
# coding: utf-8

# # Поиск ответов на вопросы с использованием BM25
# 
# В данном ноутбуке реализуется простая **вопросно-ответная система (Question Answering, QA)** на основе алгоритма информационного поиска **BM25**. В ходе работы выполняется загрузка и предобработка текстовых данных, включающая очистку текста, нормализацию регистра, удаление спецсимволов и обработку стоп-слов. Также применяется аугментация вопросов для повышения устойчивости системы к различным формулировкам пользовательских запросов.
# 
# После подготовки данных ответы индексируются с помощью алгоритма **BM25**, который позволяет находить наиболее релевантный ответ для заданного вопроса. Качество найденных ответов оценивается с использованием **эмбеддингов модели Qwen3-Embedding** и вычисления **косинусного сходства** между истинными и предсказанными ответами.

# # Содержание
# 
# * [Импорт библиотек](#0)
# * [1. Загрузка данных](#1)
# * [2. Предобработка данных](#2)
# * [3. Аугментация данных](#3)
# * [4. Алгоритм поиска BM25](#4)

# <a id=0></a>
# ## Импорт библиотек

# In[ ]:


import pandas as pd
import numpy as np
import missingno as msno
import string
import regex

from pymorphy3 import MorphAnalyzer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from augmentex import WordAug, CharAug
from rank_bm25 import BM25Okapi

import ollama
from sklearn.metrics.pairwise import cosine_similarity

from joblib import Parallel, delayed


# In[101]:


# Логирование
from tqdm.auto import tqdm
tqdm.pandas()


# ###

# <a id=1></a>
# # 1. Загрузка данных

# In[3]:


df = pd.read_json("dataset.jsonl", lines=True)


# In[4]:


print('Размер датасета:', df.shape)

df.head()


# ###

# <a id=2></a>
# # 2. Предобработка данных

# ### Анализ пропусков

# In[5]:


df.apply(lambda x: x == '').sum()


# In[6]:


# Пропуск заполнен NaN
nan_rows = df.isna().sum()

# Пропуск заполнен пустой строкой
empty_rows = df.apply(lambda x: x == '').sum()

print('Всего пропусков:\n', (empty_rows + nan_rows) / df.shape[0])


# Из вывода видно, что в пропусков встречаются в колонке ```description```, где пустые значения находятся в 82% случаях. 
# 
# Скорее всего, в основном, люди ограничиваются указанием вопроса без описания.
# 
# Для задачи `QA` эта колонка нам пока не нужна. 

# In[7]:


df = df.drop(columns=['description'], errors='ignore')

df.sample(3)


# Таким образом мы оставили только релевантные для нас колонки.

# ### Анализ выбросов

# In[8]:


print('Дубликатов:', df.duplicated().sum())


# Дубликаты для обучения модели ни к чему, избавимся от них.

# In[9]:


df = df.drop_duplicates()


# ### Смесь латиницы и кириллицы

# Проведем анализ одного сэмпла, красный символ означает латиницу.

# In[10]:


text = df['question'].iloc[444] + df['answer'].iloc[444]
res = ''
for char in text:
    if char in string.ascii_lowercase:
        res += f"\x1b[0;30;41m{char}\x1b[0m"
    else:
        res += char
print(res)


# Из текста видно, что смешение символов не происходит, все слова написаны кириллицей. Поэтому действий не предпринимаем.

# ### Эмодзи и спецсимволы
# 
# 
# В тексте могут встречаться **эмодзи**. Зачастую они не несут никакой смысловой нагрузки, а являются только *фактором привлечения внимания*. Для корректной работы модели в будущем *избавимся от них*.
# 
# **Спецсимволы** встречаются редко, однако тоже не несут смысла и будут вредно сказываться на качестве данных.
# 
# **Пунктуация** в задаче QA также будет только во вред.
# 
# Оставляем только буквы, цифры, пробелы. `Пунктуацию`, ```Спецсимволы```, ```Unicode-символы``` и ```эмодзи``` будем удалять, что явялется стандартной практикой.

# In[11]:


df['answer'].iloc[12234]


# In[12]:


def keep_letters_numbers(text):
    # Оставляем только буквы, цифры, пробелы
    return regex.sub(r"[^\p{L}\p{N} ]", "", text)

keep_letters_numbers(df['answer'].iloc[12234])


# Функция работает! Распространим на весь датасет

# In[13]:


get_ipython().run_cell_magic('time', '', "df.loc[:, 'question'] = df['question'].apply(keep_letters_numbers)\ndf.loc[:, 'answer'] = df['answer'].apply(keep_letters_numbers)\n")


# In[14]:


df['answer'].iloc[123]


# ### Удаление стоп-слов

# In[15]:


# Приводим в нижний регистр
df.loc[:, 'question'] = df['question'].str.lower()
df.loc[:, 'answer'] = df['answer'].str.lower()


# In[16]:


stop_words = set(stopwords.words('russian'))
def remove_stopwords(text):
    tokens = word_tokenize(text)
    return ' '.join([word for word in tokens if word not in stop_words])


# In[17]:


get_ipython().run_cell_magic('time', '', "# Будет обрабатывать каждую колонку\ndf.loc[:, 'question'] = df['question'].progress_apply(remove_stopwords)\ndf.loc[:, 'answer'] = df['answer'].progress_apply(remove_stopwords)\n")


# In[18]:


df.sample(3)


# ### Cтемминг и лемматизация
# 
# Для лемматизации могли бы использовать `pymorphy3`, однако из-за большого количества данных этот подход оказывается слишком медленным. Стемминг работает значительно быстрее, так как не требует словарного разбора, а лишь обрезает окончания по правилам. Поэтому будем использовать `SnowballStemmer`, который поддерживает русский язык и хорошо справляется с задачей нормализации текста.

# In[19]:


# Реалиазция лемматизации
morph = MorphAnalyzer()

def lemmatize(text):
    tokens = text.split()
    return ' '.join([morph.parse(word)[0].normal_form for word in tokens])


# In[20]:


# Реализация стемминга
stemmer = SnowballStemmer("russian")

def stem(text):
    tokens = text.split()
    return ' '.join([stemmer.stem(word) for word in tokens])


# In[21]:


get_ipython().run_cell_magic('time', '', "# Будет обрабатывать каждую колонку\ncols = ['question', 'answer']\n\nwith Parallel(n_jobs=-1, batch_size=16) as parallel:\n    for col in cols:\n        df.loc[:50000, col] = parallel(\n            delayed(stem)(text) for text in df.loc[:50000, col]\n        )\n")


# ### Сбор датасета

# Итоговый датасет имеет такой вид.

# In[22]:


df.sample(4)


# ###

# <a id=3></a>
# # 3. Аугментация данных

# Для аугментации текста воспользуемся библиотекой `augmentex`. Применяются аугментации на уровне слов (например, `replace`, `delete`, `swap`, `stopword`, `split`, `reverse`, `text2emoji`), которые изменяют структуру и состав слов в предложении. Также используются символьные аугментации (`shift`, `orfo`, `typo`, `delete`, `insert`, `multiply`, `swap`), имитирующие опечатки, ошибки написания и случайные изменения символов.
# 

# ### Aугментации на уровне слов

# In[56]:


word_aug = WordAug(
    unit_prob=0.4, # вероятность применения аугментации
    min_aug=1, # мин. аугментаций
    max_aug=5, # макс. аугментаций
    lang="rus", # язык
    platform="pc", # платформа
    random_seed=42,
)


# Для проведения аугментации на уровне слов будет случайным образом выбрана подвыборка из **1000 сэмплов**. К выбранным данным будут применены различные методы аугментации **только к вопросам**, чтобы сгенерировать их альтернативные формулировки. При этом **ответы остаются неизменными**, поскольку они являются эталонными и не должны искажаться в процессе подготовки данных.

# In[70]:


# Подвыборка
word_aug_subset = df.sample(1000)

# Аугментируем каждый вопрос
aug_questions = word_aug.aug_batch(word_aug_subset['question'].tolist(), batch_prob=1)

word_aug_subset['question'] = aug_questions
word_aug_subset.head(3)


# ### Aугментации на уровне символов

# In[71]:


char_aug = CharAug(
    unit_prob=0.4, # вероятность применения аугментации
    min_aug=1, # мин. аугментаций
    max_aug=5, # макс. аугментаций
    mult_num=3, # число повторений
    lang="rus", # язык
    platform="pc", # платформа
    random_seed=42,
)


# Аналогично, для посимвольной аугментации также обработаем **1000 сэмплов**.

# In[72]:


# Подвыборка
char_aug_subset = df.sample(1000)

# Аугментируем каждый вопрос
aug_questions = char_aug.aug_batch(char_aug_subset['question'].tolist(), batch_prob=1)

char_aug_subset['question'] = aug_questions
char_aug_subset.head(3)


# Добавляем к исходному датасету **аугментированный**.

# In[73]:


print('Размер датасета ДО аугментаций:', df.shape)

df = pd.concat([df, word_aug_subset, char_aug_subset])

print('Размер датасета ПОСЛЕ аугментаций:', df.shape)


# ###

# <a id=4></a>
# # 4. Алгоритм поиска BM25
# 
# Для создания простой QA-системы используется алгоритм ранжирования **BM25**, широко применяемый в задачах информационного поиска. Данный метод позволяет находить наиболее релевантный ответ из базы, оценивая соответствие текста ответа пользовательскому вопросу на основе частоты терминов и их редкости в корпусе.
# 
# В рамках подхода все ответы индексируются, после чего для каждого вопроса вычисляется степень релевантности ко всем ответам. В качестве итогового предсказания выбирается ответ с максимальным значением BM25-оценки.

# ### Подготовка данных
# 
# Алгоритм BM25 не требует этапа обучения, поэтому разделение данных на обучающую и тестовую выборки не выполняется. Датасет полносью готов к индексации.

# In[117]:


questions = df['question'].tolist()[:1000]
answers = df['answer'].tolist()[:1000]


# ### Инициализация метода

# In[118]:


tokenized_answers = [a.split() for a in answers]
bm25 = BM25Okapi(tokenized_answers)


# ### Инференс системы
# 
# Для оценки качества найденных ответов используется модель **Qwen3-Embedding**. С помощью модели вычисляются эмбеддинги истинных и предсказанных ответов, после чего между ними считается косинусное сходство.

# In[119]:


# Получаем релевантные ответы
preds = [
    answers[np.argmax(bm25.get_scores(q.split()))]
    for q in tqdm(questions)
]


# In[120]:


get_ipython().run_cell_magic('time', '', "# Считаем косинусное сходство\nembed_true = ollama.embed(\n  model='qwen3-embedding:0.6b',\n  input=answers\n)\nembed_pred = ollama.embed(\n  model='qwen3-embedding:0.6b',\n  input=preds\n)\n")


# In[121]:


# Извлекаем эмбеддинги
emb_true = np.array(embed_true["embeddings"])
emb_pred = np.array(embed_pred["embeddings"])

# Считаем метрику
cos_sim = cosine_similarity(emb_true, emb_pred).diagonal()


# Проанализируем полученные значения сходства с помощью описательной статистики.

# In[122]:


pd.Series(cos_sim).describe().to_frame().T

