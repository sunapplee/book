#!/usr/bin/env python
# coding: utf-8

# # Анализ текста и привидение его к таблинчому формату

# ### Установка библиотек

# ## Импорт данных
# 
# Самые популярные форматы: txt, pdf, docs. Никто не запрещает любой текст привести к формату txt. Для примера загрузим эти форматы.

# ### txt

# In[4]:


def read_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

read_txt('text_data/0/0.txt')[:100]


# ### docs

# In[5]:


from docx import Document

def read_docx(path):
    doc = Document(path)

    return "\n".join([p.text for p in doc.paragraphs])

read_docx('text_data/1/1.docx')[:100]


# ### pdf

# In[8]:


import fitz  # PyMuPDF

def read_pdf(path):
    text = ""
    with fitz.open(path) as doc:
        for page in doc:
            text += page.get_text()
    return text

read_pdf('text_data/2/2.pdf')[:100]


# Представим, нам дали набор txt файлов.

# In[9]:


import pandas as pd
import glob
import os

files = []
texts = []
labels = []

classes = os.listdir('text_data/')
for target in classes:
    file_paths = glob.glob(rf'text_data/{target}/*.txt')
    for file_path in file_paths:
        files.append(file_path)
        texts.append(read_txt(file_path))
        labels.append(target)

df = pd.DataFrame(
    {
        'file_path': files,
        'text': texts,
        'label': labels
    }
)

df.head(3)


# ## Базовая отчистка текста

# In[10]:


import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zа-я0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


df['text'] = df['text'].map(clean_text)
df.head(3)


# ## Преобразование текста в числовые признаки

# In[11]:


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=1000,
)

preprocess_text = vectorizer.fit_transform(df['text']).toarray()
preprocess_text_df = pd.DataFrame(preprocess_text, 
                            columns=[f'feature_{i}' for i in range(1000)])

total_df = pd.concat([df, preprocess_text_df], axis=1)


# ## Итоговый датасет
# 
# Данный датасет можно использовать для дальнейшнего анализа данных.

# In[12]:


total_df.head(3)

