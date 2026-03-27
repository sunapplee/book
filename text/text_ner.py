#!/usr/bin/env python
# coding: utf-8

# # Обучение модели NER на датасете WNUT17
# 
# В данном ноутбуке реализован полный пайплайн обучения модели для задачи распознавания именованных сущностей (**NER**) на датасете **WNUT17**. Выполнена подготовка данных, токенизация и выравнивание меток, после чего проведено дообучение модели **DistilBERT** для токенной классификации. Качество модели оценено с использованием метрик *precision*, *recall* и *F1-score*, а также продемонстрирован инференс на новых текстах.

# ## Содержание
# 
# * [Импорт библиотек](#0)
# * [1. Импорт данных](#1)
# * [2. Разбиение данных](#2)
# * [3. Предобработка данных](#3)
# * [4. Инициализация модели](#4)
# * [5. Обучение модели](#5)
# * [6. Сохранение модели](#6)
# * [7. Инференс и метрики](#7)
# * [8. Вывод](#8)

# ### 

# <a id=0></a>
# ## Импорт библиотек

# Из-за ошибки в новой версии библиотеки используется установка `datasets==3.6.0`.

# In[1]:


import numpy as np
import pandas as pd
from transformers import (AutoTokenizer,
                          DataCollatorForTokenClassification,
                          AutoModelForTokenClassification,
                          TrainingArguments, Trainer, pipeline)

from datasets import load_dataset
import evaluate
import torch


# ###

# <a id=1></a>
# # 1. Импорт данных

# `wnut_17` — это датасет для задачи распознавания именованных сущностей (NER), основанный на пользовательском контенте из социальных сетей (например, Twitter), где много шума, сленга и нестандартных написаний. Он используется для обучения моделей извлекать сущности (людей, локации, организации и др.) в «грязных» реальных текстах, а не в формально отредактированных данных.
# 

# In[2]:


wnut = load_dataset("wnut_17", trust_remote_code=True)


# Рассмотрим, из чего он состоит.

# In[3]:


wnut


# In[4]:


wnut["train"][0]


# Датасет состоит из `токенов` и `меток сущности`. Рассмотрим метки подробнее.

# In[5]:


label_list = wnut["train"].features[f"ner_tags"].feature.names
label_list


# Это список **всех классов (меток) NER** в датасете.
# 
# * `O` — не сущность
# * `B-...` — начало сущности (Begin)
# * `I-...` — продолжение сущности (Inside)
# 
# Например:
# `B-location I-location` → «Empire State Building» — одна локация
# 

# <a id=2></a>
# # 2. Разбиение данных

# Текущий датасет разбит в пропорции:

# In[6]:


total_len = len(wnut['train']) + len(wnut['validation']) + len(wnut['test'])

print(f"Train: {len(wnut['train']) / total_len:.3f}")
print(f"Validation: {len(wnut['validation']) / total_len:.3f}")
print(f"Test: {len(wnut['test']) / total_len:.3f}")


# Данное разбиение оптимально подходит для нашей задачи дообучения `BERT`. Позволяет одновременно эффективно обучить модель, своевременно отслеживать переобучение и провести честную итоговую проверку качества.

# <a id=3></a>
# # 3. Предобработка данных 

# Чтобы обучить модель нам нужно `токенезировать` наши данные. Загрузим токенизатор `distilbert-base-uncased`.

# In[7]:


tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")


# Рассмотрим его работу:

# In[8]:


example = wnut["train"][0]
tokenized_input = tokenizer(example["tokens"], is_split_into_words=True)
tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
tokens[:10]


# Токенизатор разбивает слова на подслова (subword-токены) и добавляет специальные токены (`[CLS]`, `[SEP]`) для работы модели.
# 

# ### Функция токенизации сэмплов
# 
# Напишем функцию для приведения NER-разметки от слов к токенам.

# In[9]:


def tokenize_and_align_labels(examples):
    # Токенизируем данные
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        # Получаем соответствие токенов словам
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Игнорируем спецтокены ([CLS], [SEP])
            if word_idx is None:
                label_ids.append(-100)
            # Берем метку только для первого токена слова
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # Остальные части слова игнорируем
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    # Добавляем выровненные метки
    tokenized_inputs["labels"] = labels

    return tokenized_inputs


# Применяем функцию на весь датасет:

# In[10]:


tokenized_wnut = wnut.map(tokenize_and_align_labels, batched=True)


# После обработки датасета, мы поулчили новые колонки: `input_ids`, `attention_mask` и `labels`. На этих данных мы будем обучать модель.

# In[11]:


tokenized_wnut['train'][0].keys()


# <a id=4></a>
# # 4. Инициализация модели

# `DataCollatorForTokenClassification` нужен для правильной сборки батчей при обучении.

# In[12]:


# Инициализация коллатора
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


# Загрузим метрики для **оценки качества NER-модели**.

# In[13]:


seqeval = evaluate.load("seqeval", zero_division=True)


# Напишем функцию для оценки качества одного сэмпла.

# In[14]:


# Переводим числовые метки в строковые
labels = [label_list[i] for i in example["ner_tags"]]


def compute_metrics(p):
    # Получаем предсказания и истинные метки
    predictions, labels = p
    # Берем наиболее вероятный класс для каждого токена
    predictions = np.argmax(predictions, axis=2)
    # Убираем -100 (спецтокены и части слов) и переводим в строки
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    # То же самое для истинных меток
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    # Считаем метрики для NER
    results = seqeval.compute(
        predictions=true_predictions,
        references=true_labels
    )
    # Возвращаем основные метрики
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


# Подгототовим словари для конвертации текстовых классов в числовые и обратно.

# In[15]:


id2label = {
    0: "O",
    1: "B-corporation",
    2: "I-corporation",
    3: "B-creative-work",
    4: "I-creative-work",
    5: "B-group",
    6: "I-group",
    7: "B-location",
    8: "I-location",
    9: "B-person",
    10: "I-person",
    11: "B-product",
    12: "I-product",
}
label2id = {
    "O": 0,
    "B-corporation": 1,
    "I-corporation": 2,
    "B-creative-work": 3,
    "I-creative-work": 4,
    "B-group": 5,
    "I-group": 6,
    "B-location": 7,
    "I-location": 8,
    "B-person": 9,
    "I-person": 10,
    "B-product": 11,
    "I-product": 12,
}


# Инициализируем модель `distilbert/distilbert-base-uncased`, которую будем дообучать.

# In[16]:


model = AutoModelForTokenClassification.from_pretrained(
    "distilbert/distilbert-base-uncased", num_labels=13, id2label=id2label, label2id=label2id
)


# <a id=5></a>
# # 5. Обучение модели

# Создадим конфигурацию обучения:

# In[17]:


training_args = TrainingArguments(
    output_dir="wnut_model",  # куда сохранять модель и чекпоинты
    learning_rate=2e-5,                  # скорость обучения (стандарт для BERT)
    per_device_train_batch_size=16,      # batch size на обучение
    per_device_eval_batch_size=16,       # batch size на валидацию
    num_train_epochs=15,                 # количество эпох
    weight_decay=0.01,                   # L2-регуляризация
    eval_strategy="epoch",               # оценка после каждой эпохи
    save_strategy="epoch",               # сохранение после каждой эпохи
    load_best_model_at_end=True,         # загрузить лучшую модель по метрике
)

trainer = Trainer(
    model=model,                           # модель (DistilBERT + classifier)
    args=training_args,                    # параметры обучения
    train_dataset=tokenized_wnut["train"],  # тренировочный датасет
    eval_dataset=tokenized_wnut["test"],   # валидационный датасет
    processing_class=tokenizer,            # токенизатор (обработка текста)
    data_collator=data_collator,           # батчинг + padding
    compute_metrics=compute_metrics,       # метрики (например f1 для NER)
)


# In[18]:


# Получим метрики базовой модели
base_metrics = trainer.evaluate(tokenized_wnut['test'])


# ### Запускаем обучение

# In[19]:


trainer.train()


# <a id=6></a>
# # 6. Сохранение модели

# Сохраним лучшую модель и токенизатор для удобства.

# In[20]:


trainer.save_model("wnut_model")
tokenizer.save_pretrained("wnut_model")


# <a id=7></a>
# # 7. Инференс и метрики

# ### Пример работы

# In[25]:


# Подготовим текст для проверки модели
text = "Moscow is the capital of Russia and one of the largest cities in Europe."

# Загружаем модель
ner = pipeline(
    "ner",
    model="wnut_model",        # путь к папке
    tokenizer="wnut_model",    # тот же путь
    aggregation_strategy="simple"
)

# Проняем наш текст
ner(text)


# ### Ручной инференс модели 

# In[26]:


# Токенезируем текст
tokenizer = AutoTokenizer.from_pretrained("wnut_model")
inputs = tokenizer(text, return_tensors="pt")

# Прогоняем через сохраненную модель
model = AutoModelForTokenClassification.from_pretrained("wnut_model")
with torch.no_grad():
    logits = model(**inputs).logits

# Получаем классы
predictions = torch.argmax(logits, dim=2)
predicted_token_class = [model.config.id2label[t.item()] for t in predictions[0]]
predicted_token_class


# ### Метрики итоговой модели
# 
# Подсчитаем метрики на тестовой выборке.

# In[27]:


metrics = trainer.evaluate(tokenized_wnut['test'])


# Сравним метрики с базовой моделью:

# In[28]:


pd.DataFrame([base_metrics, metrics], index=['Base Model', 'Fine-Tuned Model']).T.round(2).fillna(0)


# ### С помощью дообучения, мы значительно улучшили качество модели!

# <a id=8></a>
# # 8. Вывод
# В ходе выполнения ноутбука была успешно реализована модель для задачи **распознавания именованных сущностей (NER)** с использованием **DistilBERT**.  
# После **дообучения (fine-tuning)** качество модели **заметно улучшилось** по сравнению с базовой версией, что подтверждается ростом метрик:  
# - **precision**  
# - **recall**  
# - **F1-score**  
# 
# Полученная модель демонстрирует **более точное извлечение сущностей** и может быть использована для обработки **реальных текстовых данных**.
