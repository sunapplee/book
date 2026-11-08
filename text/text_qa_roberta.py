#!/usr/bin/env python
# coding: utf-8

# # Дообучение модели RoBERTa для Question Answering
# 
# В данном ноутбуке выполняется дообучение трансформерной модели **RoBERTa** для задачи **Question Answering (Q/A)** на датасете **SberQuAD**.
# 
# В работе выполняются:
# - подготовка и токенизация данных  
# - дообучение модели  
# - подбор гиперпараметров  
# - инференс на валидационной выборке  
# - оценка качества (Exact Match и F1)  
# - сохранение обученной модели

# ## Содержание
# 
# * [1. Дообучение модели Q/A](#1)
#     * [1.1 Выбор Q/A датасета](#1-1)
#     * [1.2 Разбиение данных](#1-2)
#     * [1.3 Выбор модели](#1-3)
#     * [1.4 Обучение модели](#1-4)
#     * [1.5 Инференс модели](#1-5)
#     * [1.6 Метрики и сравнение с базовой RoBERTa](#1-6)
#     * [1.7 Выбор гиперпараметров для подбора](#1-7)
#     * [1.8 Подбор гиперпараметров](#1-8)
#     * [1.9 Метрики после подбора](#1-9)
#     * [1.10 Сохранение модели](#1-10)
#     * [1.11 Результаты работы модели](#1-11)

# ## Импорт библиотек

# In[1]:


import pandas as pd
import numpy as np
from tqdm import tqdm

from datasets import load_dataset

from torch.utils.data import DataLoader
import evaluate
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator,
)

import matplotlib.pyplot as plt


# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ##

# <a id=1></a>
# 
# # 1. Дообучение модели Q/A 

# <a id=1-1></a>
# ## 1.1 Выбор Q/A датасета
# Для решения задачи построения вопросно-ответной системы (Q/A) был выбран русскоязычный датасет `SQuAD`, представляющий собой адаптацию набора данных SQuAD для русского языка. Данный датасет содержит пары «вопрос–контекст–ответ», где ответ представляет собой фрагмент текста, извлекаемый из предоставленного контекста.
# > [Датасет доступен на платформе Hugging Face](https://huggingface.co/datasets/kuznetsoffandrey/sberquad)

# In[3]:


dataset = load_dataset(
    "parquet",
    data_files={
        "train": "data/train-00000-of-00001.parquet",
        "validation": "data/validation-00000-of-00001.parquet",
    }
)

print(dataset)


# In[4]:


example = dataset["train"][0]
print('Пример из датасета:')
for key, value in example.items():
    if key == "context":
        print(f"{key}: {value[:200]}...")
    else:
        print(f"{key}: {value}")


# <a id=1-2></a>
# ## 1.2 Разбиение данных
# 
# Так как данные уже распределены по файлам, то самостоятельно это делать не нужно. Текущая пропорция данных: `90/10`. Данное соотношение позволит качественно дообучить модель и объективно оценить метрику.

# <a id=1-3></a>
# ## 1.3 Выбор модели
# 
# Для решения задачи вопросно-ответного анализа могут использоваться различные архитектуры нейронных сетей. Распространённым подходом являются encoder-модели семейства **BERT**, например **RoBERTa**, которые извлекают ответ из заданного контекста. Также существуют encoder-decoder модели, такие как **T5**, и decoder-модели, например **GPT-2**, способные генерировать ответы. Однако для рассматриваемой задачи наиболее подходящей является **RoBERTa**, поскольку используемый датасет предполагает извлечение ответа непосредственно из контекста, а модели семейства BERT демонстрируют высокую эффективность в задачах *extractive question answering*.

# Для обучения будет использована базовая версия модели — **RoBERTa-base**, которая обладает сбалансированным соотношением качества и вычислительных требований и часто применяется в качестве baseline-модели.

# In[5]:


# Инициализируем модель
model_name = "FacebookAI/roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name, ignore_mismatched_sizes=True)

# Переводим на GPU (если есть)
model.to(device)

print('Модель имеет:', sum(p.numel() for p in model.parameters()) / 1e+6, 'МЛН параметров')


# <a id=1-4></a>
# ## 1.4 Обучение модели

# ### Подготовка данных для обучения

# In[6]:


MAX_LENGTH = 384  # Максимальная длина входа
DOC_STRIDE = 64  # Шаг при разбиении длинных контекстов
TRAIN_SIZE = 10000  # Сколько примеров для обучения
VAL_SIZE = 1000     # Сколько примеров для валидации


# Напишем функцию подготовки данных.

# In[7]:


def prepare_train_features(examples):
    # Токенизируем question и context
    tokenized = tokenizer(
        examples["question"],      # список вопросов
        examples["context"],       # список контекстов
        truncation=True,  # обрезаем только context (question не трогаем)
        max_length=MAX_LENGTH,     # максимальная длина последовательности
        stride=DOC_STRIDE,         # перекрытие окон при длинном контексте
        return_overflowing_tokens=True,  # разбиваем длинный context на несколько окон
        return_offsets_mapping=True,     # вернуть mapping: token -> (start_char, end_char)
        padding="max_length",     # паддинг до max_length
        return_token_type_ids=False
    )

    # mapping: какое окно относится к какому исходному примеру
    # например: [0,0,0,1,1] → первые 3 окна из example0, следующие 2 из example1
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")

    # offsets: для каждого токена хранится (start_char, end_char) в исходном тексте
    # это нужно чтобы перевести char-позицию ответа → token-позицию
    offset_mapping = tokenized["offset_mapping"]
    tokenized["context"] = []

    # списки, куда будем записывать позиции ответа
    tokenized["start_positions"] = []
    tokenized["end_positions"] = []

    # проходим по каждому window (каждой токенизированной последовательности)
    for i, offsets in enumerate(offset_mapping):

        # узнаём из какого исходного примера этот window
        sample_idx = sample_mapping[i]
        tokenized["context"].append(examples["context"][sample_idx])

        # берём ответ для этого примера
        answers = examples["answers"][sample_idx]

        # если ответа нет
        if len(answers["answer_start"]) == 0:
            # ставим 0 → обычно это CLS токен
            tokenized["start_positions"].append(0)
            tokenized["end_positions"].append(0)
            continue

        # начало ответа в символах
        start_char = answers["answer_start"][0]

        # конец ответа в символах
        end_char = start_char + len(answers["text"][0])

        # показывает какие токены относятся к question и context
        # None → special tokens
        # 0 → question
        # 1 → context
        sequence_ids = tokenized.sequence_ids(i)

        # ищем первый токен контекста
        ctx_start = next(i for i, s in enumerate(sequence_ids) if s == 1)

        # ищем последний токен контекста
        ctx_end = max(i for i, s in enumerate(sequence_ids) if s == 1)

        # проверяем находится ли ответ внутри текущего окна
        if offsets[ctx_start][0] > end_char or offsets[ctx_end][1] < start_char:
            # если ответ вне окна → ставим (0,0)
            tokenized["start_positions"].append(0)
            tokenized["end_positions"].append(0)
        else:
            # ищем start token

            idx = ctx_start
            # двигаемся пока начало токена <= start_char
            while idx <= ctx_end and offsets[idx][0] <= start_char:
                idx += 1

            # предыдущий токен содержит начало ответа
            tokenized["start_positions"].append(idx - 1)

            # ищем end token

            idx = ctx_end
            # двигаемся назад пока конец токена >= end_char
            while idx >= ctx_start and offsets[idx][1] >= end_char:
                idx -= 1

            # следующий токен содержит конец ответа
            tokenized["end_positions"].append(idx + 1)

    # возвращаем токенизированные данные + позиции ответа
    return tokenized


# Применим функцию к датасету.

# In[8]:


train_subset = dataset["train"].select(range(TRAIN_SIZE))
val_subset = dataset["validation"].select(range(VAL_SIZE))

tokenized_train = train_subset.map(
    prepare_train_features,
    batched=True,
    remove_columns=train_subset.column_names,
    desc="Токенизация train"
)

tokenized_val = val_subset.map(
    prepare_train_features,
    batched=True,
    remove_columns=val_subset.column_names,
    desc="Токенизация validation"
)

len(tokenized_train), len(tokenized_val)


# В итоге получаем датасет, полностью готовый для обучения!

# In[9]:


tokenized_train


# ### Запуск обучения

# In[10]:


# Конфигурация обучения
training_args = TrainingArguments(
    output_dir="./qa_bert_finetuned",       # папка для чекпоинтов
    eval_strategy="epoch",                  # оценка после каждой эпохи
    learning_rate=3e-5,                     # learning rate (типичный для BERT)
    per_device_train_batch_size=8,          # batch size
    per_device_eval_batch_size=8,
    num_train_epochs=2,                     # число эпох
    weight_decay=0.01,                      # регуляризация
    warmup_steps=100,                       # warmup первые 100 шагов
    logging_steps=500,                      # как часто писать логи
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",                       # отключаем wandb
    fp16=True # Ускорение
)

# класс обучения
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    processing_class=tokenizer,
    data_collator=default_data_collator,
)


# In[11]:


# Очищаем память перед обучением
torch.cuda.empty_cache()


# In[12]:


print('Запуск fine-tuning...')
train_result = trainer.train()
print('Обучение завершено!')


# <a id=1-5></a>
# ## 1.5 Инференс модели

# In[13]:


# Переводим модель в режим инфреренса
model_finetuned = trainer.model
model_finetuned.eval()
model_finetuned = model_finetuned.to(device)


# In[14]:


# Загрузим evaluate для нашего датасета
squad_metric = evaluate.load("squad")


# In[15]:


# Данные будем передавать батчами
batch_size = 32
tokenized_val.set_format(type="torch")
dataloader = DataLoader(tokenized_val, batch_size=batch_size)

# ограничения для поиска ответа
max_answer_length = 30   # максимальная длина ответа
top_k = 20               # сколько лучших start/end токенов рассматривать

# сюда будем сохранять позиции предсказанных ответов
all_start_preds = []
all_end_preds = []

# проходим по батчам
for batch in tqdm(dataloader, desc="Inference"):

    # переносим данные на устройство
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    # инференс модели
    with torch.no_grad():
        outputs = model_finetuned(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    # logits начала и конца ответа
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    current_batch_size = start_logits.size(0)

    # обрабатываем каждый пример в батче
    for i in range(current_batch_size):

        start_log = start_logits[i]
        end_log = end_logits[i]

        # берём top-k наиболее вероятных start и end позиций
        start_top = torch.topk(start_log, top_k).indices
        end_top = torch.topk(end_log, top_k).indices

        best_score = -1e9
        best_start = 0
        best_end = 0

        # ищем лучший span среди top-k кандидатов
        for start_idx in start_top:
            for end_idx in end_top:

                # конец ответа не может быть раньше начала
                if end_idx < start_idx:
                    continue

                # ограничиваем максимальную длину ответа
                if end_idx - start_idx > max_answer_length:
                    continue

                # score span = сумма вероятностей start и end
                score = start_log[start_idx] + end_log[end_idx]

                # если нашли более вероятный span — сохраняем
                if score > best_score:
                    best_score = score
                    best_start = start_idx
                    best_end = end_idx

        # сохраняем позиции лучшего span
        all_start_preds.append(best_start)
        all_end_preds.append(best_end)


# In[24]:


# Списки для предсказаний и эталонных ответов
predictions = []
references = []

# Проходим по всем примерам
for i in range(len(all_start_preds)):
    # Предсказанные позиции начала и конца ответа
    start = all_start_preds[i]
    end = all_end_preds[i]
    # Токены текущего примера
    input_ids = tokenized_val[i]["input_ids"]

    # конец не может быть раньше начала
    if end < start:
        end = start
    # Токены предсказанного ответа
    offsets = tokenized_val[i]["offset_mapping"]
    context = tokenized_val[i]["context"]
    start_char = offsets[start][0]
    end_char = offsets[end][1]
    pred_text = context[start_char:end_char]
    # Добавляем предсказание
    predictions.append({
        "id": str(i),
        "prediction_text": pred_text
    })
    # Истинные позиции ответа
    true_start = tokenized_val[i]["start_positions"]
    true_end = tokenized_val[i]["end_positions"]
    # Токены правильного ответа
    offsets = tokenized_val[i]["offset_mapping"]
    context = tokenized_val[i]["context"]
    start_char = offsets[true_start][0]
    end_char = offsets[true_end][1]
    true_text = context[start_char:end_char]
    # Добавляем эталонный ответ
    references.append({
        "id": str(i),
        "answers": {
            "text": [true_text],
            "answer_start": [true_start.item()]
        }
    })


# <a id=1-6></a>
# ## 1.6 Метрики и сравнение с базовой RoBERTa

# ### Загрузка базовой модели

# In[25]:


# baseline модель (без дообучения)
baseline_model = AutoModelForQuestionAnswering.from_pretrained(model_name)
baseline_model = baseline_model.to(device)
baseline_model.eval()


# ### Инференс базовой модели

# In[26]:


baseline_start_preds = []
baseline_end_preds = []

# проходим по батчам валидационного датасета
for batch in tqdm(dataloader, desc="Baseline inference"):

    # переносим данные на устройство
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    # отключаем градиенты (инференс)
    with torch.no_grad():
        outputs = baseline_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    # logits начала и конца ответа
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    # обрабатываем каждый пример батча
    for i in range(start_logits.size(0)):

        # выбираем наиболее вероятные позиции
        start = torch.argmax(start_logits[i]).item()
        end = torch.argmax(end_logits[i]).item()

        baseline_start_preds.append(start)
        baseline_end_preds.append(end)


# Адаптируем токены под текст.

# In[27]:


baseline_predictions = []

# формируем текстовые ответы из токенов
for i in range(len(baseline_start_preds)):

    start = baseline_start_preds[i]
    end = baseline_end_preds[i]

    input_ids = tokenized_val[i]["input_ids"]

    # получаем токены ответа
    offsets = tokenized_val[i]["offset_mapping"]
    context = tokenized_val[i]["context"]
    start_char = offsets[start][0]
    end_char = offsets[end][1]
    pred_text = context[start_char:end_char]

    baseline_predictions.append({
        "id": str(i),
        "prediction_text": pred_text
    })


# ### Сравнение моделей

# In[28]:


# метрики baseline модели
baseline_results = squad_metric.compute(
    predictions=baseline_predictions,
    references=references
)

# метрики дообученной модели
finetuned_results = squad_metric.compute(
    predictions=predictions,
    references=references
)

print("BASELINE")
print("Exact Match:", baseline_results["exact_match"])
print("F1:", baseline_results["f1"])

print("\nFINETUNED")
print("Exact Match:", finetuned_results["exact_match"])
print("F1:", finetuned_results["f1"])


# ### Визуализация сравнения

# In[31]:


# собираем результаты в словарь для визуализации
results = {
    "pretrained": {
        "em": [baseline_results["exact_match"] / 100],  # переводим из %
        "f1": [baseline_results["f1"] / 100]
    },
    "finetuned": {
        "em": [finetuned_results["exact_match"] / 100],
        "f1": [finetuned_results["f1"] / 100]
    }
}

# сколько примеров использовали для оценки
N_EVAL = len(predictions)


# In[32]:


# создаём 2 графика: EM и F1
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

metrics = ["em", "f1"]                 # какие метрики рисуем
titles = ["Exact Match", "F1 Score"]   # названия графиков
colors = ["#3498db", "#e74c3c"]        # цвета столбцов

# строим графики
for ax, metric, title in zip(axes, metrics, titles):

    # средние значения метрик
    pre_score = np.mean(results["pretrained"][metric])
    ft_score = np.mean(results["finetuned"][metric])

    # строим столбцы
    bars = ax.bar(
        ["RoBERTa\n(pretrained)", "RoBERTa\n(fine-tuned)"],
        [pre_score, ft_score],
        color=colors
    )

    # подписываем значения над столбцами
    for bar, score in zip(bars, [pre_score, ft_score]):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.02,
            f"{score:.3f}",
            ha="center",
            fontsize=14,
            fontweight="bold"
        )

    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_ylim(0, 1.1)

# общий заголовок
plt.suptitle(f"Сравнение моделей на {N_EVAL} примерах", fontsize=14, fontweight="bold")

plt.tight_layout()
plt.show()


# После стандартного дообучения получили качество **лучше**, чем на базовой модели. Для улучшения результата подберем гиперпараметры для дообучения.

# <a id=1-7></a>
# ## 1.7 Выбор гиперпараметров для подбора
# 
# Для улучшения качества модели будет выполнен подбор трёх ключевых гиперпараметров:
# 
# * **learning_rate** — шаг обновления весов модели (`1e-5 – 5e-5`), влияет на скорость и стабильность обучения  
# * **num_train_epochs** — количество эпох обучения (`2 – 4`), позволяет избежать недообучения или переобучения  
# * **weight_decay** — коэффициент регуляризации (`0.0 – 0.1`), уменьшает переобучение и улучшает обобщающую способность модели

# <a id=1-8></a>
# ## 1.8 Подбор гиперпараметров

# ### Подготовка данных для тюнинга гиперпараметров
# 
# Для ускорения подбора гиперпараметров используется уменьшенная обучающая выборка. После выбора оптимальных параметров модель будет обучаться на полном наборе данных.

# In[33]:


hp_train = tokenized_train.select(range(6000))
hp_val = tokenized_val.select(range(750))


# ### Функция инициализации модели

# In[34]:


def model_init():
    return AutoModelForQuestionAnswering.from_pretrained(model_name)


# ### Создаем новый Trainer

# In[35]:


tuning_training_args = TrainingArguments(
    output_dir="./tuning_hyperparams_qa_bert_finetuned",
    eval_strategy="steps",      # оценка по шагам
    eval_steps=200,
    save_strategy="no",
    logging_steps=50,

    max_steps=500,              # ограничиваем обучение
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    fp16=True,
    report_to="none"
)

tuning_trainer = Trainer(
    model_init=model_init,
    args=tuning_training_args,
    train_dataset=hp_train,
    eval_dataset=hp_val,
    processing_class=tokenizer,
    data_collator=default_data_collator
)


# ### Пространство гиперпараметров
# Определяем диапазоны значений, из которых будут случайно выбираться параметры.

# In[36]:


def hp_space(trial):
    return {
        # learning rate выбирается из диапазона 1e-5 – 5e-5
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        # количество эпох
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 4),
        # коэффициент регуляризации
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1)
    }


# ### Запуск random search
# 
# Trainer выполняет несколько запусков обучения с разными гиперпараметрами.

# In[37]:


best_run = tuning_trainer.hyperparameter_search(
    direction="maximize",   # максимизируем метрику
    hp_space=hp_space,      # пространство гиперпараметров
    n_trials=10             # число случайных комбинаций
)

print(best_run)


# ### Применение лучших гиперпараметров
# 
# После поиска используем найденные параметры для финального обучения модели.

# In[38]:


# Загружаем модель
model2fn = AutoModelForQuestionAnswering.from_pretrained(model_name)
model2fn = model2fn.to(device)
model2fn.eval()


# In[39]:


# Конфигурация обучения
total_training_args = TrainingArguments(
    output_dir="./total_qa_bert_finetuned", # папка для чекпоинтов
    eval_strategy="epoch",                  # оценка после каждой эпохи
    learning_rate=3e-5,                     # learning rate (типичный для BERT)
    per_device_train_batch_size=8,          # batch size
    per_device_eval_batch_size=8,
    num_train_epochs=4,                     # число эпох
    weight_decay=0.01,                      # регуляризация
    warmup_steps=100,                       # warmup первые 100 шагов
    logging_steps=500,                      # как часто писать логи
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",                       # отключаем wandb
    fp16=True # Ускорение
)

# класс обучения
total_trainer = Trainer(
    model=model2fn,
    args=total_training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    processing_class=tokenizer,
    data_collator=default_data_collator,
)


# In[40]:


# применяем лучшие гиперпараметры
for n, v in best_run.hyperparameters.items():
    setattr(total_trainer.args, n, v)

# финальное обучение модели
total_trainer.train()


# <a id=1-9></a>
# ## 1.9 Метрики после подбора

# In[41]:


# Переводим модель в режим инфреренса
model_total = total_trainer.model
model_total.eval()
model_total = model_total.to(device)


# ### Инференс итоговой модели

# In[42]:


# Данные будем передавать батчами
batch_size = 32
tokenized_val.set_format(type="torch")
dataloader = DataLoader(tokenized_val, batch_size=batch_size, shuffle=False)

# ограничения для поиска ответа
max_answer_length = 30   # максимальная длина ответа
top_k = 20               # сколько лучших start/end токенов рассматривать

# сюда будем сохранять позиции предсказанных ответов
all_start_preds = []
all_end_preds = []

# проходим по батчам
for batch in tqdm(dataloader, desc="Inference"):

    # переносим данные на устройство
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    # инференс модели
    with torch.no_grad():
        outputs = model_total(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    # logits начала и конца ответа
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    current_batch_size = start_logits.size(0)

    # обрабатываем каждый пример в батче
    for i in range(current_batch_size):

        start_log = start_logits[i]
        end_log = end_logits[i]

        # берём top-k наиболее вероятных start и end позиций
        start_top = torch.topk(start_log, top_k).indices
        end_top = torch.topk(end_log, top_k).indices

        best_score = -1e9
        best_start = 0
        best_end = 0

        # ищем лучший span среди top-k кандидатов
        for start_idx in start_top:
            for end_idx in end_top:

                # конец ответа не может быть раньше начала
                if end_idx < start_idx:
                    continue

                # ограничиваем максимальную длину ответа
                if end_idx - start_idx > max_answer_length:
                    continue

                # score span = сумма вероятностей start и end
                score = start_log[start_idx] + end_log[end_idx]

                # если нашли более вероятный span — сохраняем
                if score > best_score:
                    best_score = score
                    best_start = start_idx
                    best_end = end_idx

        # сохраняем позиции лучшего span
        all_start_preds.append(best_start)
        all_end_preds.append(best_end)


# In[43]:


# Списки для предсказаний и эталонных ответов
predictions = []
references = []

# Проходим по всем примерам
for i in range(len(all_start_preds)):
    # Предсказанные позиции начала и конца ответа
    start = all_start_preds[i]
    end = all_end_preds[i]
    # Токены текущего примера
    input_ids = tokenized_val[i]["input_ids"]

    # конец не может быть раньше начала
    if end < start:
        end = start
    # Токены предсказанного ответа
    offsets = tokenized_val[i]["offset_mapping"]
    context = tokenized_val[i]["context"]
    start_char = offsets[start][0]
    end_char = offsets[end][1]
    pred_text = context[start_char:end_char]
    # Добавляем предсказание
    predictions.append({
        "id": str(i),
        "prediction_text": pred_text
    })
    # Истинные позиции ответа
    true_start = tokenized_val[i]["start_positions"]
    true_end = tokenized_val[i]["end_positions"]
    # Токены правильного ответа
    offsets = tokenized_val[i]["offset_mapping"]
    context = tokenized_val[i]["context"]
    start_char = offsets[true_start][0]
    end_char = offsets[true_end][1]
    true_text = context[start_char:end_char]
    # Добавляем эталонный ответ
    references.append({
        "id": str(i),
        "answers": {
            "text": [true_text],
            "answer_start": [true_start.item()]
        }
    })


# ### Сравнение моделей

# In[44]:


# метрики baseline модели
baseline_results = squad_metric.compute(
    predictions=baseline_predictions,
    references=references
)

# метрики итоговой дообученной модели
total_results = squad_metric.compute(
    predictions=predictions,
    references=references
)

print("BASELINE")
print("Exact Match:", baseline_results["exact_match"])
print("F1:", baseline_results["f1"])

print("\nTOTAL FINETUNED")
print("Exact Match:", total_results["exact_match"])
print("F1:", total_results["f1"])


# ### Визуализация сравнения

# In[46]:


# собираем результаты в словарь для визуализации
results = {
    "pretrained": {
        "em": [baseline_results["exact_match"] / 100],  # переводим из %
        "f1": [baseline_results["f1"] / 100]
    },
    "finetuned": {
        "em": [total_results["exact_match"] / 100],
        "f1": [total_results["f1"] / 100]
    }
}

# сколько примеров использовали для оценки
N_EVAL = len(predictions)


# In[47]:


# создаём 2 графика: EM и F1
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

metrics = ["em", "f1"]                 # какие метрики рисуем
titles = ["Exact Match", "F1 Score"]   # названия графиков
colors = ["#3498db", "#e74c3c"]        # цвета столбцов

# строим графики
for ax, metric, title in zip(axes, metrics, titles):

    # средние значения метрик
    pre_score = np.mean(results["pretrained"][metric])
    ft_score = np.mean(results["finetuned"][metric])

    # строим столбцы
    bars = ax.bar(
        ["RoBERTa\n(pretrained)", "RoBERTa\n(total fine-tuned)"],
        [pre_score, ft_score],
        color=colors
    )

    # подписываем значения над столбцами
    for bar, score in zip(bars, [pre_score, ft_score]):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.02,
            f"{score:.3f}",
            ha="center",
            fontsize=14,
            fontweight="bold"
        )

    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_ylim(0, 1.1)

# общий заголовок
plt.suptitle(f"Сравнение моделей на {N_EVAL} примерах", fontsize=14, fontweight="bold")

plt.tight_layout()
plt.show()


# Как видно из графиков, после подбора гиперпараметров **качество улучшилось**!

# <a id=1-10></a>
# ## 1.10 Сохранение модели

# Сохраним модель и токенизатор в директорию `best_roberta`.

# In[48]:


model_total.save_pretrained("best_roberta")
tokenizer.save_pretrained("best_roberta")


# <a id=1-11></a>
# ## 1.11 Результаты работы модели
# 
# Мы разработали **лучшую** на наших данных модель. Рассмотрим на конкретных примерах работу модели.

# In[51]:


for i in np.random.randint(len(predictions), size=12):
    text_pred, text_true = predictions[i]['prediction_text'], references[i]['answers']['text']

    q = tokenizer.decode(tokenized_val[i]).split('</s></s>')[0][3:]
    if text_true != '' and text_pred != '':
        print('Вопрос:', q)
        print('Правильный ответ:', text_true[0])
        print('Ответ модели:', text_pred)
        print()


# ###
