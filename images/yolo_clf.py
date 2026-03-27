#!/usr/bin/env python
# coding: utf-8

# # Классификация изображений с использованием YOLO
# 
# В данном ноутбуке проводится обучение модели **YOLO для задачи классификации изображений**.  
# Данные предварительно подготовлены и организованы по классам, после чего разделены на **обучающую и валидационную выборки**.
# 
# Цель работы — обучить модель классификации изображений и оценить её качество на отложенных данных.

# ## Содержание
# 
# * [Импорт библиотек](#0)
# * [1. Загрузка данных](#1)
# * [2. Классификация изображений](#2)
# * [3. Сохранение модели](#3)
# * [4. Метрики модели](#4)
# * [5. Визуализация классификации](#5)
# * [6. Вывод](#6)

# ###

# <a id=0></a>
# ## Импорт библиотек

# In[1]:


from ultralytics import YOLO
from ultralytics.data.split import split_classify_dataset

from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score

import matplotlib.pyplot as plt
import random
from pathlib import Path
from tqdm import tqdm


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# ###

# <a id=1></a>
# # 1. Загрузка данных
# 
# Данные прeдставлены в директории `preprocessed_images_clf` в формате:
# ```
# preprocessed_images_clf/
# |-- 0/
#     |-- img0.png
# |-- 1/
#     |-- img1.png
# |-- 2/
#     |-- img2.png
# |-- 3/
#     |-- img3.png
# |-- 4/
#     |-- img4.png
# |-- 5
#     |-- img5.png
# ...
# 
# 
# ```
# 
# Однако, `ultralytics` требует разделения данных на обучющую и тестовую выборку.

# In[3]:


data_path = 'preprocessed_images_clf/'


# ### Разделение данных
# 
# Разделим данные на две части: **80% / 20%**.
# 
# * **80%** — обучающая выборка. Используется непосредственно для дообучения модели и изменения её весов.
# * **20%** — валидационная выборка. Примеры из этого набора не участвуют в обучении, но используются для контроля процесса: по валидационному *loss* и метрикам сохраняется лучшая модель.
# 
# Такое разбиение позволяет одновременно эффективно обучить модель и провести честную итоговую проверку качества.

# In[4]:


# Используем готовую функцию для разбиения датасета
split_data_path = split_classify_dataset(source_dir=data_path, train_ratio=0.8)


# Итоговая структура для обучения:
# 
# ```
# preprocessed_images_clf_split/
# |-- train/
#     |-- 0/
#         |-- img0.png
#     |-- 1/
#         |-- img1.png
# 
# |-- val/
#     |-- 0/
#         |-- img0.png
#     |-- 1/
#         |-- img1.png
# 
# 
# ```

# <a id=2></a>
# # 2. Классификация изображений
# 
# Для классификации изображений возьмем модель `yolo26n-cls` из библиотеки `ultralytics`. Данная модель имеет **1.5 МЛН** параметров.

# In[5]:


# Инициализируем модель
model = YOLO('yolo26n-cls.pt')


# In[6]:


# Запускаем обучение
results = model.train(data=split_data_path, # Данные для обучения и валидации
                      epochs=30, # Количество эпох
                      augment=False, # Отключаем аугментации
                      project=".",
                      name="yolo_classifier"
)


# Модель обучилась с хорошим `loss`! Оценим качество модели с помощью известных метрик.

# <a id=3></a>
# # 3. Сохранение модели
# 
# После каждой эпохи `ultralytics` сохраняет веса. Нам нужна лучшая модель за все обучение. Веса для нее хранятся в `runs\classify\yolo_classifier3\weights\best.pt` .

# <a id=4></a>
# # 4. Метрики модели
# 
# Оценим качество моделей на валидационной выборке с использованием метрик **Accuracy** и **F1-score**.

# ### Загрузим лучшую модель.

# In[7]:


model = YOLO("E:/Heckfy/atom/REA/preparing_total/runs/classify/yolo_classifier3/weights/best.pt")


# ### Инференс модели

# In[8]:


# Валидационные файлы
files = sorted(Path("preprocessed_images_clf_split/val").rglob("*.png"))
# Прогоняем через модель
results = model.predict(files, verbose=False)


# In[9]:


# Получаем предсказанный класс
model_names = model.names
y_pred = [int(model_names[r.probs.top1]) for r in results]
# Получаем реальный класс
y_true = [int(f.parent.name) for f in files]


# ### Метрики

# In[10]:


print(f'Accuracy для YOLO: {accuracy_score(y_true, y_pred):.3f}')
print(f"F1 для YOLO: {f1_score(y_true, y_pred, average='macro'):.3f}")


# <a id=5></a>
# # 5. Визуализация классификации
# 
# Отобразим случайные сэмплы и сравним предсказанный класс с настоящим.

# In[11]:


fig, axs = plt.subplots(3, 3, figsize=(12, 12))
axs = axs.ravel()

# 9 случаных сэмплов
random_samples = random.sample(tuple(zip(results, y_true)), k=9)

# Проходимся по каждому сэмплу
for i, (res, y_true_sample) in enumerate(random_samples):
    # Получаем предсказания
    y_pred_sample = int(model_names[res.probs.top1])

    # Отрисовываем график
    axs[i].imshow(res.orig_img)
    axs[i].set_title(f'Предсказанный класс: {y_pred_sample}\nИстинный класс: {y_true_sample}')

    # Убираем оси
    axs[i].set_xticks([])
    axs[i].set_yticks([])


# <a id=6></a>
# # 6. Вывод
# 
# В ходе работы была обучена модель **YOLO (`yolo26n-cls`)** для задачи классификации изображений. Данные были разделены на обучающую и валидационную выборки, после чего выполнено дообучение предобученной модели.
# 
# По результатам оценки на валидационной выборке модель показала следующие метрики:
# 
# - **Accuracy:** 0.785  
# - **F1-score:** 0.780  
# 
# Полученные значения метрик свидетельствуют о **хорошем качестве классификации** и сбалансированной работе модели по различным классам. Модель корректно обобщает данные и может использоваться для дальнейших экспериментов или интеграции в систему распознавания изображений.
