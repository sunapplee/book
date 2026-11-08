#!/usr/bin/env python
# coding: utf-8

# # Классификация изображений с использованием PyTorch
# 
# В данном ноутбуке реализована задача **классификации изображений** с применением методов глубокого обучения.  
# Проводится предобработка изображений, включая изменение размера и нормализацию гистограммы, а также аугментация данных для увеличения обучающей выборки.
# 
# Для обучения используются предобученные архитектуры **ResNet50, EfficientNet-B0 и ConvNext-Small**, которые дообучаются под заданное число классов.  
# Качество моделей оценивается на тестовой выборке с использованием метрик **Accuracy** и **F1-score**.

# ## Содержание
# 
# * [Импорт библиотек](#0)
# * [1. Загрузка данных](#1)
# * [2. Предобработка изображений](#2)
# * [3. Аугментация данных](#3)
# * [4 Классификация изображений](#4)
#     * [4.1 ResNet](#4-1)
#     * [4.2 EfficientNet](#4-2)
#     * [4.3 ConvNext](#4-3)
# * [5. Вывод](#5)

# ###

# <a id=0></a>
# ## Импорт библиотек

# In[1]:


import pandas as pd
import numpy as np

import cv2
import PIL
import albumentations as A
from torchvision.models import resnet50, efficientnet_b0, convnext_small, ResNet50_Weights
from torchvision.datasets import ImageFolder
import torchvision.transforms.v2 as tfs
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torch import nn
from torch import optim
from sklearn.metrics import accuracy_score, f1_score

import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm
import os


# ###

# <a id=1></a>
# # 1. Загрузка данных

# In[2]:


# Путь к файлам
images_path = 'images_clf/train'
images = {}

# Проходимся по каждому классу
for target_path in Path(images_path).iterdir():
    target = target_path.name
    # Проходимся по каждой картинке
    for image_path in tqdm(Path(target_path).iterdir(), desc=f'Класс {target}'):
        # Сохраняют картинки
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        images.setdefault(target, []).append(img)


# <a id=2></a>
# # 2. Предобработка изображений

# In[3]:


# Сюда будем сохранять обработанные изображения
preprocessed_images = {}


# ## Изменение размера
# 
# Приведем изображения к одному размеру `224x224`.

# In[4]:


# Проходимся по каждому классу
for target in images:
    # Проходимся по каждой картинке
    for img in tqdm(images[str(target)], desc=f'Класс {target}'):
        # Меняем размер
        img = cv2.resize(img, (224, 224))
        preprocessed_images.setdefault(target, []).append(img)


# ## Нормализация гистограмм
# 
# Нормализация гистограммы используется для `улучшения контраста изображения`.
# Метод перераспределяет значения яркости пикселей таким образом,
# чтобы использовать весь диапазон интенсивностей.

# Вот пример `обычного` и `нормализованного` изображения:

# In[5]:


equalized = cv2.equalizeHist(images['0'][0])

fig, axs = plt.subplots(1, 2, figsize=(10, 16))

axs[0].imshow(images['0'][0], cmap='gray')
axs[0].set_title('Оригинал')
axs[0].set_xticks([])
axs[0].set_yticks([])

axs[1].imshow(equalized, cmap='gray')
axs[1].set_title('Нормализация гистограмм')
axs[1].set_xticks([])
axs[1].set_yticks([]);


# Как видим, изображение стало **более четким**. Отнормируем гистограммы для всего датасета.

# In[6]:


# Проходимся по каждому классу
for target in preprocessed_images:
    # Проходимся по каждой картинке
    for i, img in tqdm(enumerate(preprocessed_images[str(target)]), desc=f'Класс {target}'):
        # Нормируем гистограмму
        equalized = cv2.equalizeHist(img)
        preprocessed_images[str(target)][i] = equalized


# ### ДО/ПОСЛЕ нормализации
# 
# Рассмотрим оригинальные изображения и обработанные.

# In[7]:


fig, axs = plt.subplots(6, 4, figsize=(15, 15))
axs = axs.reshape((12, 2))

for target, ax in zip(images, axs):
    for i, img in enumerate(images[target]):
        pr_img = preprocessed_images[target][i]
        ax[0].imshow(img, cmap='gray')
        ax[0].set_title(f'Оригинал класса {target}')
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[1].imshow(pr_img, cmap='gray')
        ax[1].set_title(f'Обработка класса {target}')
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        # plt.show()

        break


# После преобработки мы получили **более качественные** для классификации изображения!

# <a id=3></a>
# # 3. Аугментация данных
# 
# Для увеличения обучющей выборки проведем **аугментацию данных** c помощью библиотеки `albumentations`.

# Напишем функцию, которая на вход будет получать **изображения**, а выдавать его аугментированные вариации.
# 
# `Список аугментаций`:
# * Зеркальное отражение по горизонтали
# * Поворот
# * Случайное Приближение/Отдаление
# * Смена яркости
# * Смена контрастности
# * Случайные обрезки
# * Случайный наклон
# * Случайное вращение

# In[8]:


def augment_image(image: np.array) -> dict:

    # Зеркальное отражение по горизонтали
    transform = A.HorizontalFlip(p=1.0)
    flipped = transform(image=image)['image']

    # Поворот
    transform = A.RandomRotate90(p=1.0)
    turned = transform(image=image)['image']

    # Случайное Приближение/Отдаление
    transform = A.Affine(
        scale=(0.3, 2),
        p=1.0
    )
    zoomed = transform(image=image)['image']

    # Смена яркости
    transform = A.RandomBrightnessContrast(
        brightness_limit=(-0.5, 0.5),
        p=1.0
    )
    brighted = transform(image=image)['image']

    # Смена контрастности
    transform = A.RandomBrightnessContrast(
        contrast_limit=(-0.5, 0.5),
        p=1.0
    )
    contrasted = transform(image=image)['image']

    # Случайные обрезки
    transform = A.RandomResizedCrop(
        size=image.shape,
        scale=(0.008, 1.0),
        p=1.0
    )
    cropped = transform(image=image)['image']

    # Случайный наклон
    transform = A.Affine(
        shear=(-30, 30),
        p=1.0
    )
    sheared = transform(image=image)['image']

    # Случайное вращение
    transform = A.Rotate(
        limit=(-30, 30),
        p=1.0
    )
    rotated = transform(image=image)['image']

    return {
            'original': image,
            'mirroring': flipped,
            'turning': turned,
            'zooming': zoomed,
            'brightning': brighted,
            'contrasting': contrasted,
            'cropping': cropped,
            'shearing': sheared,
            'rotating': rotated
        }


# ### Визуализация аугментаций

# In[9]:


# Проводим аугментацию
sample = preprocessed_images['3'][0]
sample_augs = augment_image(sample)


# In[10]:


fig, axs = plt.subplots(3, 3, figsize=(9, 9))

for ax, (name, value) in zip(axs.ravel(), sample_augs.items()):

    ax.imshow(value, cmap='gray')
    ax.set_title(name)

    ax.set_xticks([])
    ax.set_yticks([])


# Все аугментации отработали успешно!

# ### Подготовка датасета
# 
# Сохраним в директории `preprocessed_images_clf` предобработанные изображения и по 10 аугментаций для каждого кластера. Итоговый датасет **будет полностью готов для дальнейшего обучения моделей классификации**.

# In[11]:


# Создаем директорию
os.makedirs('preprocessed_images_clf', exist_ok=True)
# Ограничиваем количество аугментаций на класс
AUG_COUNT = 10


# Проходимся по каждому классу
for target in preprocessed_images:
    os.makedirs(f'preprocessed_images_clf/{target}', exist_ok=True)
    # Проходимся по каждой картинке
    for i, img in tqdm(enumerate(preprocessed_images[str(target)]), desc=f'Класс {target}'):

        # Аугментация
        if i < AUG_COUNT:
            # Проводим аугментацию и сохраняем все изображения
            aug_dict = augment_image(img)
            for aug_type in aug_dict:
                cv2.imwrite(f'preprocessed_images_clf/{target}/{i}_{aug_type}.png', aug_dict[aug_type])
        # Остальные изображения без аугментации
        else:
            cv2.imwrite(f'preprocessed_images_clf/{target}/{i}_original.png', img)


# <a id=4></a>
# # 4. Классификация изображений

# ### Создание `Dataset`
# 
# Используем модуль `ImageFolder` для создания представления наших данных

# In[12]:


transform = tfs.Compose([
    tfs.Resize((224, 224)),
    tfs.Grayscale(num_output_channels=3),
    tfs.ToImage(),
    tfs.ToDtype(torch.float32, scale=True),
    tfs.Normalize(
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225)
    )
])

dataset = ImageFolder('preprocessed_images_clf/',
                     transform=transform)
num_classes = len(dataset.classes)


# ### Разделение выборки
# 
# Разделим данные на три части: **70% / 15% / 15%**.
# 
# * **70%** — обучающая выборка. Используется непосредственно для дообучения модели и изменения её весов.
# * **15%** — валидационная выборка. Примеры из этого набора не участвуют в обучении, но используются для контроля процесса: по валидационному *loss* и метрикам сохраняется лучшая модель.
# * **15%** — тестовая выборка. Полностью откладывается до конца эксперимента и используется только для финальной, объективной оценки качества модели на данных, которые она никогда не видела.
# 
# Такое разбиение позволяет одновременно эффективно обучить модель, своевременно отслеживать переобучение и провести честную итоговую проверку качества.

# In[13]:


# Определим размеры выборок
train_size = int(0.7 * len(dataset))
val_size = (len(dataset) - train_size) // 2
test_size = len(dataset) - train_size - val_size


# Разделяем датасет
train_subset, val_subset, test_subset = random_split(dataset, [train_size, val_size, test_size])


# ### Создание `DataLoader`

# In[14]:


train_dataloader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=4)
test_dataloader = DataLoader(test_subset, batch_size=32, shuffle=False, num_workers=4)


# ### Функция обучения моделей

# In[15]:


def train_model(model, train_loader, val_loader, best_model_path, epochs=3, lr=1e-5):
    # Переводим модель на gpu
    model = model.cuda()
    # Конфигурация
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                         step_size=5,
                                         gamma=0.1)

    best_val_loss = float("inf")   # для сохранения лучшей модели
    for epoch in range(epochs):
        # Режим обучения
        model.train()
        epoch_loss = 0

        for imgs, labels in tqdm(train_loader, desc='Обучение', leave=True, ncols=100):
            # перенос батча на GPU
            imgs, labels = imgs.cuda(), labels.cuda()

            # Проход данных через модель
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            # Обновляем метрику
            epoch_loss += loss.item()

        # Уменьшаем lr
        scheduler.step()

        train_loss = epoch_loss / len(train_loader)
        print(f"\nEpoch {epoch+1}: Train loss = {train_loss:.4f}")

        # Валидация
        val_loss, acc, f1 = evaluate(model, val_loader, criterion)
        print(f"Validation: Loss={val_loss:.4f} | Acc={acc:.4f} | F1={f1:.4f}")

        # Сохраняем лучшую модель
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Лучшая модель сохранена - {best_model_path}")

    return model


# ### Функция подсчета метрик

# In[16]:


def evaluate(model, val_loader, criterion):

    # Режим инференса
    model.eval()
    total_loss = 0

    all_labels = []
    all_preds = []

    # Инференс модели
    with torch.no_grad():
        # По каждому батчу
        for imgs, labels in tqdm(val_loader, desc='Валидация'):
            # Переводим на cuda
            imgs, labels = imgs.cuda(), labels.cuda()

            # Прогоняем через модель
            logits = model(imgs)
            # Сохранение лосса
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # Предсказанные классы
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = total_loss / len(val_loader)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    return val_loss, acc, f1


# <a id=4-1></a>
# ## 4.1 ResNet

# ### Инициализация модели

# In[17]:


model_resnet50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model_resnet50.fc = torch.nn.Linear(model_resnet50.fc.in_features, num_classes)


# ### Обучение модели

# In[18]:


trained_model_resnet50 = train_model(model_resnet50, train_dataloader, val_dataloader, best_model_path="models/best_model_clf_torch_resnet50.bin", epochs=10)


# ### Подсчет метрик
# 
# Загрузим лучшую обученную модель и посмотрим на метрику на тестовом датасете.

# In[19]:


best_model_resnet50 = resnet50(weights=None)
best_model_resnet50.fc = torch.nn.Linear(model_resnet50.fc.in_features, num_classes)
best_model_resnet50.load_state_dict(torch.load('models/best_model_clf_torch_resnet50.bin'))
best_model_resnet50 = best_model_resnet50.cuda()

criterion = nn.CrossEntropyLoss()

val_loss, acc, f1 = evaluate(best_model_resnet50, test_dataloader, criterion)

print(f'Лосс для ResNet50: {val_loss:.3f}')
print(f'Accuracy для ResNet50: {acc:.3f}')
print(f'F1 для ResNet50: {f1:.3f}')


# ### Визуализация предсказаний

# In[20]:


# Для денормализации
mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

# Отображаем 9 картинок
fig, axs = plt.subplots(3, 3, figsize=(9, 9))
for ax, i in zip(axs.ravel(), np.random.randint(1, len(test_subset), size=9)):

    img, label = test_subset[i]

    # вход в модель
    img_input = img.unsqueeze(0).cuda()
    with torch.no_grad():
        logit = best_model_resnet50(img_input)
        pred = torch.argmax(logit).item()

    # денормализация
    img_show = img.cpu() * std + mean
    img_show = img_show.permute(1, 2, 0).numpy()

    ax.imshow(img_show)
    ax.set_title(f"pred={pred} true={label}")
    ax.axis("off")


# <a id=4-2></a>
# ## 4.2 EfficientNet

# ### Инициализация модели

# In[21]:


model_en_b0 = efficientnet_b0(weights="IMAGENET1K_V1")
model_en_b0.classifier[1] = nn.Linear(1280, num_classes)


# ### Обучение модели

# In[22]:


trained_model_en_b0 = train_model(model_en_b0, train_dataloader, val_dataloader, best_model_path="models/best_model_clf_torch_en_b0.bin", epochs=10)


# ### Подсчет метрик
# 
# Загрузим лучшую обученную модель и посмотрим на метрику на тестовом датасете.

# In[23]:


best_model_en_b0 = efficientnet_b0(weights=None)
best_model_en_b0.classifier[1] = nn.Linear(1280, num_classes)
best_model_en_b0.load_state_dict(torch.load('models/best_model_clf_torch_en_b0.bin'))
best_model_en_b0 = best_model_en_b0.cuda()

criterion = nn.CrossEntropyLoss()

val_loss, acc, f1 = evaluate(best_model_en_b0, test_dataloader, criterion)

print(f'Лосс для EfficientNet: {val_loss:.3f}')
print(f'Accuracy для EfficientNet: {acc:.3f}')
print(f'F1 для EfficientNet: {f1:.3f}')


# ### Визуализация предсказаний

# In[24]:


# Для денормализации
mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

# Отображаем 9 картинок
fig, axs = plt.subplots(3, 3, figsize=(9, 9))
for ax, i in zip(axs.ravel(), np.random.randint(1, len(test_subset), size=9)):

    img, label = test_subset[i]

    # вход в модель
    img_input = img.unsqueeze(0).cuda()
    with torch.no_grad():
        logit = best_model_en_b0(img_input)
        pred = torch.argmax(logit).item()

    # денормализация
    img_show = img.cpu() * std + mean
    img_show = img_show.permute(1, 2, 0).numpy()

    ax.imshow(img_show)
    ax.set_title(f"pred={pred} true={label}")
    ax.axis("off")


# <a id=4-3></a>
# ## 4.3 ConvNext

# ### Инициализация модели

# In[25]:


model_convnext = convnext_small(weights="IMAGENET1K_V1")
model_convnext.classifier[2] = nn.Linear(768, num_classes)


# ### Обучение модели

# In[26]:


trained_model_convnext = train_model(model_convnext, train_dataloader, val_dataloader, best_model_path="models/best_model_clf_torch_convnext.bin", epochs=10)


# ### Подсчет метрик
# 
# Загрузим лучшую обученную модель и посмотрим на метрику на тестовом датасете.

# In[27]:


best_model_convnext = convnext_small(weights=None)
best_model_convnext.classifier[2] = nn.Linear(768, num_classes)
best_model_convnext.load_state_dict(torch.load('models/best_model_clf_torch_convnext.bin'))
best_model_convnext = best_model_convnext.cuda()

criterion = nn.CrossEntropyLoss()

val_loss, acc, f1 = evaluate(best_model_convnext, test_dataloader, criterion)

print(f'Лосс для ConvNext: {val_loss:.3f}')
print(f'Accuracy для ConvNext: {acc:.3f}')
print(f'F1 для ConvNext: {f1:.3f}')


# ### Визуализация предсказаний

# In[28]:


# Для денормализации
mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

# Отображаем 9 картинок
fig, axs = plt.subplots(3, 3, figsize=(9, 9))
for ax, i in zip(axs.ravel(), np.random.randint(1, len(test_subset), size=9)):

    img, label = test_subset[i]

    # вход в модель
    img_input = img.unsqueeze(0).cuda()
    with torch.no_grad():
        logit = best_model_convnext(img_input)
        pred = torch.argmax(logit).item()

    # денормализация
    img_show = img.cpu() * std + mean
    img_show = img_show.permute(1, 2, 0).numpy()

    ax.imshow(img_show)
    ax.set_title(f"pred={pred} true={label}")
    ax.axis("off")


# <a id=5></a>
# # 5. Вывод
# В ходе работы была реализована задача **классификации изображений** с использованием фреймворка **PyTorch**. Был проведён полный цикл подготовки данных: загрузка изображений, изменение размера до `224×224`, нормализация гистограмм для улучшения контраста и аугментация данных для увеличения обучающей выборки. 
# 
# Для решения задачи были дообучены несколько предобученных моделей: **ResNet50, EfficientNet-B0 и ConvNext-Small**. Качество моделей оценивалось на тестовой выборке с использованием метрик **Accuracy** и **F1-score**. 
# 
# По результатам экспериментов наилучшее качество показала модель **ConvNext-Small**, достигнув примерно **Accuracy ≈ 0.66** и **F1 ≈ 0.59**, что значительно выше результатов **ResNet50** и **EfficientNet-B0**. 
# 
# Таким образом, использование более современной архитектуры позволило получить **наиболее точную модель классификации**, пригодную для дальнейшего применения или дообучения.
# 
