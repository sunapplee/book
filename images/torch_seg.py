#!/usr/bin/env python
# coding: utf-8

# # DIGITAL SKILLS 2023
# ## C2. Модуль C. Построение, обучение и оптимизация модели

# #####

# # Содержание
# 
# * [Импорт библиотек](#0)
# * [1. Сегментация изображений «с учителем»](#1)
#     * [1.1 Подготовка данных](#1-1)
#     * [1.2 Выбор моделей сегментации](#1-2)
#     * [1.3 Обучение моделей](#1-3)
#     * [1.4 Визуализация результатов обучения](#1-4)
#     * [1.5 Расчет метрик](#1-5)
#     * [1.6 Визуализация предсказаний](#1-6)
# * [2. Сегментация набора из Модуля С](#2)

# ###

# <a id=0></a>
# ## Импорт библиотек

# In[282]:


from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import io
import segmentation_models_pytorch as smp
import torch
from torch import optim, nn
import torchvision.transforms.functional as F
import segmentation_models_pytorch.metrics.functional as metrics
from torchvision.models import segmentation

import numpy as np
import pandas as pd

import cv2

from ultralytics.data.converter import convert_segment_masks_to_yolo_seg

import os
from pathlib import Path
from tqdm import tqdm

import plotly.express as px
import matplotlib.pyplot as plt


# ###

# <a id=1></a>
# # 1. Сегментация изображений «с учителем»

# <a id=1-1></a>
# ## 1.1 Подготовка данных

# Обучать модели будем с помощью библиотеки torch. Для работы с ним напишем класс ```FieldDataset```, чтобы удобно загрузить все изображения с масками, которые получили в прошлом модуле.

# In[310]:


class FieldDataset(Dataset):
    def __init__(self, transform=None,
                images_path=r'E:\Heckfy\atom\Atomskills2026\15.02.2026\homework\preprocess_images\images',
                labels_path=r'E:\Heckfy\atom\Atomskills2026\15.02.2026\homework\preprocess_images\labels'):
        self.images_path = images_path
        self.labels_path = labels_path
        self.transform = transform
        self.samples = []

        # Проходимся по каждому файлу в каждой папке
        for image in Path(self.images_path).iterdir():
            for image_path in Path(image).iterdir():
                # Пропускаем директории
                if image_path.is_dir():
                    continue
                # Сохраняем пути
                label_path = os.path.join(self.labels_path, image.name, f'{image_path.stem}_mask.jpg')
                self.samples.append((
                    image_path,
                    label_path
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]

        # читаем как тензоры
        img = io.read_image(img_path) / 255.0
        img = F.resize(img, (224, 224))

        # Mаска загружается как RGB
        mask = io.read_image(label_path)
        mask = F.resize(mask, (224, 224))
        # берем 1 канал и бинаризуем его, чтобы получить одноканальную маску
        mask = mask[0:1, :, :]     # → [1,H,W]
        mask = (mask > 127).float()

        # аугментации (если заданы)
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        return img, mask


# In[311]:


dataset = FieldDataset()


# Разделим данные на **обучающую, валидационную и тестовую** выборки.
# 
# - **Обучающая (70%)** — для качественного обучения модели. Этого объема достаточно, чтобы модель смогла "выучить" закономерности в данных.
# - **Валидационная (15%)** — для настройки гиперпараметров и контроля переобучения.
# - **Тестовая (15%)** — для финальной объективной оценки качества модели на новых данных.

# In[24]:


# Получаем пропорции датасета
total = len(dataset)
train_size = int(0.7 * total)
val_size   = int(0.15 * total)
test_size  = total - train_size - val_size

print("Всего образцов:", total)
print("Обучающая выборка (70%):", train_size)
print("Валидационная выборка (15%):", val_size)
print("Тестовая выборка (15%):", test_size)


# In[25]:


train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])


# Создаем ```DataLoader``` для обучения модели, размер батча для обучения - ```8```.

# In[26]:


train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)


# ####

# <a id=1-2></a>
# ## 1.2 Выбор моделей сегментации

# Рассмотрим ```6``` моделей сегментации.
# 
# 1) **U-Net** — архитектура с skip connections, эффективна для восстановления пространственных деталей, особенно в биологических изображениях. Хороша тем, что точно восстанавливает границы объектов даже при ограниченном количестве обучающих данных.  
# 2) **FPN** — первая Fully Convolutional сеть для семантической сегментации; обрабатывает изображения любого размера. Проста и интерпретируема, служит основой для многих других архитектур.  
# 3) **DeepLab V3** — использует atrous convolution и ASPP для захвата многомасштабного контекста. Отлично справляется с объектами разных размеров и форм, обеспечивая высокую точность.  
# 4) **LR-ASPP** — облегчённая версия ASPP, оптимизирована для мобильных устройств. Быстрая и при этом достаточно точная, подходит для встраивания в ресурсозависимые системы.  
# 5) **YOLO26n-seg** — модель объединяет детектирование и сегментацию масок в реальном времени. Отличается высокой скоростью и применима в задачах, где важна производительность.
# 
# Для выбора модели, обучим каждую и выберем лучшую для наших спутниковых снимков.

# <a id=1-3></a>
# ## 1.3 Обучение моделей

# ### 1. U-net

# #### Инициализация модели

# In[31]:


# Загружаем модель Unet
model = smp.Unet(
    encoder_name="resnet34",        # backbone
    encoder_weights="imagenet",     # предобученные веса
    in_channels=3,                  # RGB
    classes=1,                      # бинарная сегментация
)

# Основную часть модель не дообучаем
for param in model.parameters():
    param.requires_grad = False

# Обучаем только сегментнатора
for param in model.segmentation_head.parameters():
    param.requires_grad = True

# Перевидим модель на cpu/gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# Оптимизатор Adam
optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
# Планировщик learning rate
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30)
# Loss-функция
criterion = nn.BCEWithLogitsLoss()


# #### Обучение модели

# In[32]:


# Функция обучения модели
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for imgs, masks in tqdm(loader, desc='Обучение', leave=True, ncols=100):
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


# In[33]:


# Фунция проверки качества на валидационом датасете
def val_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc='Валидация', ncols=100):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks)
            total_loss += loss.item()
    return total_loss / len(loader)


# In[34]:


# Главный цикл обучения

# Конфигурационные данные
best_val_loss = float("inf")
NUM_EPOCHS = 5
unet_losses = []

# Запуск обучения
for epoch in range(NUM_EPOCHS):
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    val_loss = val_epoch(model, val_loader, criterion)
    scheduler.step()

    # Добавляем логирование
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | train: {train_loss:.4f} | val: {val_loss:.4f}")

    unet_losses.append(val_loss)

    # сохраняем лучшую модель
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_unet.pth")
        print("  ✓ saved best model")


# #### Расчет метрик

# Выполним расчет индексов ```IoU``` и ```Dice``` для оценки полученного кластеризатора.
# 
# Метрику ```IoU``` импортируем из библиотеки ```segmentation_models_pytorch```.
# 
# Метрику ```Dice``` получим из формулы ниже:
# 
# ![image.png](attachment:38af0c2d-0e2c-4fd4-9355-f619b546a5b9.png)

# In[35]:


# Сюда сохраняем метрики
all_tp, all_fp, all_fn, all_tn = [], [], [], []

# Проходимся по тестовому набору
for imgs, masks in tqdm(test_loader, desc='Расчет метрик', ncols=100):
    imgs, masks = imgs.to(device), masks.to(device).long()
    # Инфреренс модели
    with torch.no_grad():
        output = model(imgs)
    # Вычисляем метрики
    tp, fp, fn, tn = metrics.get_stats(output, masks, mode='binary', threshold=0.5)
    # Добавляем метрики
    all_tp.append(tp)
    all_fp.append(fp)
    all_fn.append(fn)
    all_tn.append(tn)

# Собираем всё вместе
all_tp = torch.cat(all_tp)
all_fp = torch.cat(all_fp)
all_fn = torch.cat(all_fn)
all_tn = torch.cat(all_tn)

# Считаем IoU по всему датасету
iou = metrics.iou_score(all_tp, all_fp, all_fn, all_tn, reduction='micro')

# Выводим Dice
dice = (2 * iou) / (1 + iou)

print(f"IoU для Unet: {iou:.4f}")
print(f"Dice для Unet: {dice:.4f}")


# ### 2. FCN

# #### Инициализация модели

# In[36]:


# Загружаем модель FPN
model = smp.FPN(
    encoder_name="resnet50",        # backbone
    encoder_weights="imagenet",     # предобученные веса
    in_channels=3,                  # RGB
    classes=1,                      # бинарная сегментация
)

# Основную часть модель не дообучаем
for param in model.parameters():
    param.requires_grad = False

# Обучаем только сегментнатора
for param in model.segmentation_head.parameters():
    param.requires_grad = True

# Перевидим модель на cpu/gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# Оптимизатор Adam
optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
# Планировщик learning rate
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30)
# Loss-функция
criterion = nn.BCEWithLogitsLoss()


# #### Обучение модели

# In[37]:


# Функция обучения модели
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for imgs, masks in tqdm(loader, desc='Обучение', ncols=100):
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


# In[38]:


# Фунция проверки качества на валидационом датасете
def val_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc='Валидация', ncols=100):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks)
            total_loss += loss.item()
    return total_loss / len(loader)


# In[39]:


# Главный цикл обучения

# Конфигурационные данные
best_val_loss = float("inf")
NUM_EPOCHS = 5
fpn_losses = []

# Запуск обучения
for epoch in range(NUM_EPOCHS):
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    val_loss = val_epoch(model, val_loader, criterion)
    scheduler.step()

    # Добавляем логирование
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | train: {train_loss:.4f} | val: {val_loss:.4f}")

    fpn_losses.append(val_loss)

    # сохраняем лучшую модель
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_fpn.pth")
        print("  ✓ saved best model")


# #### Расчет метрик

# Выполним расчет индексов ```IoU``` и ```Dice``` для оценки полученного кластеризатора.
# 
# Метрику ```IoU``` импортируем из библиотеки ```segmentation_models_pytorch```.
# 
# Метрику ```Dice``` получим из формулы ниже:
# 
# ![image.png](attachment:38af0c2d-0e2c-4fd4-9355-f619b546a5b9.png)

# In[40]:


# Сюда сохраняем метрики
all_tp, all_fp, all_fn, all_tn = [], [], [], []

# Проходимся по тестовому набору
for imgs, masks in tqdm(test_loader, desc='Расчет метрик', ncols=100):
    imgs, masks = imgs.to(device), masks.to(device).long()
    # Инфреренс модели
    with torch.no_grad():
        output = model(imgs)
    # Вычисляем метрики
    tp, fp, fn, tn = metrics.get_stats(output, masks, mode='binary', threshold=0.5)
    # Добавляем метрики
    all_tp.append(tp)
    all_fp.append(fp)
    all_fn.append(fn)
    all_tn.append(tn)

# Собираем всё вместе
all_tp = torch.cat(all_tp)
all_fp = torch.cat(all_fp)
all_fn = torch.cat(all_fn)
all_tn = torch.cat(all_tn)

# Считаем IoU по всему датасету
iou = metrics.iou_score(all_tp, all_fp, all_fn, all_tn, reduction='micro')

# Выводим Dice
dice = (2 * iou) / (1 + iou)

print(f"IoU для FPN: {iou:.4f}")
print(f"Dice для FPN: {dice:.4f}")


# ### 3. DeepLab V3

# #### Инициализация модели

# In[46]:


# Загружаем модель DeepLabV3
model = smp.DeepLabV3(
    encoder_name="resnet34",        # backbone
    encoder_weights="imagenet",     # предобученные веса
    in_channels=3,                  # RGB
    classes=1,                      # бинарная сегментация
)

# Основную часть модель не дообучаем
for param in model.parameters():
    param.requires_grad = False

# Обучаем только сегментнатора
for param in model.segmentation_head.parameters():
    param.requires_grad = True

# Перевидим модель на cpu/gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# Оптимизатор Adam
optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
# Планировщик learning rate
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30)
# Loss-функция
criterion = nn.BCEWithLogitsLoss()


# #### Обучение модели

# In[47]:


# Функция обучения модели
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for imgs, masks in tqdm(loader, desc='Обучение', ncols=100):
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


# In[48]:


# Фунция проверки качества на валидационом датасете
def val_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc='Валидация', ncols=100):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks)
            total_loss += loss.item()
    return total_loss / len(loader)


# In[49]:


# Главный цикл обучения

# Конфигурационные данные
best_val_loss = float("inf")
NUM_EPOCHS = 5
deeplab_losses = []

# Запуск обучения
for epoch in range(NUM_EPOCHS):
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    val_loss = val_epoch(model, val_loader, criterion)
    scheduler.step()

    # Добавляем логирование
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | train: {train_loss:.4f} | val: {val_loss:.4f}")

    deeplab_losses.append(val_loss)

    # сохраняем лучшую модель
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_deeplabv3.pth")
        print("  ✓ saved best model")


# #### Расчет метрик

# Выполним расчет индексов ```IoU``` и ```Dice``` для оценки полученного кластеризатора.
# 
# Метрику ```IoU``` импортируем из библиотеки ```segmentation_models_pytorch```.
# 
# Метрику ```Dice``` получим из формулы ниже:
# 
# ![image.png](attachment:38af0c2d-0e2c-4fd4-9355-f619b546a5b9.png)

# In[50]:


# Сюда сохраняем метрики
all_tp, all_fp, all_fn, all_tn = [], [], [], []

# Проходимся по тестовому набору
for imgs, masks in tqdm(test_loader, desc='Расчет метрик', ncols=100):
    imgs, masks = imgs.to(device), masks.to(device).long()
    # Инфреренс модели
    with torch.no_grad():
        output = model(imgs)
    # Вычисляем метрики
    tp, fp, fn, tn = metrics.get_stats(output, masks, mode='binary', threshold=0.5)
    # Добавляем метрики
    all_tp.append(tp)
    all_fp.append(fp)
    all_fn.append(fn)
    all_tn.append(tn)

# Собираем всё вместе
all_tp = torch.cat(all_tp)
all_fp = torch.cat(all_fp)
all_fn = torch.cat(all_fn)
all_tn = torch.cat(all_tn)

# Считаем IoU по всему датасету
iou = metrics.iou_score(all_tp, all_fp, all_fn, all_tn, reduction='micro')

# Выводим Dice
dice = (2 * iou) / (1 + iou)

print(f"IoU для DeepLabV3: {iou:.4f}")
print(f"Dice для DeepLabV3: {dice:.4f}")


# ### 4. LR-ASPP

# #### Инициализация модели

# In[56]:


# Загружаем модель LR-ASPP
model = segmentation.lraspp_mobilenet_v3_large(
    weights="DEFAULT", # предобученные веса
    progress=True
)

# Меняем число классов
model.classifier.low_classifier  = nn.Conv2d(40, 1, kernel_size=1)
model.classifier.high_classifier = nn.Conv2d(128, 1, kernel_size=1)

# Основную часть модель не дообучаем
for param in model.parameters():
    param.requires_grad = False

# Обучаем только сегментнатора
for param in model.classifier.parameters():
    param.requires_grad = True

# Перевидим модель на cpu/gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# Оптимизатор Adam
optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
# Планировщик learning rate
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30)
# Loss-функция
criterion = nn.BCEWithLogitsLoss()


# #### Обучение модели

# In[57]:


# Функция обучения модели
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for imgs, masks in tqdm(loader, desc='Обучение', ncols=100):
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer.zero_grad()
        preds = model(imgs)["out"]
        loss = criterion(preds, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


# In[58]:


# Фунция проверки качества на валидационом датасете
def val_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc='Валидация', ncols=100):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)["out"]
            loss = criterion(preds, masks)
            total_loss += loss.item()
    return total_loss / len(loader)


# In[59]:


# Главный цикл обучения

# Конфигурационные данные
best_val_loss = float("inf")
NUM_EPOCHS = 20

lraspp_losses = []

# Запуск обучения
for epoch in range(NUM_EPOCHS):
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    val_loss = val_epoch(model, val_loader, criterion)
    scheduler.step()

    # Добавляем логирование
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | train: {train_loss:.4f} | val: {val_loss:.4f}")

    lraspp_losses.append(val_loss)

    # сохраняем лучшую модель
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_lraspp.pth")
        print("  ✓ saved best model")


# #### Расчет метрик

# In[60]:


# Сюда сохраняем метрики
all_tp, all_fp, all_fn, all_tn = [], [], [], []

# Проходимся по тестовому набору
for imgs, masks in tqdm(test_loader, desc='Расчет метрик', ncols=100):
    imgs, masks = imgs.to(device), masks.to(device).long()
    # Инфреренс модели
    with torch.no_grad():
        output = model(imgs)['out']
    # Вычисляем метрики
    tp, fp, fn, tn = metrics.get_stats(output, masks, mode='binary', threshold=0.5)
    # Добавляем метрики
    all_tp.append(tp)
    all_fp.append(fp)
    all_fn.append(fn)
    all_tn.append(tn)

# Собираем всё вместе
all_tp = torch.cat(all_tp)
all_fp = torch.cat(all_fp)
all_fn = torch.cat(all_fn)
all_tn = torch.cat(all_tn)

# Считаем IoU по всему датасету
iou = metrics.iou_score(all_tp, all_fp, all_fn, all_tn, reduction='micro')

# Выводим Dice
dice = (2 * iou) / (1 + iou)

print(f"IoU для LR-ASPP: {iou:.4f}")
print(f"Dice для LR-ASPP: {dice:.4f}")


# ### 5. YOLO26n-seg 

# #### Подготовка данных
# 
# Текущая структура изображения:
# 
# 
# preprocess_images/<br>
# |--- images/<br>
# |------ image_image_00001_20210420_const_0004/<br>
# |--------- image_image_00001_20210420_const_0004_original.jpg<br>
# |--------- image_image_00001_20210420_const_0004_brightning.jpg<br>
# |--- labels/<br>
# |------ image_image_00001_20210420_const_0004/<br>
# |--------- image_image_00001_20210420_const_0004_original_mask.jpg<br>
# |--------- image_image_00001_20210420_const_0004_brightning_mask.jpg<br>
# 
# 
# Однако, для обучения ```YOLO-модели``` такая структура не подойдет. 
# 
# Преобразуем датасет в правильный формат.

# In[307]:


# Сохраняем без вложенных папок
os.makedirs('yolo_mask/', exist_ok=True)

for mask_path in tqdm(Path(r"E:\Heckfy\atom\Atomskills2026\15.02.2026\homework\preprocess_images\labels").rglob("*.jpg"), total=17532, ncols=100):
    # Загружаем оригинальные маски
    if not mask_path.stem.endswith('original_mask'):
        continue
    # Загружаем маску
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    # Сохраняем как PNG с пикселями 0 и 1 (маска)
    binary_01 = (binary > 127).astype("uint8")  # 255 → 1
    out_path = f'yolo_mask/{str(mask_path.stem)[:-14]}.png'
    cv2.imwrite(out_path, binary_01)


# ```ultralytics``` принимает не фото маски, а ```txt-файл``` с текстовым описанием маски. С помощью метода ```convert_segment_masks_to_yolo_seg``` преобразуем маски в необходимый формат.

# In[309]:


binary_01.shape


# In[308]:


test = cv2.imread('yolo_mask/image_image_00001_20210420_const_0004.png', cv2.IMREAD_GRAYSCALE)
print(test.shape)       # (224, 224)
print(np.unique(test))  # [0, 1]


# In[289]:


convert_segment_masks_to_yolo_seg(
masks_dir="yolo_mask/",
output_dir="yolo_labels/",
classes=1
)


# #### 

# ### Итог

# Лучшую метрику показала модель ```LR-ASPP```. В дальнейшнем для инфреренса ипользуем ее.

# ####

# <a id=1-4></a>
# ## 1.4 Визуализация результатов обучения

# Мы обучали модель 15 эпох, посмотрим как менялась функция потерь со временем.

# In[65]:


px.line(x=range(1, 16), y=lraspp_losses,
       labels={
        "x": "Эпоха",
        "y": "Loss"
    }, title="График изменения Loss по эпохам",
       render_mode='notebook')


# Из вывода графика, можем сказать, что с каждой эпохой функция потерь на валидационном датасете **уменьшалась**. Скорее всего, имея больше времени, можно было бы получить результат еще лучше  

# <a id=1-5></a>
# ## 1.5 Расчет метрик

# Выполним расчет индексов ```IoU``` и ```Dice``` для тестового набора данных. 

# Метрику ```IoU``` импортируем из библиотеки ```segmentation_models_pytorch```.
# 
# Метрику ```Dice``` получим из формулы ниже:
# 
# ![image.png](attachment:cf5ff75a-ea5e-40e4-8bc6-bbb12c97814b.png)

# In[66]:


# Сюда сохраняем метрики
all_tp, all_fp, all_fn, all_tn = [], [], [], []

# Проходимся по тестовому набору
for imgs, masks in tqdm(test_loader, desc='Расчет метрик', ncols=100):
    imgs, masks = imgs.to(device), masks.to(device).long()
    # Инфреренс модели
    with torch.no_grad():
        output = model(imgs)['out']
    # Вычисляем метрики
    tp, fp, fn, tn = metrics.get_stats(output, masks, mode='binary', threshold=0.5)
    # Добавляем метрики
    all_tp.append(tp)
    all_fp.append(fp)
    all_fn.append(fn)
    all_tn.append(tn)

# Собираем всё вместе
all_tp = torch.cat(all_tp)
all_fp = torch.cat(all_fp)
all_fn = torch.cat(all_fn)
all_tn = torch.cat(all_tn)

# Считаем IoU по всему датасету
iou = metrics.iou_score(all_tp, all_fp, all_fn, all_tn, reduction='micro')

# Выводим Dice
dice = (2 * iou) / (1 + iou)

print(f"IoU для LR-ASPP: {iou:.4f}")
print(f"Dice для LR-ASPP: {dice:.4f}")


# <a id=1-6></a>
# ## 1.6 Визуализация предсказаний
# 
# В предыдущих модулях мы определили 5 комбинаций признаков. Вот они:
# 
# - нет облачности, малая доля (<3%) с/х угодий
# - малая доля (<3%) облачности, 3 с/х области
# - средняя доля облачности (25-75%), есть с/х поле
# - нет облачности, средняя доля с/х угодий (25-75%)
# - высокая доля облачности (>75%), нет с/х полей
# 
# Загрузим данные по изображениям и отберем 10 изображений для визуализации предсказаний.

# In[72]:


table_df = pd.read_excel(r'E:\Heckfy\atom\Atomskills2026\15.02.2026\homework\table.xlsx')


# In[209]:


# 1 признак, отбираем по 2 изображения
feature_1_images = table_df[(table_df['Наличие облачности'] == 'Нет') & (table_df['% с/х угодий'].str.rstrip('%').astype(float) < 3)].sample(2)['Файл'].tolist()

# 2 признак, отбираем по 2 изображения
feature_2_images = table_df[(table_df['Доля облачности'].str.rstrip('%').astype(float) < 3) & (table_df['Количество областей (с/х поля)'] == 3)].sample(2)['Файл'].tolist()

# 3 признак, отбираем по 2 изображения
feature_3_images = table_df[(table_df['Доля облачности'].str.rstrip('%').astype(float) > 25) & \
                (table_df['Доля облачности'].str.rstrip('%').astype(float) < 75) & \
                (table_df['Есть ли с/х поле?'] == 'Да')].sample(2)['Файл'].tolist()

# 4 признак, отбираем по 2 изображения
feature_4_images = table_df[(table_df['Наличие облачности'] == 'Нет') & \
                (table_df['% с/х угодий'].str.rstrip('%').astype(float) > 25) & \
                (table_df['% с/х угодий'].str.rstrip('%').astype(float) < 75)].sample(2)['Файл'].tolist()

# 5 признак, отбираем по 2 изображения
feature_5_images = table_df[(table_df['Доля облачности'].str.rstrip('%').astype(float) > 75) & \
                        (table_df['Есть ли с/х поле?'] == 'Нет')].sample(2)['Файл'].tolist()


# Теперь у нас есть списки для каждой комбинации, в них хранятся 2 изображения из этого набора.

# In[210]:


feature_1_images


# Сделаем инференс на этих моделях и посмотрим на результат!

# In[211]:


# Инициализируем модель
model = segmentation.lraspp_mobilenet_v3_large(weights=None, num_classes=2)

# Меняем число классов
model.classifier.low_classifier  = nn.Conv2d(40, 1, kernel_size=1)
model.classifier.high_classifier = nn.Conv2d(128, 1, kernel_size=1)

# Загружаем лучшую версию модели
model.load_state_dict(torch.load(r'best_lraspp.pth'))
model = model.to(device)
model = model.eval()


# In[221]:


combs = [
    "Комбинация 1. Нет облачности, малая доля (<3%) с/х угодий",
    "Комбинация 2. Малая доля (<3%) облачности, 3 с/х области",
    "Комбинация 3. Средняя доля облачности (25-75%), есть с/х поле",
    "Комбинация 4. Нет облачности, средняя доля с/х угодий (25-75%)",
    "Комбинация 5. Высокая доля облачности (>75%), нет с/х полей",
]

features_dfs = [
    feature_1_images, feature_2_images, feature_3_images, feature_4_images, feature_5_images
]

for df, comb in zip(features_dfs, combs):
    fig, axs = plt.subplots(2, 3, figsize=(13, 7))
    fig.suptitle(comb)
    for image, ax in zip(df, axs):
        # Загружаем изображениz
        img_path = rf'E:\Heckfy\atom\Atomskills2026\15.02.2026\homework\preprocess_images\images\{image[:-4]}\{image[:-4]}_original.jpg'
        label_path = rf'E:\Heckfy\atom\Atomskills2026\15.02.2026\homework\preprocess_images\labels\{image[:-4]}\{image[:-4]}_original_mask.jpg'
        img = io.read_image(img_path) / 255.0
        img = F.resize(img, (224, 224))
        # Визуализация оригинала
        ax[0].imshow(img.permute(1, 2, 0).cpu().numpy())
        ax[0].set_title('Оригинал')
        ax[1].set_title('Маска')
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        # Mаска загружается как RGB
        mask = io.read_image(label_path)
        mask = F.resize(mask, (224, 224))
        # берем 1 канал и бинаризуем его, чтобы получить одноканальную маску
        mask = mask[0:1, :, :]     # → [1,H,W]
        mask = (mask > 127).float()
        # Визуализация маски
        ax[1].imshow(mask.cpu().numpy()[0], cmap='binary')
        ax[1].set_title('Маска')
        ax[1].set_xticks([])
        ax[1].set_yticks([])

        # Прогоняем через нашу модель
        img = img.to(device)
        img = torch.unsqueeze(img, dim=0)
        output = model(img)['out']
        pred_mask = nn.functional.sigmoid(output)
        numpy_mask = pred_mask.detach().cpu().numpy()[0][0]
        numpy_mask = (numpy_mask > 0.5).astype('int8')
        # Визуализация предсказания
        ax[2].imshow(numpy_mask, cmap='binary')
        ax[2].set_title('Предсказание')
        ax[2].set_xticks([])
        ax[2].set_yticks([])


# Визуализация показала, что в целом алгоритм **хорошо** справляется со своей задачей.

# ###

# <a id=2></a>
# # 2. Сегментация набора из Модуля С
# 
# Выполним сегментацию для заданного набора данных по Краснодарской и Белгородской
# областям.

# ### Подготовка данных

# Изменим класс ```FieldDataset```, чтобы он подходил под наши новые данные. 

# In[334]:


class FieldDataset(Dataset):
    def __init__(self, transform=None,
                images_path=r'E:\Heckfy\atom\Atomskills2026\19.02.2026\homework\valid_new\images',
                labels_path=r'E:\Heckfy\atom\Atomskills2026\19.02.2026\homework\valid_new\labels'):
        self.images_path = images_path
        self.labels_path = labels_path
        self.transform = transform
        self.samples = []

        # Проходимся по каждому файлу в каждой папке
        for image in Path(self.images_path).iterdir():
            # Пропускаем директории
            if image.is_dir():
                continue
            # Сохраняем пути
            label_path = os.path.join(self.labels_path, f'label_{image.stem[6:]}.jpg')
            self.samples.append((
                image,
                label_path
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]

        # читаем как тензоры
        img = io.read_image(img_path) / 255.0
        img = F.resize(img, (224, 224))

        # Mаска загружается как RGB
        mask = io.read_image(label_path)
        mask = F.resize(mask, (224, 224))
        # берем 1 канал и бинаризуем его, чтобы получить одноканальную маску
        mask = mask[0:1, :, :]     # → [1,H,W]
        mask = (mask > 127).float()

        # аугментации (если заданы)
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        return img, mask


# In[373]:


# Воспользуемся написанным классом FieldDataset
dataset_new = FieldDataset(images_path=r'E:\Heckfy\atom\Atomskills2026\19.02.2026\homework\valid_new\images',
                           labels_path=r'E:\Heckfy\atom\Atomskills2026\19.02.2026\homework\valid_new\labels')

# Создаем загрузчик, чтобы подать данные в модель
new_dataloader = DataLoader(dataset_new, batch_size=1, shuffle=True)


# ### Инференс модели
# 
# С помощью модели определим поля на изображении и визуализируем их.

# In[382]:


count_vis = 0

# Проходимся по тестовому набору
for imgs, masks in new_dataloader:
    imgs, masks = imgs.to(device), masks.to(device).long()

    # Инфреренс модели
    with torch.no_grad():
        output = model(imgs)['out']
    output = model(imgs)['out']
    pred_mask = nn.functional.sigmoid(output)
    numpy_mask = pred_mask.detach().cpu().numpy()[0][0]
    numpy_mask = (numpy_mask > 0.5).astype('int8')

    # Отбираем только те снимки, на которых есть поля
    if numpy_mask.sum() == 0:
        continue

    fig, axs = plt.subplots(1, 3, figsize=(13, 7))

    axs[0].imshow(imgs[0].permute(1, 2, 0).cpu().numpy())
    axs[0].set_title('Оригинал')
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    axs[1].imshow(masks.cpu().numpy()[0][0], cmap='binary')
    axs[1].set_title('Маска')
    axs[1].set_xticks([])
    axs[1].set_yticks([])

    axs[2].imshow(numpy_mask, cmap='binary')
    axs[2].set_title('Предсказание')
    axs[2].set_xticks([])
    axs[2].set_yticks([])
    plt.show()

    count_vis += 1
    if count_vis > 4:
        break

