#!/usr/bin/env python
# coding: utf-8

# # Детекция сельскохозяйственных полей на спутниковых изображениях
# 
# В данном ноутбуке реализован полный пайплайн решения задачи объектной детекции: сегментационные маски преобразуются в bounding boxes, выполняется предобработка и аугментация данных, после чего обучаются модели (Faster R-CNN, RetinaNet, SSD) для автоматического обнаружения сельскохозяйственных полей на спутниковых снимках; качество оценивается с использованием метрик mAP, IoU и recall.

# ## Содержание
# * [Импорт библиотек](#0)
# * [1. Загрузка данных](#1)
# * [2. Конвертация сегментационной маски в bbox](#2)
# * [3. Предобработка данных](#3)
# * [4. Аугментация данных](#4)
# * [5. Обучение моделей детекции](#5)
#     * [5.1 Faster R-CNN](#5-1)
#     * [5.2 RetinaNet](#5-2)
#     * [5.3 SSD](#5-3)
# * [6. Вывод](#6)

# ###

# <a id=0></a>
# ## Импорт библиотек

# In[42]:


import numpy as np
import cv2

from skimage.measure import label, regionprops

import albumentations as A

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
import torchvision.transforms.v2 as tfs
import torchvision
from torchmetrics.detection import MeanAveragePrecision, IntersectionOverUnion
from torchvision.ops import box_iou

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from tqdm import tqdm
import json
from pathlib import Path
import os


# ####

# <a id=1></a>
# # 1. Загрузка данных
# 
# Данные представлены в директории `valid_new`, где
# 
# - `images` - исходные спутниковые снимки,
# - `labels` - сегментационные маски для полей.
# 

# In[2]:


# Пути данных
train_path = Path(r'E:\Heckfy\atom\Atomskills2026\19.02.2026\homework\valid_new\images')
mask_path = Path(r'E:\Heckfy\atom\Atomskills2026\19.02.2026\homework\valid_new\labels')


# In[3]:


images = {}

# Загружаем изображения
images['images'] = {}
for i, filename in enumerate(tqdm(train_path.iterdir(), desc='Загрузка изображений', total=660)):
    # Извлекаем имя файла
    name = filename.stem[6:]

    # Открываем картинку в RGB
    img = cv2.imread(filename, cv2.IMREAD_COLOR_RGB)

    # Приводим все картинки к одному размеру (224, 224)
    img = cv2.resize(img, (224, 224))

    # Сохраняем изображение в словарь
    images['images'][name] = img


# Загружаем сег.маски
images['mask'] = {}
for i, filename in enumerate(tqdm(mask_path.iterdir(), desc='Загрузка сег. масок', total=660)):
    # Извлекаем имя файла
    name = filename.stem[6:]

    # Открываем картинку в grayscale
    mask = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    # Приводим все маски к одному размеру (224, 224)
    mask = cv2.resize(mask, (224, 224))

    # Бинаризуем маску
    mask = np.where(mask > 155, 1, 0)

    # Сохраняем маску в словарь
    images['mask'][name] = mask


# ###

# <a id=2></a>
# # 2. Конвертация сегментационной маски в bbox
# 
# Для задачи детекции нам необходимся точные сегментационные маски преобразовать в **ограничивающие прямоугольники** (bounding boxes, bbox) — то есть минимальные прямоугольники, которые полностью покрывают объект на маске.
# 
# Для определения `bbox` воспользуемся методом `regionprops` из библиотеки `skimage`.

# ### Пример конвертации

# In[4]:


# Случайная маска
mask = images['mask']['image_01009_20210720_const_0317']
bbox_mask = mask.copy()
zero_mask = np.zeros_like(bbox_mask)

# Определяем поля
label_image = label(mask)
# Проходимся по каждому полю
for region in regionprops(label_image):
    # Получаем координаты bbox
    y_min, x_min, y_max, x_max = region.bbox
    # Рисуем их на маске
    cv2.rectangle(bbox_mask, (x_min, y_min), (x_max, y_max), color=2)
    cv2.rectangle(zero_mask, (x_min, y_min), (x_max, y_max), color=2)

# Визуализация
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
axs[0].imshow(mask, cmap='gray')
axs[0].set_title('Сегментацонная маска')
axs[0].set_xticks([])
axs[0].set_yticks([])

axs[1].imshow(bbox_mask, cmap='gray')
axs[1].set_title('bbox на маске')
axs[1].set_xticks([])
axs[1].set_yticks([]);

axs[2].imshow(zero_mask, cmap='gray')
axs[2].set_title('Итоговый bbox')
axs[2].set_xticks([])
axs[2].set_yticks([]);


# ### Конвертация для всего датасета

# In[5]:


images['bboxes'] = {}
images['labels'] = {}

# Проходимся по всем изображениям
for name in tqdm(images['images'], desc='Формирование bbox'):
    seg_mask = images['mask'][name]
    # Определяем поля
    label_image = label(seg_mask)

    # Сюда будем сохранять все bbox
    bboxes = []
    labels = []
    # Проходимся по каждому полю
    for region in regionprops(label_image):
        # Получаем координаты bbox
        y_min, x_min, y_max, x_max = region.bbox
        # torch-формат (x_min, y_min, x_max, y_max)
        bboxes.append([x_min, y_min, x_max, y_max])
        # Класс полей - 1
        labels.append(1)

    # Сохраняем bboxes
    images['bboxes'][name] = np.array(bboxes)
    images['labels'][name] = np.array(labels)


# После обработки датасета у нас появились новые данные.

# In[6]:


print('bboxes:')
print(images['bboxes']['image_01008_20210720_const_0001'])
print('Метки для них (поле - 1):')
print(images['labels']['image_01008_20210720_const_0001'])


# ###

# <a id=3></a>
# # 3. Предобработка данных
# 
# Отвлечемся от таргета и предобработаем изображения для качественной детекции.

# ### Нормализация гистограммы
# 
# Нормализация гистограммы используется для `улучшения контраста изображения`.
# Метод перераспределяет значения яркости пикселей таким образом,
# чтобы использовать весь диапазон интенсивностей.
# 
# Для цветных изображений (RGB) нормализация выполняется **не напрямую по каналам**,
# а с использованием цветового пространства **LAB**, где яркость отделена от цвета.
# 
# В пространстве LAB:
# 
# * канал **L** отвечает за яркость
# * каналы **A** и **B** отвечают за цвет
# 
# Это позволяет применять нормализацию **только к яркости**, не искажая цвета изображения.
# 
# Напишем функцию для нормализации гистограммы с использованием LAB:

# In[7]:


def normalize_hist(img: np.array) -> np.array:
    # Из RGB в LAB
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    # Нормализация гистограммы (equalization)
    l = cv2.equalizeHist(l)
    # Обратно
    lab = cv2.merge((l, a, b))
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return result


# Вот пример `обычного` и `нормализованного` изображения:

# In[8]:


# Случайный сэмпл
sample_img = images['images']['image_01009_20210720_const_0233']
# Нормируем гистограмму
normalized_img = normalize_hist(sample_img)

# Визуализация
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].imshow(sample_img)
axs[0].set_title('Оригинальное изображение')
axs[0].set_xticks([])
axs[0].set_yticks([])

axs[1].imshow(normalized_img)
axs[1].set_title('После нормализации гистограммы')
axs[1].set_xticks([])
axs[1].set_yticks([]);


# Как видно из визуализации, разные элементы на изображении стали **более различимы**.
# 
# Применим функцию для всего датасета.

# In[9]:


# Проходимся по всем изображениям
for name in tqdm(images['images'], desc='Нормализация гистограмм'):
    img = images['images'][name]

    # Нормируем гистограмму
    normalized_img = normalize_hist(img)

    # Сохраняем новое изображение
    images['images'][name] = normalized_img


# ### Удаление шума
# 
# Для улучшения качества детекции очистим изображение от шума с помощью метода `bilateralFilter`.
# 
# Это позволит модели лучше выделять значимые объекты и снижает влияние случайных пиксельных колебаний.
# 
# Напишем функцию для удаления шума:

# In[10]:


def denoise_image(img: np.array) -> np.array:
    # Применение размытия
    result = cv2.bilateralFilter(img, d=5, sigmaColor=75, sigmaSpace=75)
    return result


# Вот пример `обычного` и `отчищенного от шума` изображения:

# In[11]:


# Случайный сэмпл
sample_img = images['images']['image_01008_20210720_const_0001']
# Удаляем шум
denoised_img = denoise_image(sample_img)

# Визуализация
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].imshow(sample_img)
axs[0].set_title('Оригинальное изображение')
axs[0].set_xticks([])
axs[0].set_yticks([])

axs[1].imshow(denoised_img)
axs[1].set_title('После удаления шума')
axs[1].set_xticks([])
axs[1].set_yticks([]);


# Как видно из визуализации, **шума на изображении стало меньше**.
# 
# Применим функцию для всего датасета.

# In[12]:


# Проходимся по всем изображениям
for name in tqdm(images['images'], desc='Удаление шума'):
    img = images['images'][name]

    # Удаляем шум
    denoised_img = denoise_image(img)

    # Сохраняем новое изображение
    images['images'][name] = denoised_img


# ###

# <a id=4></a>
# # 4. Аугментация данных
# 
# Для увеличения обучющей выборки проведем **аугментацию данных** c помощью библиотеки `albumentations`.

# Напишем функцию, которая на вход будет получать **изображения** и **bbox**, а выдавать его аугментированные вариации. Очень важно применять аугментации, связанные с геометрией, и на bbox, чтобы не испортить данные.
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

# In[13]:


def augment_image(image: np.array, bbox: list, labels: list) -> dict:

    # Зеркальное отражение по горизонтали
    transform = A.Compose([A.HorizontalFlip(p=1.0)],
                          bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_area=1, min_visibility=0.3))
    fliped = transform(image=image, bboxes=bbox, labels=labels)
    flipped_img = fliped['image']
    flipped_bbox = fliped['bboxes']

    # Поворот
    transform = A.Compose([A.RandomRotate90(p=1.0)],
                          bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_area=1, min_visibility=0.3))
    turned = transform(image=image, bboxes=bbox, labels=labels)
    turned_img = turned['image']
    turned_bbox = turned['bboxes']

    # Случайное Приближение/Отдаление
    transform = A.Compose(
        transforms=[A.Affine(scale=(0.3, 2), p=1.0)],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_area=1, min_visibility=0.3))
    zoomed = transform(image=image, bboxes=bbox, labels=labels)
    zoomed_img = zoomed['image']
    zoomed_bbox = zoomed['bboxes']

    # Смена яркости (bbox не передаем)
    transform = A.Compose(
        transforms=[A.RandomBrightnessContrast(brightness_limit=(-0.5, 0.5), p=1.0)]
    )
    brighted = transform(image=image,)
    brighted_img = brighted['image']
    brighted_bbox = bbox

    # Смена контрастности (bbox не передаем)
    transform = A.Compose(
        transforms=[A.RandomBrightnessContrast(contrast_limit=(-0.5, 0.5), p=1.0)])
    contrasted = transform(image=image)
    contrasted_img = contrasted['image']
    contrasted_bbox = bbox

    # Случайные обрезки
    transform = A.Compose(
        transforms=[A.RandomResizedCrop(
            size=(image.shape[0], image.shape[1]),  # (H, W)
            scale=(0.008, 1.0),
            p=1.0
        )],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_area=1, min_visibility=0.3)
    )
    cropped = transform(image=image, bboxes=bbox, labels=labels)
    cropped_img = cropped['image']
    cropped_bbox = cropped['bboxes']

    # Случайный наклон (shear)
    transform = A.Compose(
        transforms=[A.Affine(shear=(-30, 30), p=1.0)],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_area=1, min_visibility=0.3)
    )
    sheared = transform(image=image, bboxes=bbox, labels=labels)
    sheared_img = sheared['image']
    sheared_bbox = sheared['bboxes']

    # Случайное вращение
    transform = A.Compose(
        transforms=[A.Rotate(limit=(-30, 30), p=1.0)],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_area=1, min_visibility=0.3)
    )
    rotated = transform(image=image, bboxes=bbox, labels=labels)
    rotated_img = rotated['image']
    rotated_bbox = rotated['bboxes']

    return {
        'original': [image, bbox, labels],
        'mirroring': [flipped_img, flipped_bbox, fliped['labels']],
        'turning': [turned_img, turned_bbox, turned['labels']],
        'zooming': [zoomed_img, zoomed_bbox, zoomed['labels']],
        'brightening': [brighted_img, brighted_bbox, labels],
        'contrasting': [contrasted_img, contrasted_bbox, labels],
        'cropping': [cropped_img, cropped_bbox, cropped['labels']],
        'shearing': [sheared_img, sheared_bbox, sheared['labels']],
        'rotating': [rotated_img, rotated_bbox, rotated['labels']],
    }


# ### Визуализация аугментаций

# In[14]:


# Проводим аугментацию
sample_img = images['images']['image_01008_20210720_const_0001'].copy()
sample_bbox = images['bboxes']['image_01008_20210720_const_0001']
sample_labels = images['labels']['image_01008_20210720_const_0001']
sample_augs = augment_image(sample_img, sample_bbox, sample_labels)

# Визуализация
fig, axs = plt.subplots(3, 3, figsize=(12, 12))
for ax, type_aug in zip(axs.ravel(), sample_augs):
    aug_image = sample_augs[type_aug][0]
    bboxes = sample_augs[type_aug][1]

    # Oтрисовывем bbox
    for bbox in bboxes:
        bbox = bbox.astype(int)
        cv2.rectangle(img=aug_image, pt1=(bbox[0], bbox[1]), pt2=(bbox[2], bbox[3]), color=(0, 255, 0), thickness=2)

    ax.imshow(aug_image)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(type_aug)


# Применим аугментации для всего датасета.

# In[15]:


# Сохраняем сюда аугментированные данные
aug_images = {}
aug_images['image'] = {}
aug_images['bboxes'] = {}
aug_images['labels'] = {}

# Проходимся по всем изображениям
for name in tqdm(images['images'], desc='Аугментация изображений'):
    # Данные по сэмплу
    img = images['images'][name]
    bboxes = images['bboxes'][name]
    labels = images['labels'][name]

    # Аугментация
    augs = augment_image(img, bboxes, labels)

    # Сохраняем каждую аумгентацию в новый словарь
    for aug_type in augs:
        aug_img, aug_bbox, aug_labels = augs[aug_type]
        aug_name = f'{name}_{aug_type}'
        aug_images['image'][aug_name] = aug_img
        aug_images['bboxes'][aug_name] = aug_bbox
        aug_images['labels'][aug_name] = aug_labels


# In[16]:


print('Количество изображений ДО аугментации:', len(images['images']))
print('Количество изображений ПОСЛЕ аугментации:', len(aug_images['image']))


# С помощью аугментации мы увеличили размер датасета **в 9 раз**!

# ### Сохранения датасета с аугментациями
# 
# Сохраним данные в таком формате:
# 
# ```
# augmented_images_det/
# |---images/
#     |---image1.png
# |---labels/
#     |---image1.json
# 
# ```

# In[17]:


# Сoздаем директорию
os.makedirs('augmented_images_det/images', exist_ok=True)
os.makedirs('augmented_images_det/labels', exist_ok=True)

for name in tqdm(aug_images['image'], desc='Сохранение датасета'):
    img = aug_images['image'][name]
    bboxes = aug_images['bboxes'][name]
    labels = aug_images['labels'][name]

    # приведение bbox к правильной форме [N, 4]
    bboxes = np.array(bboxes, dtype=np.float32)
    if bboxes.ndim == 1:
        if bboxes.size == 0:
            bboxes = np.zeros((0, 4), dtype=np.float32)
        else:
            bboxes = bboxes.reshape(1, 4)

    labels = np.array(labels, dtype=np.int64).reshape(-1)

    target = {
        "boxes": bboxes.tolist(),
        "labels": labels.tolist()
    }

    # RGB to BGR (OpenCV формат)
    img_to_save = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(f'augmented_images_det/images/{name}.png', img_to_save)

    with open(f"augmented_images_det/labels/{name}.json", "w") as f:
        json.dump(target, f)


# ###

# <a id=5></a>
# # 5. Обучение моделей детекции
# 
# В качестве моделей выберем три классических архитектуры: **Faster R-CNN**, **RetinaNet** и **SSD**. 
# 
# Каждая из них представляет отдельный подход к задаче детекции объектов — двухэтапный (**Faster R-CNN**) и одноэтапные (**RetinaNet**, **SSD**) — что позволит сравнить качество и скорость инференса в рамках единого эксперимента.

# In[18]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ## Подготовка данных

# ### Создание `Dataset`

# In[19]:


# Класс для загрузки наших данных
class DetectionDataset(Dataset):
    def __init__(self, images_path: str, labels_path: str, transforms=None):
        self.transforms = transforms

        self.image_paths = sorted(Path(images_path).glob("*.png")) # Пути к изображениям
        # Пути к таргету
        self.label_paths = [
            Path(labels_path) / (p.stem + ".json")
            for p in self.image_paths
        ]

    def __len__(self,):
        return len(self.image_paths)

    def __getitem__(self, idx):
         # Загрузка изображения
        img_path = self.image_paths[idx]
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR_RGB)

        # Загрузка таргета
        with open(self.label_paths[idx]) as f:
            label = json.load(f)

        boxes = torch.tensor(label["boxes"], dtype=torch.float32)
        labels = torch.tensor(label["labels"], dtype=torch.int64)

        if boxes.ndim == 1:
            if boxes.numel() == 0:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
            else:
                boxes = boxes.unsqueeze(0)

        # Удаляем битые bbox
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[keep]
        labels = labels[keep]

        # Предобработка
        if self.transforms:
            image = self.transforms(
                image,
            )

        target = {
            "boxes": boxes,
            "labels": labels
        }

        return image, target, img_path


# In[20]:


# Объявим трансформации для датасета
transforms = tfs.Compose([
    tfs.ToImage(),
    # tfs.Resize((224, 224)),
    tfs.ToDtype(torch.float32, scale=True),  # [0,255] → [0,1]
    tfs.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# In[21]:


# Создаем экземпляр класса
dataset = DetectionDataset('augmented_images_det/images', 'augmented_images_det/labels', transforms=transforms)


# In[22]:


# 0 - нет поля, 1 - поле есть
num_classes = 2


# ### Разделение выборки
# 
# Разделим данные на три части: **70% / 15% / 15%**.
# 
# * **70%** — обучающая выборка. Используется непосредственно для дообучения модели и изменения её весов.
# * **15%** — валидационная выборка. Примеры из этого набора не участвуют в обучении, но используются для контроля процесса: по валидационному *loss* и метрикам сохраняется лучшая модель.
# * **15%** — тестовая выборка. Полностью откладывается до конца эксперимента и используется только для финальной, объективной оценки качества модели на данных, которые она никогда не видела.
# 
# Такое разбиение позволяет одновременно эффективно обучить модель, своевременно отслеживать переобучение и провести честную итоговую проверку качества.

# In[23]:


# Определим размеры выборок
train_size = int(0.7 * len(dataset))
val_size = (len(dataset) - train_size) // 2
test_size = len(dataset) - train_size - val_size

train_subset, val_subset, test_subset = random_split(dataset, [train_size, val_size, test_size])

print('Размеры выборок:')
len(train_subset), len(val_subset), len(test_subset)


# ### Инициализация `Dataloader`

# In[24]:


# Создаем collator
def collate_fn(batch):
    return tuple(zip(*batch))

train_loader = DataLoader(train_subset, batch_size=4, shuffle=True,collate_fn=collate_fn)
val_loader = DataLoader(val_subset, batch_size=4, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_subset, batch_size=4, shuffle=False, collate_fn=collate_fn)


# ### Функция обучения

# In[25]:


# Функция обучения для одной эпохи
def train_one_epoch(
    model: torch.nn.Module, # модель детекции
    optimizer: torch.optim.Optimizer, # оптимизатор
    data_loader: DataLoader, # данные
    epoch: int, # номер текущей эпохи
) -> float:
    model.train()
    total_loss = 0.0

    # Логирование
    loop = tqdm(data_loader, desc=f"Epoch [{epoch}]")

    # Проходимся по всем данным
    for images, targets, _ in loop:
        # Переносим данные на устройство
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Модели детекции сам считает losses
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

        # Отображаем детализацию losses в прогресс-баре
        loop.set_postfix(loss=f"{losses.item():.4f}")

    # Возвращаем лосс
    return total_loss / len(data_loader)


# In[26]:


# Полный цикл обучения
def train(model, train_loader: DataLoader, num_epochs: int, best_model_path='models/best_model_fastercnn.pth'):

    # Конфигурация
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Для сохранения лучшей модели
    best_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        # Обучение на эпохе
        epoch_loss = train_one_epoch(model, optimizer, train_loader, epoch)

        # Валидация
        val_metrics = evaluate(model, val_loader)
        val_loss = val_metrics["loss"]

        # сохраняем лучшую модель по минимальному loss
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Лучшая модель сохранена (val_loss={best_loss:.4f})")

        # Шаг планировщика
        scheduler.step()
        print(
            f"\nEpoch [{epoch}/{num_epochs}]"
            f"\nTrain Loss: {epoch_loss:.4f}"
            f"\nVal Loss:   {val_metrics['loss']:.4f}"
            f"\nLR:         {scheduler.get_last_lr()[0]:.6f}\n"
        )

    return model


# ### Функция валидации

# In[44]:


def evaluate(model, data_loader, iou_threshold=0.5):
    model.eval()

    metric = MeanAveragePrecision(iou_thresholds=[iou_threshold])
    iou_metric = IntersectionOverUnion(iou_threshold=iou_threshold)

    total_loss = 0.0
    dices = []

    for images, targets, _ in tqdm(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Лосс
        model.train()
        with torch.no_grad():
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            total_loss += loss.item()

        # Инференс
        model.eval()
        with torch.no_grad():
            preds = model(images)

        # Метрики
        metric.update(preds, targets)
        iou_metric.update(preds, targets)

        # Dice
        dices.append(compute_dice(preds, targets, iou_threshold))

    metrics = metric.compute()
    iou = iou_metric.compute()

    return {
        "loss": total_loss / len(data_loader),
        "mAP": metrics["map"].item(),
        "mAP_50": metrics["map_50"].item(),
        "recall": metrics["mar_100"].item(),
        "iou": iou["iou"].item(),
        "dice": sum(dices) / len(dices) if dices else 0.0,
    }


# Для оценки качества детекции дополнительно используется самописная функция расчёта **Dice coefficient**.  
# 
# Метрика вычисляется на основе IoU между предсказанными и целевыми bounding box: для каждого предсказания выбирается наиболее подходящий ground truth, после чего применяется преобразование  
# `Dice = 2 * IoU / (1 + IoU)`.  

# In[45]:


def compute_dice(preds, targets, iou_threshold=0.5):
    dices = []

    for p, t in zip(preds, targets):
        # пропускаем, если нет предсказаний или таргета
        if not len(p["boxes"]) or not len(t["boxes"]):
            continue

        # IoU между всеми предиктами и таргетом
        iou = box_iou(p["boxes"], t["boxes"]).max(dim=1).values

        # оставляем только хорошие матчи
        iou = iou[iou > iou_threshold]

        # считаем Dice = 2*IoU / (1 + IoU) и усредняем по изображению
        if len(iou):
            dices.append((2 * iou / (1 + iou)).mean().item())

    # усреднение по батчу
    return sum(dices) / len(dices) if dices else 0.0


# ### Функция визуализации

# In[28]:


# Денормазилация для корректного отображения аудио
def denormalize(image):
    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(3,1,1)
    return image * std + mean


# In[38]:


def visualize_sample(model, dataset, score_thresh=0.7):

    # Берем подвыборку для визуализации
    rand_indx = np.random.randint(0, len(dataset), size=4)
    samples = [dataset[i] for i in rand_indx]

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    for sample, ax in zip(samples, axs.ravel()):
        image, target, filepath = sample

        ax.set_title(filepath.stem)
        ax.set_xticks([])
        ax.set_yticks([])

        # Оригинальное изображение
        orig_image = cv2.imread(filepath, cv2.IMREAD_COLOR_RGB)
        ax.imshow(orig_image)
        # Оригинальный таргет
        boxes = target["boxes"]

        # Отображаем bbox фактические
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.tolist()
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor='red',
                facecolor='none',
                label='Исходные bbox' if i == 0 else ''
            )
            ax.add_patch(rect)

        # Подготовка входа
        img = image.to(device)
        # Инференс модели
        with torch.no_grad():
            prediction = model([img])[0]
        pred_boxes = prediction["boxes"].cpu()
        pred_scores = prediction["scores"].cpu()

        first = True
        # Отображаем bbox предсказанные
        for i, (box, score) in enumerate(zip(pred_boxes, pred_scores)):
            if score < score_thresh:
                continue

            x1, y1, x2, y2 = box.tolist()

            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor='blue',
                facecolor='none',
                linestyle='--',
                label='Предсказнные bbox' if first else ""
            )

            first = False

            ax.add_patch(rect)

            # Подпись confidence
            ax.text(
                x1, y1 - 5,
                f"{score:.2f}",
                color='white',
                fontsize=10,
                backgroundcolor='black'
            )

    plt.suptitle('Визуализация детекции')
    plt.legend()
    plt.axis('off')
    plt.show()


# <a id=5-1></a>
# ## 5.1 Faster R-CNN

# ### Инициализация модели

# In[30]:


# предобученная модель
model_fasterrcnn = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")

# меняем классификатор (2 класса)
in_features = model_fasterrcnn.roi_heads.box_predictor.cls_score.in_features
model_fasterrcnn.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

model_fasterrcnn = model_fasterrcnn.to(device)


# ### Обучение модели

# In[31]:


# Запуск скрипта обучения
trained_model = train(model_fasterrcnn, train_loader, num_epochs=10)


# ### Метрики
# 
# Для оценки качества детекции используем следующие метрики:
# 
# * **Loss** — средняя функция потерь на валидации (чем меньше, тем лучше).
# * **mAP** — основная метрика качества детекции, учитывающая точность и полноту.
# * **mAP@0.5** — mAP при IoU = 0.5 (более мягкий критерий).
# * **Recall (mar_100)** — доля найденных объектов.
# * **IoU** — точность совпадения предсказанных и истинных bbox.
# 
# Метрики позволяют оценить модель по трём аспектам: точность, полнота и качество локализации объектов.
# 
# 

# Загружаем лучшую модель по пути `models/best_model_fastercnn.pth`

# In[32]:


model_fasterrcnn = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")

# меняем классификатор (2 класса)
in_features = model_fasterrcnn.roi_heads.box_predictor.cls_score.in_features
model_fasterrcnn.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# Загружаем веса из файла
model_fasterrcnn.load_state_dict(torch.load('models/best_model_fastercnn.pth'))

model_fasterrcnn_best = model_fasterrcnn.to(device)


# In[46]:


# Инференс
metrics = evaluate(model_fasterrcnn_best, test_loader)

print(
    f"Loss: {metrics['loss']:.4f} | "
    f"mAP: {metrics['mAP']:.4f} | "
    f"mAP@0.5: {metrics['mAP_50']:.4f} | "
    f"Recall: {metrics['recall']:.4f} | "
    f"IoU: {metrics['iou']:.4f} | "
    f"Dice: {metrics['dice']:.4f} | "
)


# ### Визуализация детекции

# In[39]:


# Используем функцию визуализации
visualize_sample(model_fasterrcnn_best, dataset)


# <a id=5-2></a>
# ## 5.2 RetinaNet

# ### Инициализация модели

# In[47]:


# Загружаем модель
model_retina = torchvision.models.detection.retinanet_resnet50_fpn_v2(weights="DEFAULT")

# Получаем число каналов
in_channels = model_retina.head.classification_head.conv[0][0].in_channels
num_anchors = model_retina.head.classification_head.num_anchors

# Заменяем голову
model_retina.head.classification_head = torchvision.models.detection.retinanet.RetinaNetClassificationHead(
    in_channels,
    num_anchors,
    num_classes
)
model_retina = model_retina.to(device)


# ### Обучение модели

# In[49]:


# Запуск скрипта обучения
trained_model = train(model_retina, train_loader, num_epochs=5, best_model_path='models/best_model_retina.pth')


# ### Метрики
# 
# Для оценки качества детекции используем следующие метрики:
# 
# * **Loss** — средняя функция потерь на валидации (чем меньше, тем лучше).
# * **mAP** — основная метрика качества детекции, учитывающая точность и полноту.
# * **mAP@0.5** — mAP при IoU = 0.5 (более мягкий критерий).
# * **Recall (mar_100)** — доля найденных объектов.
# * **IoU** — точность совпадения предсказанных и истинных bbox.
# 
# Метрики позволяют оценить модель по трём аспектам: точность, полнота и качество локализации объектов.
# 
# 

# Загружаем лучшую модель по пути `models/best_model_retina.pth`

# In[50]:


model_retina_best = torchvision.models.detection.retinanet_resnet50_fpn_v2(weights="DEFAULT")

# Получаем число каналов
in_channels = model_retina.head.classification_head.conv[0][0].in_channels
num_anchors = model_retina.head.classification_head.num_anchors

# Заменяем голову
model_retina_best.head.classification_head = torchvision.models.detection.retinanet.RetinaNetClassificationHead(
    in_channels,
    num_anchors,
    num_classes
)

# Загружаем веса из файла
model_retina_best.load_state_dict(torch.load('models/best_model_retina.pth'))

model_retina_best = model_retina_best.to(device)


# In[51]:


# Инференс
metrics = evaluate(model_retina_best, test_loader)

print(
    f"Loss: {metrics['loss']:.4f} | "
    f"mAP: {metrics['mAP']:.4f} | "
    f"mAP@0.5: {metrics['mAP_50']:.4f} | "
    f"Recall: {metrics['recall']:.4f} | "
    f"IoU: {metrics['iou']:.4f} | "
    f"Dice: {metrics['dice']:.4f} | "
)


# ### Визуализация детекции

# In[56]:


# Используем функцию визуализации
visualize_sample(model_retina_best, dataset, score_thresh=0.5)


# <a id=5-3></a>
# ## 5.3 SSD

# ### Инициализация модели

# In[53]:


# Загружаем модель
model_ssd = torchvision.models.detection.ssd300_vgg16(
    weights=None,
    weights_backbone="DEFAULT",
    num_classes=num_classes
)

model_ssd = model_ssd.to(device)


# ### Обучение модели

# In[54]:


# Запуск скрипта обучения
trained_model = train(model_ssd, train_loader, num_epochs=10, best_model_path='models/best_model_ssd.pth')


# ### Метрики
# 
# Для оценки качества детекции используем следующие метрики:
# 
# * **Loss** — средняя функция потерь на валидации (чем меньше, тем лучше).
# * **mAP** — основная метрика качества детекции, учитывающая точность и полноту.
# * **mAP@0.5** — mAP при IoU = 0.5 (более мягкий критерий).
# * **Recall (mar_100)** — доля найденных объектов.
# * **IoU** — точность совпадения предсказанных и истинных bbox.
# 
# Метрики позволяют оценить модель по трём аспектам: точность, полнота и качество локализации объектов.
# 
# 

# Загружаем лучшую модель по пути `models/best_model_ssd.pth`

# In[57]:


model_ssd_best = torchvision.models.detection.ssd300_vgg16(
    weights=None,
    weights_backbone='DEFAULT',
    num_classes=num_classes
)

# Загружаем веса из файла
model_ssd_best.load_state_dict(torch.load('models/best_model_ssd.pth'))

model_ssd_best = model_ssd_best.to(device)


# In[58]:


# Инференс
metrics = evaluate(model_ssd_best, test_loader)

print(
    f"Loss: {metrics['loss']:.4f} | "
    f"mAP: {metrics['mAP']:.4f} | "
    f"mAP@0.5: {metrics['mAP_50']:.4f} | "
    f"Recall: {metrics['recall']:.4f} | "
    f"IoU: {metrics['iou']:.4f} | "
    f"Dice: {metrics['dice']:.4f} | "
)


# ### Визуализация детекции

# In[69]:


# Используем функцию визуализации
visualize_sample(model_ssd_best, dataset, score_thresh=0.3)


# ###

# <a id=6></a>
# # 6. Вывод
# 
# В работе был реализован **полный пайплайн задачи детекции** сельскохозяйственных полей: от преобразования сегментационных масок в **bounding box** до обучения и оценки моделей.
# 
# Проведена **предобработка** и **аугментация данных**, что позволило увеличить датасет **в 9 раз** и повысить обобщающую способность моделей.
# 
# Среди рассмотренных архитектур:
# 
# * **Faster R-CNN** — показала **лучшее качество** (*mAP ≈ 0.78*), обеспечив баланс между точностью и полнотой
# * **RetinaNet** — продемонстрировала **сопоставимые результаты** с более высоким *recall*
# * **SSD** — **значительно уступила** по качеству
# 
# Полученные результаты подтверждают, что **двухэтапные детекторы** лучше подходят для данной задачи по сравнению с одноэтапными.
