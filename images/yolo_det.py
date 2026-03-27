#!/usr/bin/env python
# coding: utf-8

# # Детекция объектов с использованием Ultralytics YOLO
# 
# В ноутбуке представлен процесс обучения модели детекции объектов на основе архитектуры YOLO с использованием библиотеки **Ultralytics**.
# 
# Рассматриваются ключевые этапы: подготовка и конвертация данных, формирование датасета, дообучение предобученной модели, а также оценка качества с использованием метрик *Precision*, *Recall*, *IoU* и *Dice*.
# 
# Модель обучается на пользовательском датасете и оценивается на валидационной выборке, что обеспечивает объективную оценку её эффективности.
# 

# ## Содержание
# 
# * [Импорт библиотек](#0)
# * [1. Конвертация данных в Yolo-формат](#1)
# * [2. Разделение выборки](#2)
# * [3. Конфигурационный файл датасета для YOLO](#3)
# * [4. Обучение модели детекции](#4)
# * [5. Сохранение модели](#5)
# * [6. Метрики модели](#6)
# * [7. Визуализация детекции](#7)
# * [8. Вывод](#8)

# ##

# <a id=0></a>
# ## Импорт библиотек

# In[1]:


from ultralytics import YOLO
from ultralytics.data.split import split_classify_dataset

import torch
from torchvision.ops import box_iou
import cv2

import matplotlib.pyplot as plt
from matplotlib import patches

import random
from pathlib import Path
from pprint import pprint
from tqdm import tqdm
import os
import json


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# ###

# <a id=1></a>
# # 1. Конвертация данных в Yolo-формат

# В предыдущем ноутбуке мы дообучали модели детекции из **PyTorch**, где данные имели следующий формат:
# ```
# augmented_images_det/
# |---images/
#     |---image1.png
# |---labels/
#     |---image1.json
# 
# ```
# 
# Однако при переходе к использованию **Ultralytics YOLO**, требуется другой формат аннотаций. **YOLO** ожидает, что каждая разметка будет храниться в `.txt` файле и иметь вид:
# 
# ```
# <class_id> <x_center> <y_center> <width> <height>
# ```
# 
# Поэтому, напишем функцию для конвертации данных.

# In[3]:


def convert_json_labels(json_dir='augmented_images_det/labels', images_dir='augmented_images_det/images', out_labels_dir='yolo_dataset_det/labels'):
    # Создаем папку с конвертированныеми данными
    os.makedirs(out_labels_dir, exist_ok=True)

    for file in tqdm(os.listdir(json_dir), desc='Конвертация'):
        if not file.endswith(".json"):
            continue

        # Пути к файлам
        json_path = os.path.join(json_dir, file)
        img_path = os.path.join(images_dir, file.replace(".json", ".png"))
        txt_path = os.path.join(out_labels_dir, file.replace(".json", ".txt"))

        # Читаем json
        with open(json_path, "r") as f:
            data = json.load(f)
        boxes = data["boxes"]
        labels = data["labels"]

        # Сохраняем размеры изображения
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        lines = []

        # Проходимся по каждому bbox
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box

            # YOLO конвертация + нормализация
            xc = (x1 + x2) / 2 / w
            yc = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            cls = int(label) - 1  # YOLO с 0
            # Добавляем строку
            lines.append(f"{cls} {xc} {yc} {bw} {bh}")

        # Сохраняем
        with open(txt_path, "w") as f:
            f.write("\n".join(lines))


# Запуск конвертации
convert_json_labels()


# Пример конвертации:

# In[4]:


with open('augmented_images_det/labels/image_01008_20210720_const_0001_brightening.json') as f:
    print('PyTorch-формат:')
    pprint(f.readlines()[0])
print()
with open('yolo_dataset_det/labels/image_01008_20210720_const_0001_contrasting.txt') as f:
    print('YOLO-формат:')
    print(f.readlines()[0])


# <a id=2></a>
# # 2. Разделение выборки
# 
# Разделим данные на две части: **80% / 20%**.
# 
# * **80%** — обучающая выборка. Используется непосредственно для дообучения модели и изменения её весов.
# * **20%** — валидационная выборка. Примеры из этого набора не участвуют в обучении, но используются для контроля процесса: по валидационному *loss* и метрикам сохраняется лучшая модель.
# 
# Такое разбиение позволяет одновременно эффективно обучить модель и провести честную итоговую проверку качества.

# In[5]:


# Используем готовую функцию для разбиения датасета
split_data_path = split_classify_dataset(source_dir='yolo_dataset_det', train_ratio=0.8)


# Итоговый датасет для обучения выглядит так:
# ```
# yolo_dataset_det_split /
# ├── images/
# │   ├── train/
# │   ├── val/
# ├── labels/
# │   ├── train/
# │   ├── val/
# ```

# <a id=3></a>
# # 3. Конфигурационный файл датасета для YOLO
# 
# В корне проекта создадим конфигурационный файл `data.yaml`, это инструкция для **YOLO**, где лежит датасет и какие классы.
# 
# ```
# path: yolo_dataset_det_split
# train: train/images
# val: val/images
# 
# names:
#   0: field
# ```

# <a id=4></a>
# # 4. Обучение модели детекции
# 
# 
# В качестве модели была выбрана архитектура **YOLO** из библиотеки `Ultralytic`. Использовалась предобученная модель `yolo11n.pt`, которая обеспечивает быстрый старт обучения и хорошее качество за счёт уже выученных признаков.
# 
# Модель дообучается (fine-tuning) на нашем датасете с использованием файла `data.yaml`, где заданы пути к данным и классы.

# ### Инициализация модели

# In[8]:


model = YOLO("yolo26n.pt")


# ### Обучение модели

# In[9]:


results = model.train(
    data="yolo_dataset_det_split/data.yaml", # Путь к конфигурации
    epochs=30, # Количество эпох
    augment=False, # Отключаем аугментации
    imgsz=224 # Размер изображений
)

print(results.results_dict)


# <a id=5></a>
# # 5. Сохранение модели
# 
# После каждой эпохи `ultralytics` сохраняет веса. Нам нужна лучшая модель за все обучение. Веса для нее хранятся в `runs\detect\train9\weights\best.pt` .

# In[10]:


custom_model_path = r"E:\Heckfy\atom\REA\preparing_total\runs\detect\train9\weights\best.pt"
best_model = YOLO(custom_model_path)


# <a id=6></a>
# # 6. Метрики модели

# ### Функции для оценки

# Функция `match_boxes` выполняет сопоставление предсказанных `bounding box` с эталонными на основе метрики `IoU`.

# In[11]:


def match_boxes(gt_boxes, pred_boxes):
    matches = []  # Список для хранения совпадений (GT, Pred, IoU)
    used_preds = set()  # Индексы предсказаний, которые уже использованы

    for gt in gt_boxes:  # Проходим по каждому GT-боксу

        # Считаем IoU между текущим GT и всеми предсказанными боксами
        ious = torch.stack([box_iou(gt.unsqueeze(0), p.unsqueeze(0))[0, 0] for p in pred_boxes])

        # Исключаем уже использованные предсказания
        for idx in used_preds:
            ious[idx] = -1

        # Находим индекс предсказания с максимальным IoU
        best_idx = torch.argmax(ious)

        # Проверяем, что совпадение достаточно хорошее (IoU > 0.5)
        if ious[best_idx] > 0.5:
            used_preds.add(best_idx)  # Помечаем предсказание как использованное

            # Сохраняем тройку: (GT, лучший Pred, IoU)
            matches.append((gt, pred_boxes[best_idx], ious[best_idx]))

    return matches  # Возвращаем список всех сопоставлений


# Функция `dice_score` вычисляет **Dice коэффициент** для сэмпла.

# In[12]:


def dice_score(box1, box2):
    # Вычисляем координаты пересечения
    x1 = torch.max(box1[0], box2[0])
    y1 = torch.max(box1[1], box2[1])
    x2 = torch.min(box1[2], box2[2])
    y2 = torch.min(box1[3], box2[3])

    # Площадь пересечения
    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    # Площади боксов
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])

    # Dice коэффициент
    return 2 * inter / (area1 + area2 + 1e-6)


# Функция `evaluate_detection` возвращает метрики для модели.

# In[13]:


def evaluate_detection(model, dataset):
    total_tp = 0  # Общее количество True Positives
    total_fp = 0  # Общее количество False Positives
    total_fn = 0  # Общее количество False Negatives

    all_ious = []   # Список IoU
    all_dices = []  # Список Dice

    # Проходимся по всем изображениям
    for img_path, gt_boxes in tqdm(dataset, total=1188, desc='Инференс модели'):

        # Запускаем инференс YOLO для одного изображения
        result = model(img_path, verbose=False)[0]

        # Предсказанные боксы в формате xyxy
        pred_boxes = result.boxes.xyxy.cpu()

        # GT боксы переводим в torch.Tensor
        gt_boxes = torch.tensor(gt_boxes)

        # Если нет предсказаний → все GT считаются пропущенными (FN)
        if len(pred_boxes) == 0:
            total_fn += len(gt_boxes)
            continue

        # Если нет GT → все предсказания ложные (FP)
        if len(gt_boxes) == 0:
            total_fp += len(pred_boxes)
            continue

        # Сопоставляем GT и Pred
        matches = match_boxes(gt_boxes, pred_boxes)

        # True Positives = количество совпадений
        tp = len(matches)

        # False Positives = лишние предсказания
        fp = len(pred_boxes) - tp

        # False Negatives = пропущенные GT
        fn = len(gt_boxes) - tp

        # Накопление общей статистики
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Считаем IoU и Dice только для TP
        for gt, pred, iou in matches:
            all_ious.append(iou.item())
            all_dices.append(dice_score(gt, pred).item())

    # Precision = TP / (TP + FP)
    precision = total_tp / (total_tp + total_fp + 1e-6)

    # Recall = TP / (TP + FN)
    recall = total_tp / (total_tp + total_fn + 1e-6)

    # Средний IoU по всем TP
    mean_iou = sum(all_ious) / len(all_ious) if all_ious else 0

    # Средний Dice по всем TP
    mean_dice = sum(all_dices) / len(all_dices) if all_dices else 0

    # Возвращаем словарь с метриками
    return {
        "precision": precision,
        "recall": recall,
        "mean_iou": mean_iou,
        "mean_dice": mean_dice
    }


# ### Загрузка валидационных данных

# In[14]:


images_path = 'yolo_dataset_det_split/val/images'
images = []
labels = []
for i in Path(images_path).iterdir():
    images.append(i)
    labels_path = f'augmented_images_det/labels/{i.stem}.json'
    with open(labels_path, 'r') as f:
        data = json.load(f)['boxes']
        labels.append(data)

# Создание датасета с путями файлов и bboxes
val_dataset = zip(images, labels)


# ### Получение метрик

# In[15]:


metrics = evaluate_detection(best_model, val_dataset)


# In[16]:


print(f'Precision для модели YOLO: {metrics["precision"]:.3f}')
print(f'Recall для модели YOLO: {metrics["recall"]:.3f}')
print(f'IoU для модели YOLO: {metrics["mean_iou"]:.3f}')
print(f'Dice для модели YOLO: {metrics["mean_dice"]:.3f}')


# <a id=7></a>
# # 7. Визуализация детекции

# In[17]:


# Берем 4 случайные изображения
vis_dataset = random.choices(list(zip(images, labels)), k=4)


# In[18]:


fig, axs = plt.subplots(2, 2, figsize=(9, 9))

# Инференс каждого случайного изображения
for ax, (img_path, gt_boxes) in zip(axs.ravel(), vis_dataset):

    # Визуализируем изображение
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR_RGB)
    ax.imshow(img)

    # Запускаем инференс YOLO для одного изображения
    result = model(img_path, verbose=False)[0]
    # Предсказанные боксы в формате xyxy
    pred_boxes = result.boxes.xyxy.cpu()

    # Отображаем bbox фактические
    for i, box in enumerate(gt_boxes):
        x1, y1, x2, y2 = box
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

    first = True
    # Отображаем bbox прдесказанные
    for i, box in enumerate(pred_boxes):
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

plt.suptitle('Визуализация детекции от YOLO')
plt.legend()
plt.show()


# <a id=8></a>
# # 8. Вывод
# 
# В рамках работы была обучена модель детекции объектов на основе `Ultralytics YOLO` с использованием пользовательского датасета. Были выполнены ключевые этапы: конвертация разметки в YOLO-формат, разбиение данных на обучающую и валидационную выборки, настройка конфигурации и дообучение предобученной модели.
# 
# По результатам обучения модель продемонстрировала хорошие показатели качества: высокая точность **(Precision ≈ 0.845)**, полнота **(Recall ≈ 0.732)**, а также высокие значения **IoU** и **Dice**, что свидетельствует о качественном совпадении предсказанных и истинных границ объектов. Визуальный анализ также подтвердил корректность работы модели.
