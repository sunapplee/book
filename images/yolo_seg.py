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
#     * [1.4 Расчет метрик](#1-5)
#     * [1.5 Визуализация предсказаний](#1-6)
# * [2. Сегментация набора из Модуля С](#2)

# ###

# <a id=0></a>
# ## Импорт библиотек

# In[1]:


get_ipython().system('pip install ultralytics -q')


# In[2]:


from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import io
# import segmentation_models_pytorch as smp
import torch
from torch import optim, nn
import torchvision.transforms.functional as F
# import segmentation_models_pytorch.metrics.functional as metrics
from torchvision.models import segmentation

import numpy as np
# import pandas as pd

import cv2

from ultralytics.data.converter import convert_segment_masks_to_yolo_seg

import os
from pathlib import Path
import shutil

# import plotly.express as px
import matplotlib.pyplot as plt

from tqdm import tqdm


# In[4]:


from google.colab import drive
drive.mount('/content/drive', force_remount=True)


# ###

# <a id=1></a>
# # 1. Сегментация изображений «с учителем»

# Разделим данные на **обучающую, валидационную и тестовую** выборки.
# 
# - **Обучающая (70%)** — для качественного обучения модели. Этого объема достаточно, чтобы модель смогла "выучить" закономерности в данных.
# - **Валидационная (15%)** — для настройки гиперпараметров и контроля переобучения.
# - **Тестовая (15%)** — для финальной объективной оценки качества модели на новых данных.

# Рассмотрим ```1``` модель сегментации.
# 
# 1) **YOLO26n-seg** — модель объединяет детектирование и сегментацию масок в реальном времени. Отличается высокой скоростью и применима в задачах, где важна производительность.

# <a id=1-3></a>
# ## 1.3 Обучение модели

# ### 1. YOLO26n-seg

# #### Подготовка данных
# 
# Текущая структура изображений:
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

# Создим новую папку `preprocess_images_flat/`, внутри которой будут чистые `images/` и `labels/` без **вложенности**.

# In[6]:


get_ipython().system('cp -r drive/MyDrive/Консист/atomskills2026/digitalskills2023/pr_images preprocess_images')


# In[7]:


import os
import cv2
from tqdm import tqdm

src_root = "preprocess_images"
dst_root = "preprocess_images_flat"

os.makedirs(os.path.join(dst_root, "images"), exist_ok=True)
os.makedirs(os.path.join(dst_root, "labels"), exist_ok=True)

for folder in ["images", "labels"]:
    src_base = os.path.join(src_root, folder)
    dst_base = os.path.join(dst_root, folder)

    for sub in tqdm(os.listdir(src_base)):
        subpath = os.path.join(src_base, sub)
        if os.path.isdir(subpath) and not sub.startswith('.'):

            for file in os.listdir(subpath):
                src = os.path.join(subpath, file)

                # пропускаем вложенные директории
                if os.path.isdir(src):
                    continue

                # определяем путь назначения
                dst = os.path.join(dst_base, file)

                # если это маска → привести имя к *.jpg без _mask
                is_mask = file.endswith('_mask.jpg')
                if is_mask:
                    dst = dst[:-9] + '.png'   # убираем "_mask"

                # если файл уже существует → пропускаем
                if os.path.exists(dst):
                    continue

                # ---- читаем изображение ----
                img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
                if img is None:
                    print(f"[WARNING] Нет файла: {src}")
                    continue

                # Если это маска — бинаризуем
                if is_mask:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    binary = (gray > 127).astype('uint8')
                    binary = cv2.resize(binary, (224, 224), interpolation=cv2.INTER_NEAREST)
                    fixed = (binary >= 1).astype('uint8')
                    img = fixed  # перезаписываем


                # ---- сохраняем ----
                cv2.imwrite(dst, img)

print("Готово! Сохранено в preprocess_images_flat/")


# ```ultralytics``` принимает не фото маски, а ```txt-файл``` с текстовым описанием маски. С помощью метода ```convert_segment_masks_to_yolo_seg``` преобразуем маски в необходимый формат.

# In[14]:


convert_segment_masks_to_yolo_seg(
masks_dir="preprocess_images_flat/labels/",
output_dir="preprocess_images_flat/yolo_labels/",
classes=2,
)


# In[15]:


import random


# In[16]:


# ==== НАСТРОЙКИ =====
IMAGES_DIR = "preprocess_images_flat/images/"       # где лежат изображения
LABELS_DIR = "preprocess_images_flat/yolo_labels/"       # где лежат yolo txt
OUTPUT_DIR = "dataset/"      # куда создадим train/val/test

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# ====================

# создаём итоговые директории
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)

# список всех изображений
image_files = [f for f in os.listdir(IMAGES_DIR)
               if f.lower().endswith((".jpg", ".jpeg", ".png"))]

random.shuffle(image_files)

n_total = len(image_files)
n_train = int(n_total * TRAIN_RATIO)
n_val = int(n_total * VAL_RATIO)

train_files = image_files[:n_train]
val_files = image_files[n_train:n_train + n_val]
test_files = image_files[n_train + n_val:]

splits = {
    "train": train_files,
    "val": val_files,
    "test": test_files
}

def move_files(file_list, split_name):
    for img_name in file_list:
        label_name = os.path.splitext(img_name)[0] + ".txt"

        # пути
        img_src = os.path.join(IMAGES_DIR, img_name)
        label_src = os.path.join(LABELS_DIR, label_name)

        img_dst = os.path.join(OUTPUT_DIR, "images", split_name, img_name)
        label_dst = os.path.join(OUTPUT_DIR, "labels", split_name, label_name)

        # копируем изображение
        shutil.copy(img_src, img_dst)

        # если нет метки — создаём пустой txt
        if os.path.exists(label_src):
            shutil.copy(label_src, label_dst)
        else:
            open(label_dst, "w").close()  # пустой label

for split, files in splits.items():
    print(f"✔ Копируем {split}: {len(files)} файлов")
    move_files(files, split)

print("Готово! Датасет разделён.")


# In[18]:


cat data.yaml


# In[19]:


from ultralytics import YOLO

# Загружаем предобученную модель
model = YOLO("yolo26n-seg.pt")

# Запуск обучения
results = model.train(
    data="data.yaml",     # путь к yaml
    epochs=1,           # количество эпох
    imgsz=224,            # размер картинки
    batch=8,              # размер батча
    workers=0,
    device="mps",
    mosaic=0.0, # Отключение аугментаций
    copy_paste=0.0,
    erasing=0.0,
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
    mask_ratio=2
)


# ####

# <a id=1-5></a>
# ## 1.4 Расчет метрик

# Загрузим тестовый датасет.

# In[26]:


import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class YOLOSegDataset(Dataset):
    def __init__(self, root, img_dir="images/test", lbl_dir="labels/test", img_size=640):
        self.root = root
        self.img_dir = os.path.join(root, img_dir)
        self.lbl_dir = os.path.join(root, lbl_dir)
        self.img_paths = sorted(os.listdir(self.img_dir))
        self.img_size = img_size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_name = self.img_paths[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]

        # ------ load polygons ------
        txt_path = os.path.join(
            self.lbl_dir,
            img_name.replace(".jpg", ".txt").replace(".png", ".txt")
        )

        mask = np.zeros((h, w), dtype=np.uint8)

        if os.path.exists(txt_path):
            with open(txt_path, "r") as f:
                for line in f:
                    cls, *coords = map(float, line.strip().split())

                    xs = coords[0::2]
                    ys = coords[1::2]

                    # denormalize
                    xs = (np.array(xs) * w).astype(np.int32)
                    ys = (np.array(ys) * h).astype(np.int32)

                    poly = np.stack([xs, ys], axis=1)

                    # draw polygon
                    cv2.fillPoly(mask, [poly], 1)

        # ------ resize ------
        img = cv2.resize(img, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        img = torch.tensor(img).permute(2, 0, 1).float() / 255.
        mask = torch.tensor(mask).long()

        return img, mask


# In[27]:


dataset = YOLOSegDataset(root="dataset")
test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)


# Выполним расчет индексов ```IoU``` и ```Dice``` для тестового набора данных.

# Метрику ```IoU``` импортируем из библиотеки ```segmentation_models_pytorch```.
# 
# Метрику ```Dice``` получим из формулы ниже:
# 
# ![image.png](attachment:cf5ff75a-ea5e-40e4-8bc6-bbb12c97814b.png)

# In[38]:


from ultralytics import YOLO

model = YOLO("yolo26n-seg.pt")

all_tp, all_fp, all_fn, all_tn = [], [], [], []

for imgs, masks_gt in tqdm(test_loader):
    imgs = imgs
    masks_gt = masks_gt

    with torch.no_grad():
        preds = model(imgs, verbose=False)[0]

    if preds.masks is None:
        pred_mask = torch.zeros_like(masks_gt)
    else:
        inst = preds.masks.data
        pred_mask = (inst.sigmoid() > 0.5).any(dim=0).long()

    gt = masks_gt.bool()
    pm = pred_mask.bool()

    tp = (pm & gt).sum()
    fp = (pm & ~gt).sum()
    fn = (~pm & gt).sum()
    tn = (~pm & ~gt).sum()

    all_tp.append(tp)
    all_fp.append(fp)
    all_fn.append(fn)
    all_tn.append(tn)

TP = torch.stack(all_tp).sum().float()
FP = torch.stack(all_fp).sum().float()
FN = torch.stack(all_fn).sum().float()
TN = torch.stack(all_tn).sum().float()

iou = TP / (TP + FP + FN + 1e-6)
dice = (2 * TP) / (2 * TP + FP + FN + 1e-6)

print()
print("IoU:", iou.item())
print("Dice:", dice.item())


# <a id=1-6></a>
# ## 1.5 Визуализация предсказаний
# 

# In[40]:


import random
import matplotlib.pyplot as plt
import torch

# Сколько примеров показать
N = 3

# случайные индексы из датасета
idxs = random.sample(range(len(dataset)), N)

plt.figure(figsize=(12, 4 * N))

for i, idx in enumerate(idxs):
    img, gt_mask = dataset[idx]

    # готовим изображение (обратно к numpy)
    img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')

    # предсказание
    with torch.no_grad():
        pred = model(img.unsqueeze(0), verbose=False)[0]

    if pred.masks is None:
        pred_mask = torch.zeros_like(gt_mask)
    else:
        inst = pred.masks.data
        pred_mask = (inst.sigmoid() > 0.5).any(dim=0).cpu().long()

    # --- РИСУЕМ ---
    # Оригинал
    plt.subplot(N, 3, i*3 + 1)
    plt.imshow(img_np)
    plt.title("Оригинал")
    plt.axis("off")

    # GT маска
    plt.subplot(N, 3, i*3 + 2)
    plt.imshow(gt_mask.cpu(), cmap="gray")
    plt.title("GT Маска")
    plt.axis("off")

    # Предсказанная маска
    plt.subplot(N, 3, i*3 + 3)
    plt.imshow(pred_mask.cpu(), cmap="gray")
    plt.title("Предсказанная маска")
    plt.axis("off")

plt.tight_layout()
plt.show()

