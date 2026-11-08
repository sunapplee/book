#!/usr/bin/env python
# coding: utf-8

# #  Полный отчёт: Работа с изображениями в задачах искусственного интеллекта  
# *Для подготовки к чемпионату  (на примере датасета «Dog Breed Image Dataset»)*
# 

# ## 🔹 Способы организации датасета
# 
# Перед тем как приступить к загрузке и обработке изображений, важно понять, **как организованы данные**. От этого зависит способ их считывания и дальнейшей обработки. Существует два основных подхода, которые применяются на практике.
# 

# ###  Способ 1: Каждая папка — отдельный класс (основной формат)
# 
# Это **наиболее распространённый и рекомендуемый способ** организации размеченных изображений. 
# 
# **Структура файловой системы выглядит следующим образом:**
# ```
# dog_breeds/
# ├── Labrador/
# │   ├── 001.jpg
# │   ├── 002.jpg
# │   └── ...
# ├── Beagle/
# │   ├── 101.jpg
# │   └── ...
# ├── Poodle/
# │   └── ...
# └── ...
# ```
# 
# В этой структуре:
# - Название каждой подпапки (например, `Labrador`) **является меткой класса**.
# - Все изображения, находящиеся внутри этой подпапки, **относятся к данному классу**.
# - Нет необходимости хранить отдельный файл с разметкой — структура папок сама по себе содержит всю необходимую информацию.
# 
# **Преимущества этого подхода:**
# - Простота и интуитивная понятность.
# - Прямая поддержка ведущими библиотеками машинного обучения, такими как PyTorch (`torchvision.datasets.ImageFolder`), TensorFlow (`tf.keras.utils.image_dataset_from_directory`) и YOLO (Ultralytics).
# - Минимальный риск ошибок при загрузке данных.
# 

# ###  Способ 2: Метка класса указана в имени файла
# 
# В некоторых случаях, особенно в устаревших или кастомных датасетах, все изображения могут находиться в **одной общей папке**, а метка класса **кодируется непосредственно в имени файла**.
# 
# **Пример структуры:**
# ```
# images/
# ├── Labrador_001.jpg
# ├── Beagle_002.jpg
# ├── Poodle_003.jpg
# └── ...
# ```
# 
# Здесь:
# - Класс (например, `Labrador`) является **первой частью имени файла**, отделённой от остального идентификатора символом подчёркивания `_`.
# - Чтобы определить метку, необходимо **проанализировать имя файла** и извлечь из него соответствующую часть.
# 

# In[1]:


import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Для CNN (нейросети)
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms


# ## 🔹  Загрузка данных

# ### Загрузка по папкам (основной способ)
# Эта функция автоматически определяет класс по названию папки и не требует ручного анализа имён файлов

# In[3]:


#(класс = имя папки)

DATASET_DIR = Path("dataset")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

rows = []
for class_dir in sorted([p for p in DATASET_DIR.iterdir() if p.is_dir()]):
    label = class_dir.name
    for img_path in class_dir.rglob("*"):
        if img_path.suffix.lower() in IMG_EXTS:
            rows.append({"path": str(img_path), "label": label})

df = pd.DataFrame(rows)
print("Всего изображений:", len(df))
print("Классы:", df["label"].unique())
print(df["label"].value_counts())


# ### Загрузка по именам файлов
# Для менее распространённого формата, где метка содержится в имени файла, реализуем отдельную функцию:

# In[5]:


DATASET_DIR_FLAT = Path("dataset")  

def label_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    return re.split(r"[_-]", stem)[0]

rows2 = []
for img_path in DATASET_DIR_FLAT.rglob("*"):
    if img_path.is_file() and img_path.suffix.lower() in IMG_EXTS:
        rows2.append({"path": str(img_path), "label": label_from_filename(img_path.name)})

df2 = pd.DataFrame(rows2)
print("Всего изображений:", len(df2))
print("Классы:", df2["label"].unique())
print(df2["label"].value_counts())


# ## 🔹  Предобработка изображений
# 
# После загрузки изображение необходимо **подготовить к подаче в модель**. Все изображения должны быть:
# 
# - Преобразованы в единый цветовой формат (обычно RGB),
# - Приведены к одинаковому размеру (например, 224×224 пикселей),
# - Нормализованы (значения пикселей приведены к диапазону [0, 1] или [-1, 1]).
# 
# Это необходимо, потому что нейронные сети **не могут работать с изображениями разного размера**, а нормализация **ускоряет и стабилизирует обучение**.
# 

# In[6]:


# функция предобработки в numpy (RGB + resize + нормализация)

TARGET_SIZE = (128, 128)

def preprocess_pil_to_numpy(img_path: str, target_size=TARGET_SIZE) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img, dtype=np.float32) / 255.0  # нормализация в [0, 1]
    return arr  # shape: (H, W, 3)

# тест
sample_path = df["path"].iloc[0]
x = preprocess_pil_to_numpy(sample_path)


# In[7]:


#transforms для PyTorch

torch_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(), 
    # Можно добавить нормализацию mean/std, но для простого старта достаточно ToTensor()
])


# ## 🔹  Балансировка классов (чтобы “примеров было +- поровну”)
# 
# В реальных датасетах часто встречаются **дисбалансированные классы**: одни породы представлены сотнями изображений, другие — всего десятком. Это приводит к тому, что модель **«учится игнорировать» редкие классы**.

# ###  Для RandomForest (sklearn):
# 
# Добавьте параметр `class_weight="balanced"` при создании модели — это автоматически взвесит ошибки на редких классах сильнее.

# In[8]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)


# ### Балансировка “чтобы поровну” = WeightedRandomSampler
# Чтобы классы встречались примерно одинаково в обучении — делай семплинг через WeightedRandomSampler.

# In[9]:


import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler

# 1. Определяем трансформации
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(root="dataset", transform=train_transform)

# 3. Балансировка через семплер
targets = torch.tensor(dataset.targets) 
class_count = torch.bincount(targets)
class_weight = 1.0 / class_count.float()
sample_weight = class_weight[targets]

sampler = WeightedRandomSampler(
    weights=sample_weight,
    num_samples=len(dataset),
    replacement=True
)

# 4. Создаём DataLoader
train_loader = DataLoader(dataset, batch_size=32, sampler=sampler, num_workers=2)

print(f" Датасет загружен: {len(dataset)} изображений, {len(dataset.classes)} классов")
print(f" DataLoader с балансировкой готов!")


# ### Для YOLOv8-cls (Ultralytics)
# В YOLOv8 классификация обычно “ест” структуру папок, а балансировку чаще решают подготовкой данных (оверсэмплинг редких классов копированием/доп-аугментациями) или своими загрузчиками, потому что “скейлер” как в sklearn там не используется — нормализация/препроцессинг спрятаны внутри пайплайна.
# 

# ## 🔹 Аугментация изображений 
# 
# Аугментация — это **ключевой приём** в компьютерном зрении, который позволяет искусственно увеличить объём обучающих данных и повысить устойчивость модели к вариациям во входных данных.
# 
# Цель аугментации — **смоделировать различные условия съёмки**, такие как:
# - Другой ракурс (поворот, отражение),
# - Изменённое освещение (яркость, контраст),
# - Частичное закрытие объекта (случайное стирание),
# - Отсутствие цветовой информации (чёрно-белый режим).
# 
# Это помогает модели **не переобучаться** на конкретные примеры, а **научиться распознавать суть объекта**.
# 
# ### Основные виды аугментации
# 
# | Тип аугментации | Описание | Пример использования |
# |------------------|--------|----------------------|
# | **Горизонтальное отражение** | Зеркальное отображение изображения по вертикальной оси. | Полезно для симметричных объектов (лица, собаки, автомобили). |
# | **Поворот** | Поворот изображения на случайный угол в заданном диапазоне. | Имитирует изменение угла съёмки. |
# | **Изменение цвета** | Случайное изменение яркости, контраста, насыщенности и оттенка. | Моделирует разные условия освещения. |
# | **Чёрно-белый режим** | Случайное преобразование изображения в градации серого. | Повышает устойчивость к цветовым артефактам. |
# | **Случайное стирание (Random Erasing)** | Закрашивание случайного прямоугольного участка изображения чёрным цветом. | Имитирует потерю части объекта (например, за кустом). |
# 
# ### Полный набор трансформаций для обучения
# 
# Для реализации аугментации мы будем использовать библиотеку `torchvision.transforms`, которая предоставляет удобные и эффективные инструменты.
# 
# # Получаем в итоге большой расширенный датасэт dataset_aug

# In[16]:


from pathlib import Path
import shutil
import random
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split

TARGET_SIZE = (128, 128)

DATASET_DIR = Path("dataset")
TRAIN_AUG_DIR = Path("dataset_train_aug")   # train = оригиналы train + их аугменты
TEST_CLEAN_DIR = Path("dataset_test_clean") # test = только оригиналы test
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 1) Индексация оригиналов
rows = []
for class_dir in sorted([p for p in DATASET_DIR.iterdir() if p.is_dir()]):
    label = class_dir.name
    for img_path in class_dir.rglob("*"):
        if img_path.suffix.lower() in IMG_EXTS:
            rows.append({"path": str(img_path), "label": label})
df = pd.DataFrame(rows)

# 2) Split оригиналов
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"]
)

def reset_dir(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

def copy_split(df_part: pd.DataFrame, out_root: Path):
    for _, row in df_part.iterrows():
        src = Path(row["path"])
        label = row["label"]
        (out_root / label).mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out_root / label / src.name)

reset_dir(TRAIN_AUG_DIR)
reset_dir(TEST_CLEAN_DIR)

# 3) Кладём оригиналы: train -> TRAIN_AUG_DIR, test -> TEST_CLEAN_DIR
copy_split(train_df, TRAIN_AUG_DIR)
copy_split(test_df, TEST_CLEAN_DIR)

# 4) Аугментации (ТОЛЬКО ДЛЯ TRAIN)
augment = transforms.Compose([
    transforms.Resize(TARGET_SIZE),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
    transforms.RandomGrayscale(p=0.25),  # 25% будут ч/б
])

def add_black_square(pil_img: Image.Image, max_frac=0.35) -> Image.Image:
    img = pil_img.copy()
    w, h = img.size
    rw = int(w * random.uniform(0.1, max_frac))
    rh = int(h * random.uniform(0.1, max_frac))
    x0 = random.randint(0, max(0, w - rw))
    y0 = random.randint(0, max(0, h - rh))
    arr = np.array(img)
    arr[y0:y0+rh, x0:x0+rw, :] = 0
    return Image.fromarray(arr)

def augment_train_only(train_df: pd.DataFrame, copies_per_image=2):
    for _, row in train_df.iterrows():
        src_path = Path(row["path"])
        label = row["label"]
        out_class_dir = TRAIN_AUG_DIR / label

        for k in range(copies_per_image):
            img = Image.open(src_path).convert("RGB")
            img = augment(img)
            img = add_black_square(img)
            img.save(out_class_dir / f"{src_path.stem}_aug{k}{src_path.suffix}")

augment_train_only(train_df, copies_per_image=2)

print("Train (with aug):", TRAIN_AUG_DIR)
print("Test (clean):", TEST_CLEAN_DIR)
print("Train orig:", len(train_df), "Test:", len(test_df))


# In[11]:


# индексируем расширенный датасет

rows_aug = []
for class_dir in sorted([p for p in TRAIN_AUG_DIR.iterdir() if p.is_dir()]):
    label = class_dir.name
    for img_path in class_dir.rglob("*"):
        if img_path.suffix.lower() in IMG_EXTS:
            rows_aug.append({"path": str(img_path), "label": label})

df_aug = pd.DataFrame(rows_aug)
print("Всего изображений после аугментации:", len(df_aug))
print(df_aug["label"].value_counts())


# ## 🔹 Train/test split + балансировка классов
# Разделение (stratify)
# Зачем: чтобы доли классов в train/test были похожи.
# 
# Балансировка для обучения
# Есть 2 популярных подхода:
# 
# WeightedRandomSampler (для DataLoader в PyTorch) — при обучении чаще подбирать редкие классы.
# 
# class_weight (для sklearn моделей) — штрафовать ошибки по редким классам сильнее.

# In[12]:


train_df, test_df = train_test_split(
    df_aug,
    test_size=0.2,
    random_state=42,
    stratify=df_aug["label"]
)

print("Train:", len(train_df), "Test:", len(test_df))
print("Train class counts:\n", train_df["label"].value_counts())
print("Test class counts:\n", test_df["label"].value_counts())


# ## 🔹  Популярные модели для задач компьютерного зрения
# 
# | Задача | Популярные модели | Когда использовать |
# |--------|------------------|-------------------|
# | **Классификация** | ResNet, EfficientNet, Vision Transformer (ViT), **YOLOv8-cls** | Определение объекта на всём изображении |
# | **Обнаружение объектов** | YOLOv8, Faster R-CNN, SSD | Нужны координаты и класс каждого объекта |
# | **Сегментация** | U-Net, Mask R-CNN, YOLOv8-seg | Требуется точное выделение границ объекта |
# | **Генерация изображений** | Stable Diffusion, GAN | Создание новых изображений |
# | **Повышение разрешения** | ESRGAN, SRCNN | Улучшение качества изображений |
# 
# > Для задач классификации на чемпионате **рекомендуется начинать с YOLOv8-cls или EfficientNet-B0**.
# 

# ### 🔹 Модель 1: RandomForest по «числам» (пиксели -> вектор)
# Идея: превратить картинку 
# 128×128×3 в вектор длины 128∗128∗3 и обучить RandomForest. 

# In[27]:


# Ячейка 5A: кодируем классы числами
classes = sorted(train_df["label"].unique())
class_to_id = {c: i for i, c in enumerate(classes)}
id_to_class = {i: c for c, i in class_to_id.items()}

train_df = train_df.copy()
test_df = test_df.copy()
train_df["y"] = train_df["label"].map(class_to_id)
test_df["y"] = test_df["label"].map(class_to_id)

print(class_to_id)


# In[28]:


# делаем признаки для sklearn 

def build_X_y(df_in: pd.DataFrame, target_size=TARGET_SIZE):
    X_list, y_list = [], []
    for _, row in df_in.iterrows():
        arr = preprocess_pil_to_numpy(row["path"], target_size=target_size)  # (H,W,3) float32 [0..1]
        feat = arr.reshape(-1) 
        X_list.append(feat)
        y_list.append(int(row["y"]))
    return np.stack(X_list), np.array(y_list)

X_train, y_train = build_X_y(train_df)
X_test, y_test = build_X_y(test_df)

print(X_train.shape, y_train.shape)


# In[29]:


# RandomForest обучение

rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"  # балансировка внутри модели
)
rf.fit(X_train, y_train)

pred_rf = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred_rf))
print("F1 macro:", f1_score(y_test, pred_rf, average="macro"))
print(classification_report(y_test, pred_rf, target_names=classes))


# ## 🔹 Модель 2: Logistic Regression (ещё один baseline по числам)

# In[30]:


#  LogisticRegression (часто сильнее, чем кажется, но требует больше итераций)

lr = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    n_jobs=None
)
lr.fit(X_train, y_train)

pred_lr = lr.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred_lr))
print("F1 macro:", f1_score(y_test, pred_lr, average="macro"))
print(classification_report(y_test, pred_lr, target_names=classes))


# ## 🔹 YOLO для классификации (YOLO-cls)
# 

# Подготовка структуры из dataset_aug
# Сейчас: dataset_aug/Beagle/*.jpg, dataset_aug/Boxer/*.jpg и т.п.
# YOLO-cls ожидает так:
# dataset_yolo_cls/train/<class>/*.jpg и dataset_yolo_cls/val/<class>/*.jpg (и опционально test).
# ​

# In[15]:


import shutil
from pathlib import Path

SRC_TRAIN = Path("dataset_train_aug")
SRC_VAL = Path("dataset_test_clean")   # чистая проверка
DST = Path("dataset_yolo_cls")

TRAIN_DIR = DST / "train"
VAL_DIR = DST / "val"

if DST.exists():
    shutil.rmtree(DST)
TRAIN_DIR.mkdir(parents=True, exist_ok=True)
VAL_DIR.mkdir(parents=True, exist_ok=True)

def copy_tree(src_root: Path, dst_root: Path):
    for class_dir in src_root.iterdir():
        if not class_dir.is_dir():
            continue
        out = dst_root / class_dir.name
        out.mkdir(parents=True, exist_ok=True)
        for p in class_dir.glob("*"):
            if p.is_file():
                shutil.copy2(p, out / p.name)

copy_tree(SRC_TRAIN, TRAIN_DIR)
copy_tree(SRC_VAL, VAL_DIR)

print("YOLO dataset ready:", DST)


# Установка и проверка ultralytics
# YOLO от Ultralytics тренируется через from ultralytics import YOLO и поддерживает режим классификации с yolo11n-cls.pt

# In[16]:


get_ipython().system('pip install -U ultralytics')


# In[17]:


from ultralytics import YOLO


# Количество эпох — это сколько раз модель “пройдёт” по всему train‑набору, но в Ultralytics обычно ставят epochs с запасом и используют `patience` (early stopping), чтобы обучение остановилось, когда метрики на val перестали улучшаться.
# 
# ## Сколько эпох ставить  (практично)
# - Для быстрого чернового прогона: **10–30 эпох** (чтобы проверить, что всё работает и метрики растут).
# - Для “нормального” обучения: часто ставят **100–300 эпох**, а остановку доверяют `patience`, чтобы не тратить лишнее время и не уйти в переобучение.
# - Если датасет маленький/простой, модель может выйти на плато быстро — тогда `patience` (например 10–20) сработает раньше, и реальные эпохи будут меньше заданных.
# 
# ## Рекомендованные настройки прямо в коде
# 1) “Быстро и безопасно”:
# ```python
# results = model.train(
#     data="dataset_yolo_cls",
#     epochs=50,
#     patience=10,   # остановится, если нет улучшения 10 эпох 
#     imgsz=224,
#     batch=32
# )
# ```
# 
# 2) “С запасом, пусть сама остановится”:
# ```python
# results = model.train(
#     data="dataset_yolo_cls",
#     epochs=300,    # частый стартовый ориентир 
#     patience=20,
#     imgsz=224,
#     batch=32
# )
# ```
# 

# In[22]:


from ultralytics import YOLO
from pathlib import Path

DST = Path("dataset_yolo_cls")

model = YOLO("yolo11n-cls.pt")  # предобученная модель для классификации 

results = model.train(
    data=str(DST),     # корень, где лежат train/val
    epochs=10,
    imgsz=224,
    batch=32,
    patience=10        # ранняя остановка, если метрика не улучшается
)
model.save('dogs_yolo_model.pt')


# In[23]:


val_res = model.val(data=str(DST))
val_res


# Что означает каждая метрика:
# Accuracy: доля правильных предсказаний (top‑1 попадание).
# 
# Top‑5 accuracy: правильный класс в топ‑5 по вероятности (важно при большом числе классов).
# ​
# 
# Precision (macro): “когда модель сказала класс, насколько часто права”, усреднение по классам без учёта размера классов.
# 
# Recall (macro): “сколько объектов каждого класса модель нашла”, усреднение по классам.
# 
# F1 (macro): баланс precision и recall, честнее при дисбалансе.
# 
# Balanced accuracy: средний recall по классам (полезно при дисбалансе).
# 
# Confusion matrix: какие классы с какими путаются.
# 
# ROC-AUC (ovr): насколько хорошо модель ранжирует правильный класс выше неправильных по вероятностям (нужны вероятности, а не только метки).

# In[24]:


import numpy as np
from pathlib import Path

VAL_DIR = Path("dataset_yolo_cls/val")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# список файлов и истинные метки (из папки)
val_paths = []
y_true_names = []

for class_dir in sorted([p for p in VAL_DIR.iterdir() if p.is_dir()]):
    for p in class_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            val_paths.append(str(p))
            y_true_names.append(class_dir.name)

# делаем предсказания батчами (Ultralytics сам батчит)
pred_results = model.predict(source=val_paths, imgsz=224, verbose=False)

# превращаем в id-метки
name_to_id = {v: int(k) for k, v in model.names.items()}  # model.names: id->name 
n_classes = len(model.names)

y_true = np.array([name_to_id[n] for n in y_true_names], dtype=int)

y_pred = []
y_proba = []

for r in pred_results:
    # probs содержит вероятности классов + top1/top5 и confidence 
    probs = r.probs
    y_pred.append(int(probs.top1))                 # id top-1 класса 
    y_proba.append(probs.data.cpu().numpy())       # полный вектор вероятностей 

y_pred = np.array(y_pred, dtype=int)
y_proba = np.stack(y_proba)

print("val samples:", len(y_true), "classes:", n_classes)


# In[26]:


from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    top_k_accuracy_score, roc_auc_score
)

print("Accuracy (top-1):", accuracy_score(y_true, y_pred))
print("Balanced accuracy:", balanced_accuracy_score(y_true, y_pred))
print("Precision macro:", precision_score(y_true, y_pred, average="macro", zero_division=0))
print("Recall macro:", recall_score(y_true, y_pred, average="macro", zero_division=0))
print("F1 macro:", f1_score(y_true, y_pred, average="macro", zero_division=0))
print("F1 weighted:", f1_score(y_true, y_pred, average="weighted", zero_division=0))

print("Top-5 accuracy:", top_k_accuracy_score(y_true, y_proba, k=min(5, y_proba.shape[1])))

# ROC-AUC для мультикласса 
try:
    print("ROC-AUC ovr macro:", roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
    print("ROC-AUC ovr weighted:", roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"))
except Exception as e:
    print("ROC-AUC error:", e)

print("\nConfusion matrix:\n", confusion_matrix(y_true, y_pred))
print("\nClassification report:\n", classification_report(y_true, y_pred, target_names=[model.names[i] for i in range(n_classes)]))

