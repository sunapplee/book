#!/usr/bin/env python
# coding: utf-8

# # **Аудиоклассификация: полный цикл обработки и моделирования**
# 
# В данном ноутбуке выполняется полный цикл анализа аудиоданных — от сырого сигнала до выбора финальной модели классификации. Рабочий процесс включает:
# 
# * **Загрузку и изучение аудиоданных**
# * **Предобработку**: шумоподавление, нормализацию, базовые преобразования
# * **Извлечение акустических признаков**: MFCC, Delta, Chroma, Spectral Centroid, Bandwidth, Rolloff и другие
# * **ML-бэйслайн (CatBoost)** на табличных фичах
# * **Генерацию Mel-спектрограмм**
# * **Обучение CNN-модели** на изображениях спектрограмм
# * **Применение аугментаций** для повышения обобщающей способности
# * **Сравнение моделей** и выбор наиболее эффективного решения
# 
# Ноутбук охватывает весь процесс построения аудиоклассификатора и позволяет оценить различия между классическими ML-подходами и нейросетевыми моделями.

# # Содержание
# * [Импорт библиотек](#0)
# * [1. Загрузка данных](#1)
# * [2. Предобработка данных](#2)
# * [3. Извлечение фичей и ML-бэйслайн](#3)
# * [4. Аугментация аудио](#4)
# * [5. Генерация Mel-спектрограмм](#5)
# * [6. Классификация спектрограмм ](#6)
# * [7. Сравнение моделей и выбор финального решения](#7)

# ###

# <a id=0></a>
# ## Импорт библиотек

# In[1]:


import librosa
import noisereduce as nr
import numpy as np
import pandas as pd
import random
import torch
from PIL import Image

import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.express as px

import os
from pathlib import Path
from tqdm import tqdm
from IPython.display import Audio
from joblib import Parallel, delayed

import catboost

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

from torchvision.datasets import ImageFolder
from torchvision import models
import torchvision.transforms.v2 as tfs
from torch.utils.data import Subset, DataLoader
from torch import nn
from torch import optim


# In[2]:


# Загружать модели будем сюда
import os
os.environ["TORCH_HOME"] = r"E:\torch-cache"


# ###

# <a id=1></a>
# # 1. Загрузка данных

# Данные представлены в директории `data_audio_pics/audio`, Загрузим их в словарь. 

# In[3]:


audio_dir = 'data_audio_pics/audio'
# Сюда сохраним данные
data_train = {}

# По каждому файлу
for p in tqdm(Path(f'data_audio_pics/audio/train').iterdir(), desc='train'):
    if p.stem.startswith('.'):
        continue

    class_name = p.stem
    for class_data in Path(p).iterdir():
        audio, sr = librosa.load(class_data, sr=None)
        data_train.setdefault(class_name, []).append(audio)


# Прослушаем пример аудио.

# In[4]:


Audio(data=data_train['0'][0], rate=22050)


# <a id=2></a>
# # 2. Предобработка данных 

# ## Шумоподавление

# Объявим функцию для удаления шума.

# In[5]:


def remove_noice(audio):
    return nr.reduce_noise(y=audio, sr=22050)


# Обработаем весь датасет.

# In[6]:


# Проходимся по всем обучающим данным
preprocessed_audio_train = {
            class_name: Parallel(n_jobs=-1)(
                delayed(remove_noice)(arr) for arr in data_train[class_name]
            )
        for class_name in tqdm(data_train, desc='train')
    }


# Аудио после шумоподавления:

# In[7]:


Audio(data=preprocessed_audio_train['0'][0], rate=22050)


# ## VAD (Voice Activity Detection)

# Моделью обнаружения речи выступит `Silero`. Загрузим ее.

# In[8]:


model, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad"
)
(get_speech_timestamps, _, read_audio, _, _) = utils


# Объявим функцию которая:
# * прогоняет через аудио **Silero VAD**
# 
# * вырезает только **участки с речью**
# 
# * возвращает очищенный аудиосигнал

# In[9]:


def detect_speech_audio(audio_22k) -> np.ndarray:

    # Ресемплируем 22050 -> 16000
    audio_16k = librosa.resample(audio_22k, orig_sr=22050, target_sr=16000)

    # Silero ожидает torch.Tensor
    wav_16k = torch.from_numpy(audio_16k).unsqueeze(0)

    # Получаем тайминги голосовых сегментов
    speech_ts = get_speech_timestamps(
        wav_16k,
        model,
        sampling_rate=16000,
        return_seconds=False
    )

    # Если речи нет — возвращаем нулевой массив исходной длины
    if len(speech_ts) == 0:
        return np.zeros_like(audio_22k, dtype=np.float32)

    # Создаём нулевой массив-маску
    mask_16k = np.zeros_like(audio_16k, dtype=np.float32)

    # На позициях start:end вставляем оригинальный сигнал
    for ts in speech_ts:
        s, e = ts["start"], ts["end"]
        mask_16k[s:e] = audio_16k[s:e]

    # Ресемплируем маску обратно к 22050 Hz
    mask_22k = librosa.resample(mask_16k, orig_sr=16000, target_sr=22050)

    return mask_22k.astype(np.float32)


# Обработаем весь датасет.

# In[10]:


# Проходимся по всем обучающим данным
preprocessed_audio_train = {
            class_name: Parallel(n_jobs=-1)(
                delayed(remove_noice)(arr) for arr in data_train[class_name]
            )
        for class_name in tqdm(preprocessed_audio_train, desc='train')
    }


# Аудио после детекции голоса:

# In[11]:


Audio(data=preprocessed_audio_train['0'][0], rate=22050)


# ### Сравнение звуковых сигналов после предобработки

# In[12]:


audio_data_original = data_train['0'][0]
audio_data_preprocess = preprocessed_audio_train['0'][0]

fig = make_subplots(rows=2, cols=1)

# Создаем временную ось
time = np.arange(len(audio_data_original)) / 22050

# ДО предобработки
fig_original = px.line(
    x=time,
    y=audio_data_original,
    labels={'x': 'Время (сек)', 'y': 'Амплитуда'},
)
fig.add_trace(fig_original.data[0], row=1, col=1)

# ПОСЛЕ предобработки
fig_pre = px.line(
    x=time,
    y=audio_data_preprocess,
    labels={'x': 'Время (сек)', 'y': 'Амплитуда'},
)
fig.add_trace(fig_pre.data[0], row=2, col=1)

fig.update_layout(height=600, title_text="Волновые формы аудиосигнала ДО/ПОСЛЕ предобработки")
fig.show()


# В результате предобработки мы сделали звук чище и убрали артефакты.

# <a id=3></a>
# 
# # 3. Извлечение фичей и ML-бэйслайн

# ### **MFCC**
# 
# *MFCC — коэффициенты мел-кепстра, описывают форму спектра и тембр.*

# In[13]:


def get_mfcc(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=32)
    mfcc_feat = mfcc.mean(axis=1)
    return mfcc_feat, mfcc


# ### **Delta MFCC**
# 
# *Delta — первая производная MFCC, характеризует скорость изменения тембра.*

# In[14]:


def get_delta1(y, sr):
    delta_mfcc = librosa.feature.delta(y)
    delta_feat = delta_mfcc.mean(axis=1)
    return delta_feat


# ### **Delta-Delta MFCC**
# 
# *Delta-Delta — вторая производная, отражает ускорение изменений MFCC.*

# In[15]:


def get_delta2(y, sr):
    delta2_mfcc = librosa.feature.delta(y, order=2)
    delta2_feat = delta2_mfcc.mean(axis=1)
    return delta2_feat


# ### **Chroma**
# 
# *Chroma — распределение энергии по 12 музыкальным высотам (полутонам).*
# 

# In[16]:


def get_chroma(y, sr):
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_feat = chroma.mean(axis=1)
    return chroma_feat


# ### **Spectral Contrast**
# 
# *Spectral Contrast — разница между пиками и впадинами спектра, отличает шумы/тональные звуки.*

# In[17]:


def get_contrast(y, sr):
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_feat = contrast.mean(axis=1)
    return contrast_feat


# ### **Spectral Centroid**
# 
# *Centroid — «центр масс» спектра, показывает насколько звук высокий или низкий.*

# In[18]:


def get_centroid(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    centroid_feat = centroid.mean(axis=1)
    return centroid_feat


# ### **Spectral Bandwidth**
# 
# *Bandwidth — ширина спектра, характеризует резкость/шумность звука.*

# In[19]:


def get_bandwidth(y, sr):
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    bandwidth_feat = bandwidth.mean(axis=1)
    return bandwidth_feat


# ### **Spectral Rolloff**
# 
# *Rolloff — частота, ниже которой сосредоточено 85% энергии, описывает наклон спектра.*

# In[20]:


def get_rolloff(y, sr):
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rolloff_feat = rolloff.mean(axis=1)
    return rolloff_feat


# ### Итоговая сборка вектора признаков

# Выносим обработку одного аудио в отдельную функцию.

# In[21]:


def extract_features_one(audio, class_name, sr=22050):
    feats = {}

    # MFCC
    mfcc, mfcc_raw = get_mfcc(audio, sr)
    for i, v in enumerate(mfcc):
        feats[f"mfcc_{i}"] = v

    # Delta
    mfcc_delta = get_delta1(mfcc_raw, sr)
    for i, v in enumerate(mfcc_delta):
        feats[f"mfcc_delta_{i}"] = v

    # Delta2
    mfcc_delta2 = get_delta2(mfcc_raw, sr)
    for i, v in enumerate(mfcc_delta2):
        feats[f"mfcc_delta2_{i}"] = v

    # Chroma
    chroma = get_chroma(audio, sr)
    for i, v in enumerate(chroma):
        feats[f"chroma_{i}"] = v

    # Centroid, Bandwidth, Rolloff
    feats["centroid"] = get_centroid(audio, sr)[0]
    feats["bandwidth"] = get_bandwidth(audio, sr)[0]
    feats["rolloff"] = get_rolloff(audio, sr)[0]

    # Contrast
    contrast = get_contrast(audio, sr)
    for i, v in enumerate(contrast):
        feats[f"contrast_{i}"] = v

    feats["target"] = class_name
    return feats


# Функция обработки аудио для обучающей выборки.

# In[22]:


def get_featured_dataset():

    # --- Обучающая выборка  ---
    train_rows = []
    for class_name, audios in preprocessed_audio_train.items():
        for audio in tqdm(audios, desc=f"class {class_name}", leave=True):
            row = extract_features_one(audio, class_name)
            train_rows.append(row)

    train_df = pd.DataFrame(train_rows)


    return train_df


# In[23]:


train_df = get_featured_dataset()


# In[24]:


train_df.head()


# ## Обучение модели
# 
# В качестве baseline-модели возьмём `CatBoost` — он даёт сильное качество «из коробки», хорошо подходит для табличных фичей и умеет обучаться на `GPU`, позволяя значительно сократить время обучения на больших датасетах.

# In[25]:


# Разделяем данные на обучающие и тестовые
X = train_df.drop(columns=["target"])
y = train_df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# In[26]:


# Инициализация модели
model = catboost.CatBoostClassifier(
    iterations=1000,
    depth=6,
    learning_rate=0.05,
    loss_function='MultiClass',
    task_type="GPU",          # <--- Включаем GPU
    devices='0',              # <--- если одна видеокарта
    verbose=100
)

# Обучение модели
model.fit(X_train, y_train)


# In[27]:


# Предсказания
y_pred = model.predict(X_test)

# Метрики
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", acc)
print("F1-score:", f1)
print("\nClassification report:\n")
print(classification_report(y_test, y_pred))


# <a id=4></a>
# 
# # 4. Аугментация аудио
# 
# 
# Для увеличения **разнообразия обучающей выборки** и повышения **устойчивости модели** к реальным условиям записи применяются различные методы *аудио-аугментации*.  
# Эти преобразования сохраняют смысловое содержание сигнала, но изменяют его характеристики.
# 

# In[28]:


# Оригинальное аудио
Audio(preprocessed_audio_train['0'][0], rate=22050)


# Ниже приведены основные методы, используемые при подготовке аудиоданных.

# #### Time Stretch
# Растягивает/сжимает аудио по времени (меняет темп).

# In[29]:


def aug_time_stretch(y, rate):
    return librosa.effects.time_stretch(y, rate=rate)


# In[30]:


# Аудио после аугментации

y_aug = aug_time_stretch(preprocessed_audio_train['0'][0], rate=1.2)   # ускорить на 20%
Audio(y_aug, rate=22050)


# #### Pitch Shift 
# 
# Смещает высоту тона вверх/вниз без изменения длины.

# In[31]:


def aug_pitch_shift(y, sr, n_steps):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)


# In[32]:


# Аудио после аугментации
y_aug = aug_pitch_shift(preprocessed_audio_train['0'][0], 22050, n_steps=+3)   # поднять на 3 полутона
Audio(y_aug, rate=22050)


# #### Time Masking
# 
# Закрывает случайный участок времени на спектрограмме.

# In[33]:


def aug_time_mask(y, time_width):
    S = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(S))
    phase = np.angle(S)

    Tmax = S_db.shape[1]
    if time_width < Tmax:
        t0 = np.random.randint(0, Tmax - time_width)
        S_db[:, t0:t0+time_width] = S_db.min()

    return librosa.istft(librosa.db_to_amplitude(S_db) * np.exp(1j * phase))


# In[34]:


# Аудио после аугментации
y_aug = aug_time_mask(preprocessed_audio_train['0'][0], time_width=40)   # закрыть 40 фреймов
Audio(y_aug, rate=22050)


# #### Frequency Masking
# 
# Закрывает случайный диапазон частот.

# In[35]:


def aug_freq_mask(y, freq_width):
    S = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(S))
    phase = np.angle(S)

    Fmax = S_db.shape[0]
    if freq_width < Fmax:
        f0 = np.random.randint(0, Fmax - freq_width)
        S_db[f0:f0+freq_width, :] = S_db.min()

    return librosa.istft(librosa.db_to_amplitude(S_db) * np.exp(1j * phase))


# In[36]:


# Аудио после аугментации
y_aug = aug_freq_mask(preprocessed_audio_train['0'][0], freq_width=20)   # закрыть 20 частотных бинов
Audio(y_aug, rate=22050)


# #### Add Noise
# 
# Добавляет белый шум к сигналу.

# In[37]:


def aug_add_noise(y, std):
    noise = np.random.normal(0, std, size=len(y))
    return y + noise


# In[38]:


# Аудио после аугментации
y_aug = aug_add_noise(preprocessed_audio_train['0'][0], std=0.02)   # добавить шум
Audio(y_aug, rate=22050)


# #### RIR (Room Impulse Response)
# 
# Добавляет эффект комнаты через свёртку с RIR.

# In[39]:


def aug_rir(y, sr, rir_path):
    rir, _ = librosa.load(rir_path, sr=sr)
    rir = rir / np.sqrt(np.sum(rir**2))  # нормализация энергии
    return np.convolve(y, rir, mode="full")[:len(y)]


# #### Volume Gain
# 
# Изменяет громкость (усиление/ослабление).

# In[40]:


def aug_volume(y, gain):
    return y * gain


# In[41]:


# Аудио после аугментации
y_aug = aug_volume(preprocessed_audio_train['0'][0], gain=1.3)   # увеличить громкость на 30%
Audio(y_aug, rate=22050)


# ### Аугментация датасета
# 
# Для демонстрации работы применим функции к одному классу датасета.

# In[42]:


test_train = preprocessed_audio_train.copy()


# In[43]:


for class_name in preprocessed_audio_train:
    # чтобы не менять список во время цикла
    original_audios = preprocessed_audio_train[class_name].copy()
    augmented_audios = []
    for audio in tqdm(preprocessed_audio_train[class_name], desc=f'Класс: {class_name}'):

        # случайно от -20% до +20%
        rate = random.uniform(0.8, 1.2)
        y_stretched = aug_time_stretch(audio, rate)

        # от -4 до +4 полутонов
        n_steps = random.randint(-4, 4)
        y_pitched = aug_pitch_shift(audio, sr, n_steps)

        # случайная ширина маски по времени
        time_width = random.randint(20, 60)
        y_time_masked = aug_time_mask(audio, time_width)

        # случайная ширина маски по частотам
        freq_width = random.randint(10, 40)
        y_freq_masked = aug_freq_mask(audio, freq_width)

        # случайный уровень шума от 0.005 до 0.03
        noise_std = random.uniform(0.005, 0.03)
        y_noisy = aug_add_noise(audio, noise_std)

        # случайное изменение громкости от -30% до +50%
        gain = random.uniform(0.7, 1.5)
        y_volume = aug_volume(audio, gain)

        # Сохраняем аугментированные аудио
        augmented_audios.extend([
            y_stretched,
            y_pitched,
            y_time_masked,
            y_freq_masked,
            y_noisy,
            y_volume
        ])

    # Добавляем аудио по классу
    preprocessed_audio_train[class_name].extend(augmented_audios)

    # Для демонстрации хватит 1 класса
    break


# In[44]:


print('Добавлено новых аугментаций:', len(augmented_audios))
print(f'Всего аудио в классе {class_name}: {len(preprocessed_audio_train[class_name])}')


# Как видим, алгоритм отработал отлично!

# <a id=5></a>
# 
# ## 5. Генерация Mel-спектрограмм
# 
# На этом этапе мы преобразуем аудиосигналы в **Mel-спектрограммы** — двумерные представления, которые показывают, как распределяется энергия по частотам во времени. Такое представление удобно тем, что его можно обрабатывать **как изображение**. В дальнейшем именно эти спектрограммы будут использоваться в качестве входных данных для обучения моделей классификации, что позволяет применять методы `компьютерного зрения`.
# 

# In[45]:


# Создаем папку с изображениями
os.makedirs('pics', exist_ok=True)

# Будем обрабатывать каждый класс
for class_name in preprocessed_audio_train:

    # Создаем папку с классом
    os.makedirs(f'pics/{class_name}', exist_ok=True)

    # Будем обрабатывать каждое аудио
    for i, audio in tqdm(enumerate(preprocessed_audio_train[class_name]), desc=f'Класс: {class_name}'):
        # Вычисляем mel-спектрограмму
        M = librosa.feature.melspectrogram(y=audio, sr=22050)
        # Переводим в ДБ
        M_db = librosa.power_to_db(M, ref=np.max)

        # Визуализируем спектрограмму
        plt.figure(figsize=(5, 5))
        librosa.display.specshow(M_db, sr=22050)
        plt.tight_layout()
        plt.axis('off')
        # Сохраняем график
        plt.savefig(f"pics/{class_name}/mel_{class_name}_{i}.png", dpi=300, bbox_inches='tight', pad_inches=0)
        # Закрываем фигуру
        plt.close()


# <a id=6></a>
# # 6. Классификация спектрограмм 
# 
# После подготовки Mel-спектрограмм переходим к этапу обучения модели. В качестве базового классификатора используем **ConvNeXt-Tiny** — современную сверточную архитектуру, которая показывает высокую точность на задачах компьютерного зрения и хорошо работает со спектрограммами как с изображениями. Мы загрузим предобученные веса, адаптируем финальный слой под число наших классов и дообучим модель на сформированном датасете спектрограмм.
# 

# ### Подготовка данных

# Загрузим наши изображения с помощью `ImageFolder`.

# In[46]:


# Объявляем трансформации
transform = tfs.Compose([
    tfs.Grayscale(num_output_channels=3),
    tfs.Resize((224, 224)),
    tfs.ToImage(),
    tfs.ToDtype(torch.float32, scale=True),
    tfs.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# In[47]:


pic_dataset = ImageFolder('pics', transform=transform)
num_classes = len(pic_dataset.classes)
pic_dataset


# Разделим данные на **обучающую**, **валидационную** и **тестовую** выборки. Для этого сохраним их индексы, а затем разделим наш основной `pic_dataset`.

# In[48]:


# Индексы датасета
indices = list(range(len(pic_dataset)))

# Делим выборку в пропорции 70/15/15
train_idx, temp_idx = train_test_split(
    indices, test_size=0.3, random_state=42
)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.5, random_state=42
)

# Создаем подвыборки
train_dataset = Subset(pic_dataset, train_idx)
val_dataset = Subset(pic_dataset, val_idx)
test_dataset = Subset(pic_dataset, test_idx)


# Инициализируем `DataLoader`.

# In[49]:


train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1)
test_loader = DataLoader(test_dataset, batch_size=1)


# In[50]:


print('Размер обучающей выборки:', len(train_dataset))
print('Размер валдиационной выборки:', len(val_dataset))
print('Размер тестовой выборки:', len(test_dataset))


# ### Инициализация модели

# In[51]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# In[52]:


# Загружаем модель, адаптируем последний слой под наши данные
model = models.convnext_tiny(weights='DEFAULT')
model.classifier[2] = nn.Linear(768, num_classes)
model = model.to(device)


# In[59]:


# Конфигурация
EPOCHS = 5
optimizer = optim.AdamW(params=model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


# ### Обучение модели

# Напишем функцию обучения модели.

# In[60]:


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_correct = 0
    for images, labels in tqdm(loader, desc="train"):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()

    accuracy = total_correct / len(loader.dataset)
    return total_loss, accuracy


# Функция валидации:

# In[61]:


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="val"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()

    accuracy = total_correct / len(loader.dataset)
    return total_loss, accuracy


# Запускаем обучение!

# In[62]:


best_val_acc = 0.0

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    # Обучение
    train_loss, train_acc = train_one_epoch(
        model, train_loader, optimizer, criterion, device
    )
    print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")

    # Валидация
    val_loss, val_acc = validate_one_epoch(
        model, val_loader, criterion, device
    )
    print(f"Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}")

    # Планировщик
    scheduler.step()

    # Сохранение лучшей модели
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model_convnext.pth")
        tqdm.write(f"Лучшая модель сохранена в best_model_convnext.pth (val_acc = {val_acc:.4f})")

    # Обновляем строку прогресса
    tqdm.write(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f} | "
        f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}"
    )


# ### Инференс

# Загрузим лучшую модель.

# In[63]:


model = models.convnext_tiny(weights='DEFAULT')
model.classifier[2] = nn.Linear(768, num_classes)
model.load_state_dict(torch.load('best_model_convnext.pth', weights_only=True))
model = model.to(device)


# Оценим метрику на тестовом датасете.

# In[64]:


with torch.no_grad():

    # Результаты
    y_true = []
    y_pred = []

    # Инференс теста
    for images, labels in tqdm(test_loader, desc='test'):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = outputs.argmax(dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Метрики
acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='weighted')
report = classification_report(y_true, y_pred)

print("Accuracy:", acc)
print("F1-score:", f1)
print("\nClassification report:\n")
print(report)


# ### Визуализация предсказаний

# In[68]:


data2vis = (
    ('pics/0/mel_0_0.png', 0),
    ('pics/3/mel_3_0.png', 3),
    ('pics/4/mel_4_0.png', 4),
)

fig, axs = plt.subplots(1, 3, figsize=(12, 12))

for i, ax in enumerate(axs.ravel()):

    # Загрузка картинки
    img_path, label = data2vis[i]
    img = Image.open(img_path).convert('RGB')

    # Обработка картинки для инференса
    transformed_img = transform(img)
    transformed_img = transformed_img.unsqueeze(0)
    transformed_img = transformed_img.to(device)

    # Предсказание модели
    output = model(transformed_img)
    pred = output.argmax(dim=1)

    # Визуализация
    ax.imshow(img, cmap='magma')
    ax.set_title(f'Реальный класс: {label}\nПредсказанный класс: {pred.item()}')
    ax.axis('off')


# <a id=7></a>
# # 7. Сравнение моделей и выбор финального решения
# 
# В ходе экспериментов были опробованы два подхода:
# 
# 1. **ML-бэйслайн** на табличных признаках (MFCC, Chroma, Spectral features) с моделью **CatBoost**,
# 2. **Классификация Mel-спектрограмм** с помощью **сверточной нейронной сети**.
# 
# Сравнение метрик показало, что **лучшее качество продемонстрировала модель на спектрограммах**, поскольку она учла более богатое представление аудиосигнала и смогла выявить сложные паттерн. 
