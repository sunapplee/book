#!/usr/bin/env python
# coding: utf-8

# # 🔹 Пункт 1: сегментация на звуки → визуализация → датасет → train/val/test
# 
# 
# пусть у нас есть длинный(на 32 минуты аудио файл audio.wav и mata.csv, которая коговорит, в каких отрезках времени какой класс)
# 

# 1. загрузка необходимых библиотек + загрузка и первоначальный анализ аудиофайла

# In[2]:


import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

AUDIO_PATH = "audio.wav"

y, sr = librosa.load(AUDIO_PATH, sr=None, mono=True)
duration = len(y) / sr

print("sr:", sr, "Гц") #частота дискретизации
print("длительность:", round(duration, 2), "сек") 
print("сэмплы:", len(y)) #кол-во сэмплов


# 2. построим график амплитуды звука

# In[3]:


plt.figure(figsize=(14, 4))
librosa.display.waveshow(y, sr=sr, alpha=0.7)
plt.tight_layout()
plt.show()


# 3. Подготовка сигнала к автоматической сегментации на звуки
# 
# 
# При анализе аудио librosa разбивает сигнал на короткие окна (фреймы) и сдвигает окно по времени с шагом hop_length (в сэмплах).
# 
# Если sr = 44100 и hop_length = 512, то один шаг по времени равен 
# 512/44100≈0.0116 секунды — примерно 11.6 миллисекунд.
# ​
# То есть onset‑энергия o_env — это последовательность значений через каждые ~11.6 мс. Чем меньше hop_length, тем точнее по времени, но тем больше точек и нагрузка на вычисления.
# ​
# 
#  - Как примерно выбирать hop_length
# Для общего аудио/музыки/звуков часто используют 256, 512 или 1024.
# Если  события длятся десятки–сотни миллисекунд, шаг в районе 10–20 мс (то есть hop_length ≈ sr/100 – sr/50) даёт нормальное разрешение.
# 
# 

# In[4]:


hop_length = 512

o_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length) #возвращает одномерный массив o_env, где каждый элемент соответствует кадру и показывает силу появления события 

onset_frames = librosa.onset.onset_detect(
    onset_envelope=o_env,
    sr=sr,
    hop_length=hop_length,
    backtrack=True,
    units="frames"
)
# находит конкретные кадры onset_frames, где происходят onsets (начала звуков). Параметр backtrack=True сдвигает найденный onset назад к ближайшему минимуму энергии, что даёт более аккуратные точки разреза для сегментации

onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length) #переводим кадры в секунды

print("n_onsets:", len(onset_frames))


# 4. Нарисуем график сигнала по времени + на нем красными пунктирами обочзначим размеченные отрезки(границы сегментов)

# In[5]:


plt.figure(figsize=(14, 4))
librosa.display.waveshow(y, sr=sr, alpha=0.6)
plt.vlines(onset_times, ymin=y.min(), ymax=y.max(), color="r", alpha=0.4, linestyle="--")
plt.tight_layout()
plt.show()


# 5. режем наше аудио на отдельные куски и складываем информацию о них в таблицу

# In[7]:


onset_samples = librosa.frames_to_samples(onset_frames, hop_length=hop_length)
cut_points = np.unique(np.concatenate([[0], onset_samples, [len(y)]]))

min_dur = 0.2  # минимум длительности сегмента, сек

segments = []
for i in range(len(cut_points) - 1):
    start = int(cut_points[i])
    end = int(cut_points[i + 1])
    seg_y = y[start:end]
    dur = (end - start) / sr
    if dur >= min_dur:
        segments.append({
            "segment_id": len(segments),
            "start_sample": start,
            "end_sample": end,
            "start_time": start / sr,
            "end_time": end / sr,
            "duration": dur,
            "y": seg_y,
            "label": "unknown"   # пока не знаем класс
        })

df_segments = pd.DataFrame(segments)
df_segments.head()


# посмотрим на длительности полученных сегментов 

# In[8]:


df_segments["duration"].describe()


# In[9]:


plt.figure(figsize=(8, 3))
plt.hist(df_segments["duration"], bins=40)
plt.xlabel("длительность (сек)")
plt.ylabel("кол-во")
plt.tight_layout()
plt.show()


# 5. почистим сегменты тишины (которые в дальнешем будут только мешать обучению)

# In[10]:


# считаем среднюю энергию (RMS) для каждого сегмента
rms_means = []
for seg in df_segments["y"]:
    rms = librosa.feature.rms(y=seg).mean()
    rms_means.append(rms)

df_segments["rms_mean"] = rms_means

# смотрим распределение RMS, чтобы понять, какой порог тишины выбрать
print(df_segments["rms_mean"].describe())

# пусть нижние 10% по громкости — это тишина
silence_thr = df_segments["rms_mean"].quantile(0.10)

# оставляем только сегменты, которые громче тишины
df_segments = df_segments[df_segments["rms_mean"] > silence_thr].reset_index(drop=True)

print("Сегментов после удаления тишины:", len(df_segments))


# 6. посмотрим файл meta, который хранит информацию про классы

# In[11]:


meta_df = pd.read_csv("meta_df.csv")
meta_df.head()


# 7. напишем функцию, которая присваивает каждому аудио-сгементу метку класса, глядя на врменные интервалы 

# In[12]:


def assign_label(seg_row, meta_df):
    mid = (seg_row["start_time"] + seg_row["end_time"]) / 2.0
    hits = meta_df[(meta_df["start_time"] <= mid) & (meta_df["end_time"] >= mid)]
    return hits.iloc[0]["label"]


# 8. дополним метками класса все наши сегменты

# In[13]:


df_segments["label"] = df_segments.apply(lambda r: assign_label(r, meta_df), axis=1)
df_segments["label"].value_counts()
df_segments.head()


# In[14]:


dataset_df = df_segments.drop(columns=["y"]).copy()
dataset_df.head()


# 9. отфильструем датасэт удалив метки, где нет метки

# In[15]:


dataset_labeled = dataset_df[dataset_df["label"].isin(["cat", "dog"])].reset_index(drop=True)
print("Всего размеченных сегментов:", len(dataset_labeled))
print(dataset_labeled["label"].value_counts())


# 10. делим на обучающую и тестовые выборки 80/20

# In[16]:


from sklearn.model_selection import train_test_split

train_val_df, test_df = train_test_split(
    dataset_labeled,
    test_size=0.2,          
    random_state=42,
    stratify=dataset_labeled["label"]
)

print("Train+Val:", len(train_val_df), "Test:", len(test_df))
print("Test labels:\n", test_df["label"].value_counts())


# 11. при необходимости можем поедлить на train и val

# In[17]:


train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.2,        
    random_state=42,
    stratify=train_val_df["label"]
)

print("Train:", len(train_df), "Val:", len(val_df), "Test:", len(test_df))


#  🔹 момент: может быть что в записи будут строго известны моменты через коротые резать, допустим паузы по 1 сек, в таком случае можно разбитть аудиофайл таким образом

# In[18]:


import librosa
import soundfile as sf
import numpy as np
from pathlib import Path

audio_path = "long_audio.wav"
out_dir = Path("segments")
out_dir.mkdir(exist_ok=True)

y, sr = librosa.load(audio_path, sr=None, mono=True)

silence_threshold = 1e-3 # порог тишины
one_sec = sr # ровно 1 секунда в сэмплах

is_silent = np.abs(y) < silence_threshold

segments = []
current_start = 0
silent_run = 0

for i, silent in enumerate(is_silent):
    if silent:
        silent_run += 1
    else:
        # только что закончилась тишина
        if silent_run > 0:
            if silent_run == one_sec:
                # РОВНО 1 сек тишины -> режем
                cut_pos = i - silent_run
                if cut_pos > current_start:
                    segments.append((current_start, cut_pos))
                    current_start = i  # новый звук после паузы
            # если тишина != 1 сек, считаем её частью звука и не режем
        silent_run = 0

if current_start < len(y):
    segments.append((current_start, len(y)))

for idx, (s, e) in enumerate(segments):
    seg = y[s:e]
    if len(seg) == 0:
        continue
    out_path = out_dir / f"segment_{idx:03d}.wav"
    sf.write(out_path, seg, sr)



#  🔹 момент2 : просто порезать аудио по 1 сек

# In[ ]:


import librosa
import soundfile as sf
from pathlib import Path

audio_path = "long_audio.wav"
out_dir = Path("chunks_1s")
out_dir.mkdir(exist_ok=True)

# читаем аудио
y, sr = librosa.load(audio_path, sr=None, mono=True)

chunk_sec = 1.0
chunk_len = int(chunk_sec * sr)  # 1 секунда в сэмплах

# режем по 1 секунде
for i, start in enumerate(range(0, len(y), chunk_len)):
    end = start + chunk_len
    chunk = y[start:end]
    if len(chunk) == 0:
        continue
    out_path = out_dir / f"chunk_{i:04d}.wav"
    sf.write(out_path, chunk, sr)


# # 🔹 Пункт 2: Представление аудио в цифровом формате

# In[124]:


import os
import librosa
import numpy as np

DATA_DIR = "cats_dogs"
SR = 16000  # одна частота дискретизации для всего датасета


# 1. сырой сигнал (waveform) каждый файл превращается в вектор амплитуд y с собственной длиной
# 
# Такое представление удобно для моделей на торч-аудио или простых базовых экспериментов. librosa.load возвращает одномерный NumPy‑массив амплитуд и частоту дискретизации
# 
# - Подойдёт для PyTorch‑моделей (1D‑CNN, RNN), где паддинг делается уже в DataLoader
# Дальше можно писать PyTorch‑датасет, который в __getitem__ возвращает один массив, а в collate_fn паддить батч до общей длины.

# In[137]:


def load_wave(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y.astype(np.float32)

def load_split(split):
    X, y = [], []
    for name, label in [("cat", 0), ("dog", 1)]:
        folder = os.path.join(DATA_DIR, split, name)
        for fname in os.listdir(folder):
            if fname.lower().endswith(".wav"):
                X.append(load_wave(os.path.join(folder, fname)))
                y.append(label)
    return X, np.array(y, dtype=np.int64)

X_train_wave, y_train_wave = load_split("train")
X_test_wave,  y_test_wave  = load_split("test")

print(len(X_train_wave), len(X_test_wave))


# 2. Мел‑спектрограммы (матрицы частота×время)
# Мел‑спектрограмма — это способ превратить звук в картинку, где по вертикали частоты (в мел‑шкале), по горизонтали время, а в ячейках яркость = энергия. Такое представление очень удобно для 2D‑CNN
# 
# Звук режут на короткие окна (обычно 20–40 мс) и для каждого окна считают спектр (STFT).​
# 
# Затем спектр пропускают через банк мел‑фильтров и суммируют энергию в нескольких диапазонах, например 64 или 128 полос

# In[ ]:


N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256

def to_melspec(y):
    S = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )              
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db.astype(np.float32)

def load_split_mels(split):
    X, y = [], []
    for name, label in [("cat", 0), ("dog", 1)]:
        folder = os.path.join(DATA_DIR, split, name)
        for fname in os.listdir(folder):
            if fname.lower().endswith(".wav"):
                path = os.path.join(folder, fname)
                y_wave, _ = librosa.load(path, sr=SR, mono=True)
                X.append(to_melspec(y_wave))
                y.append(label)
    return X, np.array(y, dtype=np.int64)

X_train_spec, y_train_spec = load_split_mels("train")
X_test_spec,  y_test_spec  = load_split_mels("test")

print(len(X_train_spec),len(X_test_spec))
# Здесь X_train_spec — список матриц разных длин по времени T_i; высота N_MELS фиксирована


# 3. MFCC‑признаки 
# Здесь получается чистый табличный датасет фиксированного размера, что идеально для SVM, RandomForest, LogisticRegression, MLP

# In[ ]:


N_MFCC = 20

def file_to_mfcc(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)  
    mfcc_mean = mfcc.mean(axis=1)                          
    return mfcc_mean.astype(np.float32)

def load_split_mfcc(split):
    X, y = [], []
    for name, label in [("cat", 0), ("dog", 1)]:
        folder = os.path.join(DATA_DIR, split, name)
        for fname in os.listdir(folder):
            if fname.lower().endswith(".wav"):
                path = os.path.join(folder, fname)
                X.append(file_to_mfcc(path)) 
                y.append(label)
    return np.stack(X), np.array(y, dtype=np.int64)

X_train_mfcc, y_train_mfcc = load_split_mfcc("train")
X_test_mfcc,  y_test_mfcc  = load_split_mfcc("test")


# обучим сразу модель LG для примера

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

clf.fit(X_train_mfcc, y_train_mfcc)
print("accuracy:", clf.score(X_test_mfcc, y_test_mfcc))


# # 3. Корреляционный анализ классов
# разберем на примереданных которые мы получили при MFCC

# посчитаем коэффициенты корреляции пирсона между всеми парами MFCC признаков
# Это показывает, какие признаки ведут себя похоже и могут быть избыточными

# In[150]:


import pandas as pd

# преобразуем в DataFrame для удобства
mfcc_cols = [f"mfcc_{i+1}" for i in range(X_train_mfcc.shape[1])]
df_mfcc = pd.DataFrame(X_train_mfcc, columns=mfcc_cols)

corr_matrix = df_mfcc.corr(method="pearson")
corr_matrix


# сделаем heatmap

# In[151]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, cmap="coolwarm", center=0, square=True)
plt.tight_layout()
plt.show()


# MFCC‑3–MFCC‑6 сильно коррелируют также 4-12, 6 - 8. В дальнейшем если дата большая слишком можно удалить один из коррелирующих признаков

# # 🔹Гистограммы одного признака по двум классам
# Построим гистограммы, например, для mfcc_1 и mfcc_2 одновременно для котов и собак. Смотрим, насколько сильно разъезжаются распределения; если почти не пересекаются, признак хорошо отличает классы
# 
# Если гистограммы почти лежат друг на друге (похожая форма и область значений), признак мало помогает отличать сигналы и даёт слабую информацию для классификации

# In[149]:


def plot_hist_for_feature(idx):
    name = mfcc_cols[idx]
    cats = X_train_mfcc[y_train_mfcc == 0, idx]
    dogs = X_train_mfcc[y_train_mfcc == 1, idx]

    plt.figure(figsize=(6, 4))
    plt.hist(cats, bins=20, alpha=0.6, label="cat", density=True)
    plt.hist(dogs, bins=20, alpha=0.6, label="dog", density=True)
    plt.xlabel(name)
    plt.legend()
    plt.tight_layout()
    plt.show()

# пример: первые два MFCC
plot_hist_for_feature(0)  # mfcc_1
plot_hist_for_feature(1)  # mfcc_2

