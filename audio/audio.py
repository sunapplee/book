#!/usr/bin/env python
# coding: utf-8

# # Анализ аудиоданных для классификации
# 
# ### Цель работы
# 
# Провести разведочный анализ аудиоданных (EDA) для задачи классификации.

# ### Используемые библиотеки

# In[4]:


import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm


# ## Загрузка аудиофайлов

# Датасет организован в виде папок, где каждая папка соответствует отдельному классу.

# In[5]:


train_dir = "data/"
classes = []
for class_name in os.listdir(train_dir):
  class_path = train_dir + class_name

  for audiofile in os.listdir(class_path):
    file_path = f'data/{class_name}/{audiofile}'
    audio, sr = librosa.load(file_path)
    classes.append([class_name, file_path, audio])

print(sr)


# In[6]:


class_counts_df = pd.DataFrame(classes, columns=['class', 'file', 'audio'])
class_counts_df


# Распределение файлов по классам позволяет выявить дисбаланс данных, который может повлиять на обучение модели.

# In[7]:


class_counts_df['class'].value_counts()


# Визуализируем временной сигнал одного аудиофайла для примера.

# In[9]:


sample_audio = class_counts_df.loc[1, 'audio']

plt.figure(figsize=(10, 3))
librosa.display.waveshow(sample_audio, sr=sr)
plt.show()


# ## Описательные статистики

# ### Длительность аудио

# In[10]:


class_counts_df['duration'] = class_counts_df['audio'].apply(lambda x: len(x) / sr)
class_counts_df.head(3)


# ### Средняя громкость

# In[11]:


class_counts_df['mean_loudness'] = class_counts_df['audio'].apply(lambda x: np.mean(np.abs(x)))
class_counts_df.head(3)


# ### Изменчивость громкости

# In[12]:


class_counts_df['loudness_variability'] = class_counts_df['audio'].apply(lambda x: np.std(x))
class_counts_df.head(3)


# ### Максимальная громкость

# In[23]:


class_counts_df['max_loudness'] = class_counts_df['audio'].apply(lambda x: np.mean(np.max(x)))
class_counts_df.head(3)


# ## Отличительные характеристики классов на основе статистик

# In[25]:


class_counts_df.groupby('class')[['duration', 'mean_loudness', 'max_loudness', 'loudness_variability']].mean()


# И тут мы оставляем вывод.

# ## Извлечение признаков (Features Extraction)
# 

# ### Функция извлечения признаков
# 
# Извлекаем комплексные признаки из аудиосигнала: MFCC, Chroma, спектральный центроид, ширина полосы и rolloff.

# In[26]:


def extract_features(y, sr=22050):
    # MFCC - 13 коэффициентов
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    # Хромограмма - 12 полутонов
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    # Центроид спектра
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    # Ширина спектра
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    # Спектральный роллофф
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    return np.concatenate([
        np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
        np.mean(chroma, axis=1), np.std(chroma, axis=1),
        [np.mean(centroid), np.std(centroid)],
        [np.mean(bandwidth), np.std(bandwidth)],
        [np.mean(rolloff), np.std(rolloff)]
    ])


# ### Применение функции к аудиофайлам и добавление признаков в датафрейм
# 

# In[27]:


class_counts_df[[f"features_{i}" for i in range(1, 57)]] = class_counts_df['audio'].apply(extract_features).tolist()

class_counts_df.sample(2)


# ## Спектрограммы

# ### STFT-спектрограмма

# In[28]:


D = librosa.stft(class_counts_df.loc[1, 'audio'])
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)


plt.figure(figsize=(10, 4))
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz')
plt.title('STFT-спектрограмма')
plt.tight_layout()
plt.show()


# In[29]:


S_db


# In[30]:


S_db.shape


# ### Mel-спектрограмма

# In[31]:


S = librosa.feature.melspectrogram(y=class_counts_df.loc[1, 'audio'], sr=sr, n_mels=20)
S_dB = librosa.power_to_db(S, ref=np.max)


plt.figure(figsize=(10, 4))
librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
plt.title('Mel-спектрограмма')
plt.tight_layout()
plt.show()


# In[32]:


S_dB


# In[33]:


S_dB.shape

