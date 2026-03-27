#!/usr/bin/env python
# coding: utf-8

# # Описание

# # Содержание
# * [Импорт библиотек](#0)
# * [1. Разметка данных](#1)
# * [2. Загрузка данных](#2)
# * [3. Аудио-эмбеддинги](#3)
# * [4. Бэйслайн ML](#4)
# * [5. Предобработка данных](#5)
# * [6. Транскрибация текста с Whisper](#6)
# * [7. Постобработка транскрибации и сопоставление с классами](#7)

# ###

# <a id=0></a>
# ## Импорт библиотек

# In[89]:


import pandas as pd
import numpy as np
from tqdm import tqdm
import re
from rapidfuzz import process
import librosa
from df import enhance, init_df
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
from faster_whisper import WhisperModel
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from jiwer import wer, cer

import matplotlib.pyplot as plt

from IPython.display import Audio


# <a id=1></a>
# # 1. Разметка данных
# 
# Для решения задачи были предоставлены аудиозаписи, относящиеся к различным классам команд. Перед обучением модели необходимо подготовить размеченный датасет, поэтому на первом этапе проводится **ручная аннотация аудио**.
# 
# В качестве инструмента для разметки был выбран **Label Studio**, который предоставляет удобный интерфейс для работы с аудиофайлами: позволяет прослушивать записи, просматривать форму сигнала и присваивать каждому файлу соответствующий класс.

# <img src="images/image1.png" width="800">

# Сначала в систему были загружены все аудиофайлы (рис. выше). 
# 
# Далее была настроена конфигурация разметки: в XML-описании интерфейса были определены классы, которые может выбирать аннотатор:
# 
# ```xml
# <View>
#   <Audio name="audio" value="$audio"/>
# 
#   <Choices name="intent" toName="audio" choice="single" required="true">
#     <Choice value="отказ"/>
#     <Choice value="отмена"/>
#     <Choice value="подтверждение"/>
#     <Choice value="начать осаживание"/>
#     <Choice value="осадить на (количество) вагон"/>
#     <Choice value="продолжаем осаживание"/>
#     <Choice value="зарядка тормозной магистрали"/>
#     <Choice value="вышел из межвагонного пространства"/>
#     <Choice value="продолжаем роспуск"/>
#     <Choice value="растянуть автосцепки"/>
#     <Choice value="протянуть на (количество) вагон"/>
#     <Choice value="отцепка"/>
#     <Choice value="назад на башмак"/>
#     <Choice value="остановка"/>
#     <Choice value="захожу в межвагонное пространство"/>
#     <Choice value="вперед на башмак"/>
#     <Choice value="сжать автосцепки"/>
#     <Choice value="назад с башмака"/>
#     <Choice value="тише"/>
#     <Choice value="вперед с башмака"/>
#     <Choice value="прекратить зарядку тормозной магистрали"/>
#     <Choice value="тормозить"/>
#     <Choice value="отпустить"/>
#   </Choices>
# 
# </View>
# 
# ```
# 
# После этого интерфейс разметки отображает аудиоплеер и список доступных меток.

# <img src="images/image3.png" width="800">

# 
# На следующем этапе выполнялась непосредственная аннотация. Для каждой записи аудио воспроизводилось, после чего выбирался наиболее подходящий класс (рис. выше). Таким образом была проведена последовательная проверка и разметка всех файлов.

# <img src="images/image2.png" width="800">

# 
# Всего было размечено **106 аудиозаписей** (рис. выше). После завершения процесса аннотации результаты были экспортированы из системы в формате **CSV**, содержащем метаданные разметки: имя аудиофайла и соответствующий ему класс. 
# 
# Полученный файл будет использоваться далее для подготовки датасета и обучения модели классификации аудио.

# In[2]:


metadata = pd.read_csv('metadata.csv')
metadata.head()


# ###

# <a id=2></a>
# # 2. Загрузка данных
# 
# Загрузим метаданные, выделим из них путь к аудиофайлам и метку класса. Адаптируем пути под текущую структуру директорий и преобразуем текстовые метки в числовые идентификаторы классов.

# ### Работа с метаданными

# In[3]:


# Словарь с классами
LABEL2ID = {
    "отказ": 0,
    "отмена": 1,
    "подтверждение": 2,
    "начать осаживание": 3,
    "осадить на (количество) вагон": 4,
    "продолжаем осаживание": 5,
    "зарядка тормозной магистрали": 6,
    "вышел из межвагонного пространства": 7,
    "продолжаем роспуск": 8,
    "растянуть автосцепки": 9,
    "протянуть на (количество) вагон": 10,
    "отцепка": 11,
    "назад на башмак": 12,
    "захожу в межвагонное пространство": 13,
    "остановка": 14,
    "вперед на башмак": 15,
    "сжать автосцепки": 16,
    "назад с башмака": 17,
    "тише": 18,
    "вперед с башмака": 19,
    "прекратить зарядку тормозной магистрали": 20,
    "тормозить": 21,
    "отпустить": 22,
}


# In[4]:


# Отбираем нужные колонки
audio_info = metadata.loc[:, ['audio', 'intent']]


# In[5]:


# Адаптируем путь под нашу текущую директорию
audio_info['audio'] = audio_info['audio'].apply(lambda x: f'data/{x.split("-")[1]}')


# In[6]:


# Создаем новую колоку TARGET
audio_info['TARGET'] = audio_info['intent'].apply(lambda x: LABEL2ID[x])
audio_info.head()


# ### Загрузка аудио

# С помощью библиотеки `librosa` загрузим аудио из набора данных с частотой дискретизации `16000 Hz`.

# In[7]:


# Список путей
audio_paths = audio_info['audio']
# Словарь для сохранения
audios = {}

for path in tqdm(audio_paths, desc='Загрузка аудио', ncols=70):
    loaded_audio, sr = librosa.load(path, sr=16000)
    audios[path] = loaded_audio


# ###

# <a id=3></a>
# # 3. Аудио-эмбеддинги
# 
# Превратим аудио в числовые представления с помощью предобученной модели `Wav2Vec2`.
# 
# На выходе получаем последовательность векторов последнего слоя энкодера. Полученные эмбеддинги будем использовать как признаки для **дальнейшего обучения моделей**.

# In[8]:


# Определяем вычислительное устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Инициализируем модель
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

# Предеводим модель в режим инференса
model = model.to(device)
model.eval()


# Напишем функцию для получения **векторов для списка аудио**.

# In[9]:


def extract_embeddings(audio):

    # Подготовка аудио перед подачей в модель
    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    # Переводим на выбранное вычислительное ус-во
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Прогоняем через модель
    with torch.no_grad():
        outputs = model(**inputs)

    # Выход последнего слоя энкодера модели
    embeddings = outputs.last_hidden_state.mean(dim=1)

    return embeddings.cpu()


# Извлечение **векторов** для всего датасета.

# In[10]:


# Будет сохранять векторы в словарь
embeddings = {}

# Проходимся по аудио
for audio_name in tqdm(audios, desc='Векторизация аудио'):
    # Прогоняем аудио через модель
    emb = extract_embeddings(audios[audio_name])

    # Добавляем в список
    embeddings[audio_name] = emb.numpy()[0]


# Представим векторы в табличном варианте.

# In[11]:


emb_df = pd.DataFrame(embeddings).T

print('Размер датасета:', emb_df.shape)
emb_df.head()


# Итого получили `768` признаков для каждого аудио, обучим модель машинного обучения на этих данных.

# ### 

# <a id=4></a>
# # 4. Бэйслайн ML
# 
# В данном разделе построим базовую модель машинного обучения для решения задачи распознавания.  
# В качестве признаков используем ранее полученные аудио-эмбеддинги.  

# ### Разделение выборки
# 
# Разделим данные на обучающую и тестовую выборку в соотношении 80/20

# In[12]:


X_train, X_test, y_train, y_test = train_test_split(emb_df, audio_info['TARGET'], test_size=0.2)


# ### Инициализация и обучение модели

# In[13]:


model = LogisticRegression(C=1.0, max_iter=200, l1_ratio=0.0)
model.fit(X_train, y_train)


# ### Инференс модели

# In[14]:


# Создадим словарь для поиска текста по классу
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# Получим предсказание модели
class_pred = model.predict(X_test)

# Переведем их в текстовое представление
pred_text = [ID2LABEL[c] for c in class_pred]
true_text = [ID2LABEL[c] for c in y_test]


# ### Расчет метрик
# 
# Для оценки качества распознавания речи используются метрики **WER** и **CER** из библиотеки `jiwer`.
# 
# - **WER (Word Error Rate)** — доля ошибок на уровне слов.  
# 
# - **CER (Character Error Rate)** — доля ошибок на уровне символов.  

# In[15]:


print(f'WER для логистической регрессии: {wer(pred_text, true_text):.3f}')
print(f'CER для логистической регрессии: {cer(pred_text, true_text):.3f}')


# ###

# <a id=5></a>
# # 5. Предобработка данных

# ### Удаление шума
# 
# Удалим шум в аудио, используя `DeepFilterNet`.

# In[16]:


# Загрузка базовой модели
noise_model, df_state, _ = init_df()


# Напишем функцию для обработка аудио.

# In[17]:


def noise_remove(audio):
    # Прогоняем аудио через модель удаления шума
    enhanced_audio = enhance(noise_model, df_state,
                             torch.tensor(audio).unsqueeze(0),
                             atten_lim_db=25 # уменьшает агрессивность
                            )
    # Возвращаем предобработку
    return enhanced_audio


# Обработаем весь датасет.

# In[18]:


# Сюда будем сохранять аудио
preprocessed_audios = {}

# Проходимся по каждому аудио
for audio_name in tqdm(audios, desc='Удаление шума'):

    # Удаляем шум
    audio_without_noise = noise_remove(audios[audio_name])

    # Сохраняем в словарь
    preprocessed_audios[audio_name] = audio_without_noise


# ##### Протестируем предобработку

# In[19]:


print('Оригинальное аудио')
Audio(audios['data/2023_11_15__10_36_00.wav'], rate=16000)


# In[20]:


print('Аудио без шумов')
Audio(preprocessed_audios['data/2023_11_15__10_36_00.wav'], rate=16000)


# In[21]:


# Визуализируем waveform
fit, axs = plt.subplots(2, 1, figsize=(6, 10))

axs[0].set_title('Оригинальное аудио')
librosa.display.waveshow(audios['data/2023_11_15__10_36_00.wav'], sr=16000, ax=axs[0])

axs[1].set_title('Аудио без шумов')
librosa.display.waveshow(preprocessed_audios['data/2023_11_15__10_36_00.wav'].numpy(), sr=16000, ax=axs[1]);


# ### Обнаружение речевой активности
# 
# Моделью обнаружения речи выступит `Silero`. Загрузим ее.

# In[22]:


vad_model = load_silero_vad()


# Объявим функцию которая прогоняет аудио через **Silero VAD** и возвращает положение участков с речью.

# In[23]:


def detect_speech_audio(audio) -> np.ndarray:

    # Silero ожидает torch.Tensor
    # wav = torch.from_numpy(audio).unsqueeze(0)

    # Получаем тайминги голосовых сегментов
    speech_timestamps = get_speech_timestamps(
      audio,
      vad_model,
      return_seconds=False,
    )

    # Возвращаем тайминги
    return speech_timestamps


# Получим тайминги для всего датасета.

# In[24]:


# Сюда будем сохранять тайминги
audios_speech_timestamps = {}

# Проходимся по каждому аудио
for audio_name in tqdm(audios, desc='Обнаружение голоса'):

    # Обнаружение шума
    tms = detect_speech_audio(audios[audio_name])

    # Сохраняем в словарь
    audios_speech_timestamps[audio_name] = tms


# ##### Протеструем предобработку

# In[25]:


# Берем тестовый звук
test_audio = 'data/2023_10_11__09_44_03.wav'
coords = [tuple(i.values()) for i in audios_speech_timestamps[test_audio]]


# In[26]:


# Определяем данные для визуализации
stamps = np.arange(preprocessed_audios[test_audio].size()[1])
condlist = [
    (stamps >= pair[0]) & (stamps <= pair[1])
    for pair in coords
]
choices = [True] * len(coords)
step = np.select(condlist, choicelist=choices, default=False)


# Отображаем график

# In[27]:


plt.plot(preprocessed_audios[test_audio].numpy()[0]);
plt.plot(step, color="red");


# ### Отфильтрованный аудиосигнал, содержащий только речевые сегменты
# 
# Удалим все фрагменты, где не была обнаружена речь.

# In[28]:


def filter_audio(audio, timestamps):

    # Сохраняем сюда фрагменты речи
    fragments = []

    np_audio = audio[0].numpy()
    # Сохраняем аудио на позициях start:end
    for ts in timestamps:
        s, e = ts["start"], ts["end"]
        fragments.append(np_audio[s:e])

    return np.concatenate(fragments)


# Предобработаем весь датасет.

# In[29]:


# Проходимся по каждому аудио
for audio_name in tqdm(preprocessed_audios, desc='Фильтрация голоса'):

    # Обнаружение шума
    filtered_audio = filter_audio(preprocessed_audios[audio_name], audios_speech_timestamps[audio_name])

    # Сохраняем в словарь
    preprocessed_audios[audio_name] = filtered_audio


# ##### Протестируем предобработку

# In[30]:


print('Оригинальное аудио')
Audio(audios['data/2023_10_11__09_44_03.wav'], rate=16000)


# In[37]:


print('Аудио с речью')
Audio(preprocessed_audios['data/2023_10_11__09_44_03.wav'], rate=16000)


# В итоге получили набор предобработанных данных готовых для анализа.

# ###

# <a id=6></a>
# # 6. Транскрибация текста с Whisper
# Для преобразования аудио в текст используется модель распознавания речи **Whisper**. В работе применяется версия **large-v3** из библиотеки `faster-whisper`, которая обеспечивает более быстрый инференс по сравнению с оригинальной реализацией.

# In[40]:


# Инициализируем модель
whisper_model = WhisperModel("large-v3", device="cuda", compute_type="float16")


# Напишем функцию для транскрибации текста.

# In[80]:


def transcribe_audio(audio):
    # Пргоняем аудио через модель
    segments, info = whisper_model.transcribe(audio,
                                              language='ru',
                                              temperature=0, )
    # Собираем фрагменты в одну строку
    text = " ".join([segment.text for segment in segments])

    # У наших классов отсутствует пунктуация, а также вся строка в нижнем регистре
    text = text.lower().strip()
    text = re.sub(r'[^A-Za-zА-Яа-яЁё0-9 ]', '', text)
    # Приводим к формату класса
    text = re.sub(r'вагон(ов|а)?', 'вагон', text)
    result = re.sub(r'\d+', '(количество)', text)

    # Возвращаем текст
    return result


# Транскрибируем весь оригинальный и обработанный датасет.

# In[81]:


original_preds = []
preprocessed_preds = []

for filename in tqdm(audios, desc='Транскрибация аудио'):
    # Обработаем оригинал
    orig_pred = transcribe_audio(audios[filename])
    original_preds.append(orig_pred)

    # Обработаем обработку
    prep_pred = transcribe_audio(preprocessed_audios[filename])
    preprocessed_preds.append(prep_pred)


# ### Расчет метрик
# 
# Для оценки качества распознавания речи используются метрики **WER** и **CER** из библиотеки `jiwer`.
# 
# - **WER (Word Error Rate)** — доля ошибок на уровне слов.  
# 
# - **CER (Character Error Rate)** — доля ошибок на уровне символов.  

# In[108]:


true_text = audio_info['intent'].tolist()

print(f'WER для оригинального аудио: {wer(original_preds, true_text):.3f}')
print(f'CER для оригинального аудио: {cer(original_preds, true_text):.3f}')
print()
print(f'WER для обработанного аудио: {wer(preprocessed_preds, true_text):.3f}')
print(f'CER для обработанного аудио: {cer(preprocessed_preds, true_text):.3f}')


# Качество на траскрибации оригинального звука получилось лучше, чем на нашей обработке. Возможно это связано со способностью модели `whisper-large-v3` лучше обрабатывать шум и речь. Для дальнешего анализа берем расшифровку оригинального звука.

# ###

# <a id=7></a>
# # 7. Постобработка транскрибации и сопоставление с классами
# 
# После получения транскрибации с помощью модели распознавания речи необходимо сопоставить полученный текст с одним из заранее заданных классов команд.
# 
# Поскольку распознавание речи может содержать ошибки (искажения слов, пропуски, лишние слова), прямое сравнение строк может работать нестабильно. Поэтому используется поиск наиболее похожей команды среди эталонного списка.

# In[109]:


# Список эталонных команд
COMMANDS = list(LABEL2ID.keys())


# Для сопоставления предсказанного текста с эталонными командами используется библиотека `rapidfuzz`.

# In[110]:


# Объявим функцию для получения лучшего совпадения
def match_command(text, commands):
    best_match, score, _ = process.extractOne(text, commands)
    return best_match


# Получение предсказанных классов.

# In[111]:


pred_labels = []

# Проходимся по каждому предсказанию
for text in original_preds:
    command = match_command(text, COMMANDS)
    pred_labels.append(command)


# In[113]:


print(f'Итоговый WER: {wer(pred_labels, true_text):.3f}')
print(f'Итоговый CER: {cer(pred_labels, true_text):.3f}')

