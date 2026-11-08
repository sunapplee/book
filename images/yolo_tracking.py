#!/usr/bin/env python
# coding: utf-8

# # Трекинг объектов на видео с помощью YOLO26
# 
# В этом ноутбуке рассматривается:
# - запуск детекции и трекинга на видео;
# - использование пользовательского конфигурационного файла трекера;
# - сохранение результата в видеофайл;
# - построение траекторий движения объектов;
# - фильтрация объектов по классам.
# 
# Ноутбук ориентирован на запуск в Jupyter Notebook.

# ---
# ## Содержание
# 
# 1. [Импорт библиотек](#Импорт-библиотек)
# 2. [Базовый запуск трекинга](#Базовый-запуск-трекинга)
# 3. [Конвертация результата в MP4](#Конвертация-результата-в-MP4)
# 4. [Работа с конфигурацией трекера](#Работа-с-конфигурацией-трекера)
# 5. [Построение траекторий движения](#Построение-траекторий-движения)
# 6. [Просмотр доступных классов модели](#Просмотр-доступных-классов-модели)
# 7. [Трекинг только по выбранным классам](#Трекинг-только-по-выбранным-классам)
# 8. [Выводы](#Выводы)

# ---
# ## Импорт библиотек

# In[1]:


from ultralytics import YOLO
from collections import defaultdict
import cv2
import numpy as np
import os
from IPython.display import Video, HTML


# ---
# ## Базовый запуск трекинга
# 
# На этом этапе выполняется:
# - загрузка модели YOLO26;
# - запуск трекинга на видео;
# - сохранение результата в папку `runs/detect/...`.
# 
# Важно:
# - `conf` — порог уверенности детекции;
# - `iou` — порог IoU для подавления пересекающихся боксов;
# - `save=True` — сохранение результата;
# - `show=True` — отображение окна (может не работать в Jupyter / headless-среде).

# In[2]:


model = YOLO("yolo26n.pt")


# In[3]:


results = model.track(
    source="video.mp4",
    conf=0.1,
    iou=0.7,
    show=True,
    save=True,
    verbose=False
)


# ---
# ## Конвертация результата в MP4
# 
# Ultralytics иногда сохраняет результат в `.avi`.
# Для удобного просмотра в Jupyter лучше перевести его в `.mp4`.

# In[14]:


os.system(
    r'ffmpeg -i E:\Heckfy\atom\REA\preparing_total\runs\detect\track5\video.avi '
    r'-c:v libx264 -c:a aac output.mp4'
)


# In[15]:


Video("output.mp4", width=1000)


# ---
# ## Работа с конфигурацией трекера
# 
# Ultralytics позволяет использовать пользовательский `.yaml`-конфиг для настройки трекера.
# 
# Стандартные конфиги обычно находятся здесь:
# 
# `.venv\Lib\site-packages\ultralytics\cfg\trackers`
# 
# Например:
# - `bytetrack.yaml`
# - `botsort.yaml`
# 
# Ниже приведён пример конфигурации `ByteTrack`.

# In[16]:


tracker_config_example = """
tracker_type: bytetrack
track_high_thresh: 0.25
track_low_thresh: 0.1
new_track_thresh: 0.25
track_buffer: 30
match_thresh: 0.8
fuse_score: True
"""


# In[17]:


print(tracker_config_example)


# ---
# ### Пояснение к основным параметрам трекера
# 
# - `tracker_type` — тип трекера (`bytetrack` или `botsort`);
# - `track_high_thresh` — высокий порог сопоставления;
# - `track_low_thresh` — нижний порог для слабых детекций;
# - `new_track_thresh` — порог создания нового трека;
# - `track_buffer` — сколько кадров хранить потерянный объект;
# - `match_thresh` — порог сопоставления объектов между кадрами;
# - `fuse_score` — учитывать ли confidence score при матчинге.
# 
# На практике `ByteTrack` часто является хорошим базовым выбором.

# ---
# ## Построение траекторий движения
# 
# В этой секции:
# - видео считывается покадрово;
# - для каждого объекта сохраняется история его положения;
# - поверх кадра рисуется траектория движения;
# - результат сохраняется в `.mp4`.
# 
# Это особенно полезно для:
# - анализа движения людей;
# - подсчёта пересечений;
# - видеонаблюдения;
# - визуализации поведения объектов.

# In[18]:


# Путь к входному видео
video_path = "video.mp4"


# In[19]:


# Открываем видео
cap = cv2.VideoCapture(video_path)


# In[20]:


# Получаем параметры видео
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


# In[21]:


# Создаём writer для сохранения результата
output_path = "output_tracking.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))


# In[22]:


# Храним историю треков:
# key = track_id
# value = список координат центра объекта
track_history = defaultdict(lambda: [])


# In[23]:


# Проходим по кадрам
while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    # Запуск трекинга на текущем кадре
    result = model.track(frame, persist=True)[0]

    # Рисуем стандартную разметку YOLO (bbox, class, id)
    annotated_frame = result.plot()

    # Если треки существуют
    if result.boxes is not None and result.boxes.is_track:
        boxes = result.boxes.xywh.cpu()
        track_ids = result.boxes.id.int().cpu().tolist()

        # Для каждого объекта сохраняем центр бокса
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))

            # Ограничиваем длину истории
            if len(track) > 30:
                track.pop(0)

            # Рисуем траекторию, если накопилось больше 1 точки
            if len(track) > 1:
                points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(
                    annotated_frame,
                    [points],
                    isClosed=False,
                    color=(230, 230, 230),
                    thickness=5
                )

    # Запись кадра в итоговый файл
    out.write(annotated_frame)


# In[24]:


# Освобождаем ресурсы
cap.release()
out.release()


# In[25]:


print(f"Видео сохранено: {output_path}")


# In[26]:


# Просмотр результата
Video("output_tracking.mp4", width=1000)


# ---
# ## Просмотр доступных классов модели
# 
# У стандартной модели YOLO26 обычно используется набор классов COCO (80 классов).
# Посмотреть их можно через `model.names`.

# In[27]:


print("Доступные классы модели:")
print(model.names)


# ---
# ### Полезные классы для трекинга
# 
# Наиболее часто используемые:
# 
# - `0` — person
# - `1` — bicycle
# - `2` — car
# - `3` — motorcycle
# - `5` — bus
# - `7` — truck
# 
# Это позволяет ограничить поиск только нужными объектами
# и ускорить обработку видео.

# ---
# ## Трекинг только по выбранным классам
# 
# Ниже показан пример трекинга только людей (`class 0`).
# Это удобно, если требуется:
# - отслеживать только людей;
# - строить траектории движения;
# - анализировать перемещение в кадре;
# - уменьшить количество лишних детекций.

# In[28]:


model = YOLO("yolo26n.pt")


# In[29]:


results = model.track(
    source="video.mp4",
    conf=0.1,
    iou=0.7,
    show=True,
    save=True,
    verbose=False,
    classes=[0]  # только люди
)


# In[30]:


# Конвертация результата в mp4
os.system(
    r'ffmpeg -i E:\Heckfy\atom\REA\preparing_total\runs\detect\track4\video.avi '
    r'-c:v libx264 -c:a aac output_people.mp4'
)


# In[31]:


Video("output_people.mp4", width=1000)


# ---
# ## Выводы
# 
# В рамках ноутбука были рассмотрены основные возможности object tracking в YOLO26:
# 
# 1. Базовый запуск трекинга на видео;
# 2. Использование конфигурационного файла трекера;
# 3. Сохранение результата в видеофайл;
# 4. Построение траекторий движения объектов;
# 5. Фильтрация объектов по классам.
# 
# Такой подход можно использовать как основу для:
# - систем видеонаблюдения;
# - анализа поведения объектов;
# - подсчёта людей и транспорта;
# - интеллектуального мониторинга видео.
