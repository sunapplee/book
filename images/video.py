#!/usr/bin/env python
# coding: utf-8

# ```Видео по сути те же самые картинки, но необходимо правильно разбить видео. Тк это не cv напрвление, то предположим, что будут простейшие датасэты, где можно явно определить класс изобраения. Возмем базовый датасэт видео(на на +- 30 мин, движение часовой стрелки, разбиваем по секундам, по 5 сек). для определния класса достаточно посмотреть начальное время, дальше прибавляем 1 секунду и ссотвественно переход на минуты и часы. ```

#  ultralytics — библиотека, которая содержит YOLO и режим classification (YOLO-CLS) для обучения и предсказания классов по изображению.
# 
# opencv-python (cv2) — полезен для работы с видео/изображениями (хотя в этом пайплайне извлечение кадров делаем через ffmpeg, OpenCV может пригодиться для кропа/доп. предобработки).

# Импорты

# In[3]:


import os
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO
from sklearn.metrics import accuracy_score, f1_score


# 1) Извлечь кадры (1 fps) из clock.mp4
# FFmpeg — самый простой вариант; -vf fps=1 означает 1 кадр в секунду.
# 
# Командой ffmpeg извлекаем 1 кадр в секунду и сохраняем как frame_0000001.jpg, frame_0000002.jpg, …

# In[4]:


video_path = "clock.mp4"
frames_dir = Path("frames")
frames_dir.mkdir(exist_ok=True, parents=True)

get_ipython().system('ffmpeg -hide_banner -loglevel error -i "{video_path}" -vf fps=1 "{frames_dir}/frame_%07d.jpg"')


# Считываем список кадров и проверяем:
#  - сколько файлов получилось,
#  - первые и последние имена (чтобы убедиться, что извлечение прошло нормально).

# In[5]:


frames = sorted(frames_dir.glob("frame_*.jpg"))
len(frames), frames[:3], frames[-3:]


# Модель обучается на картинках, а у нас вход — видео, поэтому сначала переводим задачу “видео” → “набор изображений”.
# 
# fps=1 даёт равномерную выборку по времени и не делает датасет слишком большим.
# 
# Результат: получаем папку frames/ с последовательностью кадров и понимаем, сколько секунд (примерно) в видео.

# 2) Сбор датасета с классами “минута + шаг секунд 5”
# 
# Здесь мы создаём новый датасет clock_minutes/ и внутри него папку all/. Мы её пересоздаём (удаляем старую и делаем новую), чтобы не мешались результаты от предыдущих запусков.
# 
# Дальше задаём, что время в видео начинается с 10:00, и считаем метки из номера кадра. Поскольку у нас 1 кадр = 1 секунда, для кадра с индексом i (начиная с нуля) “прошедшее время” равно i секунд. Чтобы получить минуту, мы делим прошедшие секунды на 60: minute = start_minute + (total_seconds // 60). Классом становится строка минуты, например "10", "11", "12".
# 
# Затем мы копируем каждый кадр в папку своего класса: clock_minutes/all/10/, clock_minutes/all/11/ и т.д. Это удобно, потому что YOLO‑CLS ожидает датасет именно в формате “папка = класс, внутри лежат картинки этого класса”. В конце мы печатаем статистику: сколько минутных классов получилось, сколько всего изображений.

# In[6]:


out_root = Path("clock_minutes")
all_root = out_root / "all"

if all_root.exists():
    shutil.rmtree(all_root)
all_root.mkdir(parents=True, exist_ok=True)

start_minute = 10
start_second = 0
fps = 1

frames = sorted(Path("frames").glob("frame_*.jpg"))
for i, img_path in enumerate(frames):
    total_seconds = start_second + i 
    minute = start_minute + (total_seconds // 60)
    cls = f"{minute:02d}" 

    dst = all_root / cls
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img_path, dst / img_path.name)


class_dirs = sorted([p for p in all_root.iterdir() if p.is_dir()])
counts = [len(list(p.glob("*.jpg"))) for p in class_dirs]
print("Classes:", len(class_dirs))
print("Images:", sum(counts))


# 3) Разбиение на train/val/test внутри каждого класса
# На этом шаге мы превращаем “all” в стандартный набор для обучения. Мы заранее очищаем и создаём папки clock_minutes/train, clock_minutes/val, clock_minutes/test. Затем для каждой папки класса в all/ берём список картинок, перемешиваем и делим на три части.
# 
# Важно: мы делим внутри каждого класса, а не “по времени кусками видео”, потому что иначе часть минут могла бы полностью уйти в test, и модель бы никогда не видела эти классы в train (тогда корректно оценить модель нельзя). Функция split_indices() нужна, чтобы даже если картинок мало, разбиение не получилось пустым (например, чтобы test не исчезал полностью). После копирования файлов в train/val/test мы печатаем число классов в каждом сплите, чтобы убедиться, что структура датасета корректная.

# In[7]:


random.seed(42)

for sp in ["train", "val", "test"]:
    d = out_root / sp
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)

def split_indices(n, train_p=0.8, val_p=0.1):
    """
    Для n>=3 гарантируем минимум 1 в train/val/test.
    """
    if n <= 1:
        return n, 0
    if n == 2:
        return 1, 1 
    n_train = max(1, int(round(n * train_p)))
    n_val = max(1, int(round(n * val_p)))
    # оставим минимум 1 в test
    if n_train + n_val > n - 1:
        n_val = max(1, n - 1 - n_train)
        if n_train + n_val > n - 1:
            n_train = n - 1 - n_val
    return n_train, n_val

for cls_dir in sorted([p for p in all_root.iterdir() if p.is_dir()]):
    imgs = sorted(cls_dir.glob("*.jpg"))
    random.shuffle(imgs)

    n = len(imgs)
    n_train, n_val = split_indices(n)

    train_imgs = imgs[:n_train]
    val_imgs = imgs[n_train:n_train+n_val]
    test_imgs = imgs[n_train+n_val:]

    for sp, items in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
        if not items:
            continue
        dst_dir = out_root / sp / cls_dir.name
        dst_dir.mkdir(parents=True, exist_ok=True)
        for p in items:
            shutil.copy2(p, dst_dir / p.name)

def list_classes(split):
    d = out_root / split
    return sorted([p.name for p in d.iterdir() if p.is_dir()])

print("Train classes:", len(list_classes("train")))
print("Val classes:", len(list_classes("val")))
print("Test classes:", len(list_classes("test")))


# 4) Аугментация train внутри каждого класса
# Здесь мы расширяем только обучающую часть (train), не трогая val/test, чтобы оценка качества оставалась честной. Мы проходим по всем изображениям в clock_minutes/train и для каждого исходного файла создаём дополнительные версии, сохраняя их в ту же папку класса (то есть метка класса не меняется).
# 
# Мы делаем два типа новых данных:
# (1) gray + cutout: переводим изображение в ч/б (grayscale), потом обратно в RGB, чтобы формат оставался стандартным, и накладываем несколько чёрных квадратов (cutout), имитируя блики, перекрытия и помехи.
# (2) color jitter + cutout: слегка меняем яркость/контраст/насыщенность, затем тоже накладываем cutout.
# 
# Суффикс _aug... в имени нужен, чтобы отличать синтетические копии от оригиналов, и мы специально пропускаем файлы, где уже есть _aug, чтобы не “аугментировать аугментации” и не раздуть датасет бесконечно. В конце печатаем, сколько новых изображений создано и сколько стало всего в train.

# In[8]:


from PIL import Image, ImageDraw, ImageEnhance
import random
from pathlib import Path

train_root = out_root / "train"

make_gray = True
make_color = True

random.seed(42)

def to_grayscale_rgb(img: Image.Image) -> Image.Image:
    return img.convert("L").convert("RGB")  # grayscale через convert("L") 

def color_jitter(img: Image.Image) -> Image.Image:
    # изменение яркости/контраста/цвета
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.85, 1.15))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.85, 1.15))
    img = ImageEnhance.Color(img).enhance(random.uniform(0.85, 1.15))
    return img

def apply_cutout(img: Image.Image, n_holes=3, min_frac=0.08, max_frac=0.18) -> Image.Image:
    w, h = img.size
    m = min(w, h)
    out = img.copy()
    draw = ImageDraw.Draw(out)
    for _ in range(n_holes):
        side = max(4, int(random.uniform(min_frac, max_frac) * m))
        x0 = random.randint(0, max(0, w - side))
        y0 = random.randint(0, max(0, h - side))
        draw.rectangle([x0, y0, x0 + side, y0 + side], fill=(0, 0, 0))
    return out

created = 0
img_paths = list(train_root.rglob("*.jpg"))
for p in img_paths:
    # не аугментируем уже созданные
    if "_aug" in p.stem:
        continue

    img = Image.open(p).convert("RGB")

    if make_gray:
        out = apply_cutout(to_grayscale_rgb(img), n_holes=3)
        out_path = p.with_name(p.stem + "_aug_gray_cut.jpg")
        if not out_path.exists():
            out.save(out_path, quality=95)
            created += 1

    if make_color:
        out = apply_cutout(color_jitter(img), n_holes=3)
        out_path = p.with_name(p.stem + "_aug_col_cut.jpg")
        if not out_path.exists():
            out.save(out_path, quality=95)
            created += 1

print("Augmented created:", created)
print("Train images now:", len(list(train_root.rglob("*.jpg"))))


# 5) Обучение YOLO‑CLS на минутных классах
# Мы загружаем предобученную классификационную модель (yolo11n-cls.pt) и запускаем обучение на clock_minutes. Параметр imgsz=224 задаёт размер входной картинки (она будет автоматически приводиться к этому размеру внутри пайплайна). epochs задаёт сколько раз модель увидит весь train‑набор, а batch — сколько картинок обрабатывается за один шаг. 
# 
# # Я использовала 10 эпох, тк это тренировочные показательные данные, в реальных случаях небходимо использовать от 80 до 300

# In[10]:


from ultralytics import YOLO

model = YOLO("yolo11n-cls.pt")
model.train(
    data="clock_minutes",
    imgsz=224,
    epochs=1,
    batch=64,
    project="runs_minutes",
    name="minute_cls"
)


# 6) Оценка качества: accuracy и F1 на train/val/test
# Мы загружаем лучший чекпойнт best.pt и считаем метрики отдельно на train, val и test. Внутри eval_split() мы:
# 
# Берём список классов как имена подпапок (10, 11, …), сортируем их и строим словарь true_to_idx, чтобы каждому имени класса соответствовал числовой индекс.
# 
# Собираем список всех изображений в данном сплите и их истинный класс (он берётся из папки, где лежит файл).
# 
# Для каждого изображения запускаем predict, берём top1 — индекс самого вероятного класса, затем переводим индекс в имя класса через best.names[pred_idx].
# 
# Сопоставляем предсказанное имя класса с индексом из true_to_idx и получаем y_pred, а истинную метку — y_true.
# 
# Считаем accuracy (доля точных попаданий) и F1 в двух вариантах:
# 
# macro — среднее по классам, где каждый класс одинаково важен,
# 
# weighted — среднее, где классы взвешены по числу примеров.
# 
# В конце выводим метрики и видим, насколько модель научилась распознавать минуты, и есть ли переобучение (если train сильно выше val/test) или недообучение (если везде низко).

# In[11]:


import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from ultralytics import YOLO

best = YOLO("runs_minutes/minute_cls/weights/best.pt")  # <-- важно!

out_root = Path("clock_minutes")

def eval_split(split_name: str):
    split_root = out_root / split_name

    # фиксируем порядок классов по именам папок
    class_names = sorted([p.name for p in split_root.iterdir() if p.is_dir()])
    true_to_idx = {name: i for i, name in enumerate(class_names)}

    samples = []
    for cls_name in class_names:
        for img_path in (split_root / cls_name).glob("*.jpg"):
            samples.append((img_path, cls_name))

    y_true, y_pred = [], []
    for img_path, true_name in samples:
        r = best.predict(source=str(img_path), verbose=False)[0]
        pred_idx = int(r.probs.top1)  # индекс предсказанного класса [web:78]

        # pred_idx относится к best.names, поэтому true_idx надо тоже привести к той же системе.
        # Самый простой контроль: проверим совпадение имён.
        pred_name = best.names[pred_idx]
        y_pred.append(true_to_idx.get(pred_name, -1))
        y_true.append(true_to_idx[true_name])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # выбросим случаи, где pred_name не найден (на практике не должно быть)
    mask = y_pred != -1
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    return acc, f1_macro, f1_weighted, len(samples), len(class_names)

for split in ["train", "val", "test"]:
    acc, f1m, f1w, n, k = eval_split(split)
    print(split, "samples:", n, "classes:", k, "acc:", acc, "f1_macro:", f1m, "f1_weighted:", f1w)


# # Повышение точности (параметры при обучении)

# ### epochs
# Зависит от того, успевает ли модель “выучить” данные: если значение маленькое, будет недообучение; если слишком большое — возможен рост train и падение val (переобучение).   
# Правильно выбирать так: ставить epochs достаточно большим и использовать `patience`, чтобы обучение остановилось при плато на валидации.   
# Пример: `epochs=100` или `epochs=200`. 
# 
# ### patience
# Зависит от того, как быстро перестаёт улучшаться метрика на валидации: маленькое значение может остановить обучение слишком рано, большое — гоняет лишние эпохи.   
# Правильно выбирать так: ставить 15–30, и если обучение часто останавливается “слишком рано”, увеличить.   
# Пример: `patience=20`. 
# 
# ### batch
# Зависит от доступной памяти (GPU/CPU RAM): чем больше batch, тем выше требования к памяти; при малом batch обучение более “шумное”.   
# Правильно выбирать так: ставить максимальный batch без ошибок памяти; если памяти мало, уменьшать batch, а затем при необходимости снижать `lr0`.   
# Пример: `batch=16` (если падает по памяти: `batch=8`). 
# 
# ### imgsz
# Зависит от размера деталей на изображении и ресурсов: больший размер входа даёт больше информации (полезно, если объект мелкий), но дороже по памяти и времени.   
# Правильно выбирать так: если часы/стрелки мелкие в кадре — пробовать увеличить `imgsz`, пока хватает памяти; если часы крупные — можно оставить меньше.   
# Пример: `imgsz=224` или `imgsz=256`. 
# 
# ### lr0
# Зависит от оптимизатора и batch: слишком большой `lr0` может делать обучение нестабильным, слишком маленький — очень медленным.   
# Правильно выбирать так: стартовать с типовых значений и уменьшать `lr0`, если метрики “скачут” или обучение не улучшается при маленьком batch.   
# Пример: `lr0=0.001` (для Adam‑типов) или уменьшить при малом batch. 
# 
# ### cos_lr
# Зависит от стратегии изменения learning rate: `cos_lr` включает косинусный scheduler, который меняет learning rate по эпохам и может улучшить сходимость.   
# Правильно выбирать так: включать, если без него качество быстро выходит на плато или обучение нестабильное.   
# Пример: `cos_lr=True`. 
# 
# ### weight_decay
# Зависит от склонности к переобучению: weight decay — L2‑регуляризация, которая штрафует большие веса и может улучшить качество на val/test.   
# Правильно выбирать так: если train заметно выше val/test — увеличить `weight_decay`; если модель недоучивается — уменьшить.   
# Пример: `weight_decay=0.0005`. 
# 
# ### hsv_h, hsv_s, hsv_v
# Зависят от того, насколько в реальности меняются освещение/цвет: эти параметры добавляют цветовую вариативность и помогают обобщению.   
# Правильно выбирать так: ставить умеренно; если часы всегда в одинаковом освещении — можно ослабить, если условия разные — усилить.   
# Пример: `hsv_h=0.02, hsv_s=0.6, hsv_v=0.4`. 
# 
# ### erasing
# Зависит от того, бывают ли перекрытия/блики: erasing случайно “стирает” области картинки и учит модель быть устойчивой к частичным потерям информации.   
# Правильно выбирать так: ставить умеренно (например 0.2–0.4); если модель начинает сильно терять качество — уменьшить.   
# Пример: `erasing=0.3`. 
# 
# ### auto_augment
# Зависит от того, нужно ли автоматически усилить разнообразие: это готовые политики аугментаций для classify.   
# Правильно выбирать так: использовать как быстрый способ усилить аугментации без ручного подбора, но проверять, что качество на val/test растёт.   
# Пример: `auto_augment="randaugment"`. 
# 
# ### fliplr / flipud (в задаче “время по стрелкам”)
# Зависят от того, сохраняется ли смысл метки при отражении: для часов отражение меняет геометрию времени, поэтому такие аугментации обычно выключают.   
# Правильно выбирать так: ставить 0, чтобы не добавлять “неправильные” примеры.   
# Пример: `fliplr=0.0, flipud=0.0`. 
# 
# ***
# 

# Повысить точность YOLO (и для классификации, и для детекции/сегментации) обычно можно тремя путями: улучшить данные, правильно выбрать размер модели и настроить обучение.
# 
# ## Что почти всегда даёт прирост
# - **Чистая разметка и одинаковые правила.** Ошибки в метках и разные “стандарты” разметки сильно ограничивают максимум качества, сколько параметры ни крути.
# - **Правильный вход (кроп/масштаб).** Если объект маленький, качество часто растёт сильнее от кропа/увеличения `imgsz`, чем от увеличения эпох.
# - **Достаточно данных + правильные аугментации.** Для классификации полезны `hsv_*`, `erasing`, `auto_augment`; для задач, где отражение меняет смысл (как часы), `fliplr/flipud` лучше выключать.
# 
# ## Какую “YOLO-модель” выбирать
# У Ultralytics модели обычно идут в размерах `n/s/m/l/x` (nano/small/medium/large/xl). Чем модель больше, тем выше потенциальная точность, но тем тяжелее обучение и инференс.
# Практический выбор такой:
# - **n** — когда железо слабое, важна скорость, или нужен быстрый базовый результат.
# - **s** — “золотая середина” для слабых GPU/ноутбуков, часто заметно точнее nano.
# - **m/l/x** — когда есть ресурсы и нужно максимум качества (обычно лучший прирост на сложных данных).
# 
# Если выбирать между версиями, то YOLO11 позиционируется как более точная и эффективная по сравнению с YOLOv8 в официальных материалах/анонсах, поэтому при прочих равных логично начинать с YOLO11-версии нужного размера.
