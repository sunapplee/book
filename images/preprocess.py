# C1_M2 - Предобработка датасета

## Импорты и константы

```python
import os, zipfile, shutil, random
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from collections import Counter

ARCHIVE_NAME = 'Gauge.zip'
TEMP_DIR = 'temp_extract'
ORIGINAL_DIR = 'images'
PROCESSED_DIR = 'preprocessed_images'


# result = subprocess.run(['7z', 'x', 'Gauge_big.z01', '-y'], 
#                       capture_output=True, text=True)
```

## Предобработка 224x224 + grayscale

```python


def preprocess_image(src_path, dst_path):
    with Image.open(src_path) as img:
        img = img.resize((224, 224)).convert('L').convert('RGB')
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        img.save(dst_path)


```

## Аугментация

```python


def augment_image(image_path, show_results=False):
    base, ext = os.path.splitext(image_path)
    tilted = f"{base}_tilted{ext}"
    corrupted = f"{base}_corrupted{ext}"

    if os.path.exists(tilted) and os.path.exists(corrupted):
        print(f"Аугментации для {os.path.basename(image_path)} уже существуют")
        return

    with Image.open(image_path) as img:
        original = img.copy()

        # Поворот
        angle = random.uniform(-15, 15)
        rotated = original.rotate(angle, expand=False)

        # Прямоугольник на исходном
        corrupted_img = original.copy()
        draw = ImageDraw.Draw(corrupted_img)
        w, h = corrupted_img.size
        rw, rh = random.randint(20, 50), random.randint(20, 50)
        x, y = random.randint(0, w - rw), random.randint(0, h - rh)
        draw.rectangle([x, y, x + rw, y + rh], fill='black')

        if show_results:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(original);
            axes[0].set_title('Исходное')
            axes[1].imshow(rotated);
            axes[1].set_title(f'Поворот {angle:.1f}°')
            axes[2].imshow(corrupted_img);
            axes[2].set_title('Прямоугольник')
            plt.show()
            plt.close()
            return

        rotated.save(tilted)
        corrupted_img.save(corrupted)


```

## Запуск

```python
# Очистка
for d in [TEMP_DIR, ORIGINAL_DIR, PROCESSED_DIR]:
    shutil.rmtree(d, ignore_errors=True)

# 1. Загрузка в images/train,test,val
with zipfile.ZipFile(ARCHIVE_NAME) as z:
    z.extractall(TEMP_DIR)

data_root = TEMP_DIR
if len(os.listdir(TEMP_DIR)) == 1:
    data_root = os.path.join(TEMP_DIR, os.listdir(TEMP_DIR)[0])

os.makedirs(ORIGINAL_DIR)
for split in ['train', 'test', 'valid']:
    src = os.path.join(data_root, split)
    dst = os.path.join(ORIGINAL_DIR, split)
    if os.path.exists(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)

# 2. Предобработка всех изображений
for root, _, files in os.walk(ORIGINAL_DIR):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            src = os.path.join(root, file)
            rel = os.path.relpath(root, ORIGINAL_DIR)
            dst = os.path.join(PROCESSED_DIR, rel, file)
            preprocess_image(src, dst)

# 3. Аугментация 1/класс (21->23)
train_dir = os.path.join(PROCESSED_DIR, 'train')
for class_name in os.listdir(train_dir):
    class_dir = os.path.join(train_dir, class_name)
    if os.path.isdir(class_dir):
        images = [f for f in os.listdir(class_dir)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                  and not f.endswith(('_tilted', '_corrupted'))]
        if images:
            augment_image(os.path.join(class_dir, images[0]))


# 5. Формирование итогового датасета из train (опционально)
train_dir = os.path.join(PROCESSED_DIR, 'train')
all_images = []

# Собираем все изображения из train
for class_name in os.listdir(train_dir):
    class_dir = os.path.join(train_dir, class_name)
    if os.path.isdir(class_dir):
        images = [os.path.join(class_dir, f) for f in os.listdir(class_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        all_images.extend(images)

# Перемешиваем и делим
random.shuffle(all_images)
train_split = all_images[:23000]
test_split = all_images[23000:24000]
valid_split = all_images[24000:26000]

# Создаем новую структуру
for split_name, split_images in [('train', train_split), ('test', test_split), ('valid', valid_split)]:
    split_dir = os.path.join(PROCESSED_DIR, split_name)
    
    # Очищаем старую структуру
    if os.path.exists(split_dir):
        shutil.rmtree(split_dir)
    
    # Копируем изображения с сохранением классов
    for img_path in split_images:
        class_name = os.path.basename(os.path.dirname(img_path))
        dst_dir = os.path.join(split_dir, class_name)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(img_path, os.path.join(dst_dir, os.path.basename(img_path)))


# 5. Архив
shutil.make_archive('preprocessed_images', 'zip', PROCESSED_DIR)

# 6. ОТЧЕТ
print("СТРУКТУРА ДАТАСЕТА:")
print("=" * 40)
sizes = Counter()
for split in ['train', 'test', 'valid']:
    split_dir = os.path.join(PROCESSED_DIR, split)
    if os.path.exists(split_dir):
        total = sum(len(files) for _, _, files in os.walk(split_dir))
        sizes[split] = total
        print(f"{split.upper()}: {total} изображений")

# 7. Демонстрация (2 примера + защита)
print("\nДЕМОНСТРАЦИЯ:")
classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
for i in range(2):
    cls = random.choice(classes)
    cls_dir = os.path.join(train_dir, cls)
    imgs = [f for f in os.listdir(cls_dir) if not f.endswith(('_tilted', '_corrupted'))]
    if imgs:
        print(f"\nПример {i + 1}:")
        augment_image(os.path.join(cls_dir, imgs[0]), show_results=True)

print("\nТест защиты:")
augment_image(os.path.join(cls_dir, imgs[0]))

shutil.rmtree(TEMP_DIR, ignore_errors=True)
```

