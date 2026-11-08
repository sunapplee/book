import uvicorn
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from PIL import Image
import io
import torchvision.transforms.v2 as tfs 
from torchvision.models import segmentation
import torch
from torch import nn
from skimage import morphology
from skimage.measure import label, regionprops
import cv2
import numpy as np

app = FastAPI()

class ImageField(BaseModel):
    filename: str

# Загружаем модель LR-ASPP
segment_model = segmentation.lraspp_mobilenet_v3_large(
    weights="DEFAULT", # предобученные веса
    progress=True
)

# Меняем число классов
segment_model.classifier.low_classifier  = nn.Conv2d(40, 1, kernel_size=1)
segment_model.classifier.high_classifier = nn.Conv2d(128, 1, kernel_size=1)

# Загружаем веса модели
segment_model.load_state_dict(torch.load('best_lraspp.pth', weights_only=True))
segment_model.eval()


# функция загрузки изображения/изображений
@app.post('/load_images')
async def load_images(request: Request):
    # Обработка списка изображений
    form = await request.form()
    for data in form:
        contents = form[data].file.read()
        # Сохраняем
        with open(f'original_files/{form[data].filename}', 'wb') as f:
            f.write(contents)

    # Возвращаем названия файлов
    return {"file_names": [form[file].filename for file in form]}


# функция обработки изображения
@app.post('/preprocess_image')
def preprocess_image(image: ImageField):
    img_path = 'original_files/' + image.filename

    # открываем через PIL
    pil_img = Image.open(img_path).convert("RGB")

    # (меняем размер)
    resized = pil_img.resize((224, 224))
    save_path = f"processed_files/{image.filename}"
    resized.save(save_path)

    # Возвращаем названия файлов
    return {'filename': image.filename}


# функция для сегментации изображений
@app.post('/segment_image')
def segment_image(image: ImageField):
    img_path = 'original_files/' + image.filename
    
    # Обработка перед инференсом
    transform = tfs.Compose(
        [tfs.ToImage(),
         tfs.Resize((224, 224)),
         tfs.ToDtype(torch.float32, scale=True),
        #  tfs.Normalize(
        #      [0.485, 0.456, 0.406],
        #      [0.229, 0.224, 0.225]
        #  )
         ]
    )

    # Загружаем изображение
    img = Image.open(img_path)
    img2infer = transform(img)
    # Добавляем ось
    img2infer = torch.unsqueeze(img2infer, dim=0)
    # Инференс
    with torch.no_grad():
        out = segment_model(img2infer)
        out = out['out']
        mask = (out > 0.5).float()

    # Преобразрование в numpy
    mask_np = mask.squeeze().cpu().numpy()
    mask_uint8 = (mask_np * 255).astype("uint8")

    # Морфология
    mask_uint8 = morphology.closing(morphology.opening(mask_uint8))

    mask_pil = Image.fromarray(mask_uint8)
    # Сохраняем маску
    mask_pil.save(f'segmentation/{image.filename}')

    # Возвращаем изображение в байтах
    buffer = io.BytesIO()
    mask_pil.save(buffer, format="PNG")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")


# функция визуализации результатов сегментации
@app.get('/visualize_image')
def visualize_image(image_filename: str):
    # Загружаем изображение
    img_path = 'original_files/' + image_filename
    img = cv2.imread(img_path)
    # Загружаем маску
    mask_path = 'segmentation/' + image_filename
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # Бинаризация маски (избавляемся от артефактов)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Нахождение контуров
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Наложение контуров на оригинал
    result = np.copy(img)
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

    # Сохранение результата
    cv2.imwrite('result/' + image_filename, result)

    # Возвращаем изображение в байтах
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(result_rgb)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")


# функция определения площади полей
@app.get('/get_area')
def get_area(image_filename: str):
    
    # Загружаем маску
    mask_path = 'segmentation/' + image_filename
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # Бинаризация маски (избавляемся от артефактов)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Выделяем области на маске
    label_image = label(binary_mask)
    regions = regionprops(label_image)

    # Масштаб 10 метров
    resolution = 10
    
    # Проходимся по каждому кластеру и вычисляем площадь каждого
    areas = []
    for region in regions:
        # Площадь в М2
        area = region.area * (resolution ** 2)
        # Площадь в га
        area = area / 10_000
        areas.append({'claster': region.label, 'area': area})

    return {'result': areas}


# Запуск
if __name__ == '__main__':
    uvicorn.run(app, port=8080, host='0.0.0.0')