from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from typing import Annotated
import uvicorn
from transformers import pipeline
from contextlib import asynccontextmanager
from glmocr import GlmOcr
import pytesseract
from doclayout_yolo import YOLOv10
from pdf2image import convert_from_bytes
import base64
from io import BytesIO
import cv2

# Путь к tesseract
pytesseract.pytesseract.tesseract_cmd = r"E:\tesseract\tesseract.exe"

ml_models = {}


# Функция обработки изображения
def get_bytes_image(img):
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def ocr_pytesseract(images):
    results = []
    for img in images:
        text = pytesseract.image_to_string(
        img,
        lang="rus+eng",
        config="--oem 3 --psm 6"
    )   
        results.append(text)

    return '\n'.join(results)


def ocr_glm(images):
    # Инициализация через кастомный конфиг (с олламой)
    with GlmOcr(config_path="../config.yaml") as parser:
        # Передаем список изображений
        result = parser.parse([get_bytes_image(img) for img in images])
        # Объединяем текст
        text = '\n'.join([page.markdown_result for page in result])

        return text


@asynccontextmanager
async def lifespan(app: FastAPI):

    model_ner   = pipeline(
            'ner',
            model="../wnut_model",        # путь к папке
            tokenizer="../wnut_model",    # тот же путь
            aggregation_strategy="simple"
        )

    # Загружаем модель с заранее скачанными весами
    model_yolo = YOLOv10("../doclayout_yolo_docstructbench_imgsz1024.pt")

    ml_models["ner"] = model_ner
    ml_models["yolo_layout"] = model_yolo
    ml_models["glm_ocr"] = ocr_glm
    ml_models["tesseract_ocr"] = ocr_pytesseract


    yield
    # Clean up the ML models and release the resources
    ml_models.clear()



app = FastAPI(lifespan=lifespan)

@app.get('/')
async def get_func():
    return {'message': 'hello'}


@app.post('/tesseract_ocr')
async def post_tesseract_ocr(files: Annotated[list[UploadFile], File()]):
    images = []
    for file in files:
        bytes_pdf = await file.read()
        images.extend(convert_from_bytes(bytes_pdf, poppler_path=r"C:\poppler\Library\bin"))

    text = ml_models['tesseract_ocr'](images)

    return {'result': text}


@app.post('/glm_ocr')
async def post_glm_ocr(files: Annotated[list[UploadFile], File()]):
    
    images = []
    for file in files:
        bytes_pdf = await file.read()
        images.extend(convert_from_bytes(bytes_pdf, poppler_path=r"C:\poppler\Library\bin"))

    text = ml_models['glm_ocr'](images)

    return {'result': text}


@app.post('/yolo_layout')
async def post_yolo_layout(files: Annotated[list[UploadFile], File()]):
    images = []
    for file in files:
        bytes_pdf = await file.read()
        images.extend(convert_from_bytes(bytes_pdf, poppler_path=r"C:\poppler\Library\bin"))

    result = ml_models['yolo_layout'].predict(
        images,
        imgsz=1024,
        conf=0.2,
        device="cuda:0"
    )

    images_paths = []

    for i, page in enumerate(result):
        img = page.orig_img.copy()
        boxes = page.boxes

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                cls_id = int(box.cls[0])
                score = float(box.conf[0])
                label = page.names[cls_id]

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    img,
                    f"{label} ({score:.2f})",
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 0, 255),
                    2
                )

        filepath = f'layout_outputs/page_{i}.jpg'
        cv2.imwrite(filepath, img)

        images_paths.append(filepath)

    return {"paths": images_paths}


@app.get('/ner')
async def get_ner(text: str):
    result = ml_models["ner"](text)
    # Обработка вывода
    for en in result:
        en['score'] = float(en['score'])
    return {"result": result}


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8660)