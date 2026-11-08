
# Документация REST API

## Описание

REST API для работы с несколькими ML-моделями:

* **Табличные данные** — прогноз AQI
* **Аудио** — классификация звуков
* **Изображения** — классификация фруктов
* **Геоданные** — регрессия по признакам

**Стек:** FastAPI, PyTorch, CatBoost, scikit-learn
**Порт:** `8002`
**Swagger:** `http://localhost:8002/docs`

---

## Запуск

### 1. Установка зависимостей

Команды выполнять из `book/API/`.

```bash
pip install -r requirements-multimodal.txt
```

### 2. Запуск сервера

```bash
uvicorn multimodal_ml_api:app --host 0.0.0.0 --port 8002 --reload
```

После запуска API будет доступен по адресу:

* `http://localhost:8002`
* `http://localhost:8002/docs`

---

## Структура проекта

```bash
.
├── multimodal_ml_api.py
├── audio.py
├── image.py
├── geo.py
├── tables.py
├── multimodal_ml_test_api.py
├── requirements-multimodal.txt
├── audio_models/
├── image_models/
├── table_models/
└── geo_models/
```

Папки `*_models/` — артефакты исходного проекта `REA/Preparing_Rea2026123/Module 4/API/`.
Код вспомогательных модулей включён в `book/API/`; веса и датасеты нужно разместить рядом самостоятельно.
`image.py` читает `image_models/data_fruits/train/` уже при импорте, для оценки дообучения нужна также папка `test/`.
Табличное и аудио-дообучение используют сохранённые train/test-наборы из соответствующих папок моделей.

---

## Эндпоинты API

| Метод  | Эндпоинт                   | Назначение                       |
| ------ | -------------------------- | -------------------------------- |
| `GET`  | `/`                        | Проверка работы API              |
| `GET`  | `/table_inference`         | Предсказание по табличным данным |
| `POST` | `/finetuning_table_single` | Дообучение на одном примере      |
| `POST` | `/finetuning_table_batch`  | Дообучение на CSV                |
| `POST` | `/audio_inference`         | Классификация аудио              |
| `POST` | `/finetuning_audio`        | Дообучение аудио-модели          |
| `POST` | `/image_inference`         | Классификация изображения        |
| `POST` | `/finetuning_image`        | Дообучение image-модели          |
| `GET`  | `/geo_inference`           | Предсказание по геоданным        |

---

## Примеры запросов

### Проверка API

```bash
curl http://localhost:8002/
```

### Табличный инференс

```bash
curl "http://localhost:8002/table_inference?CO=300.3&NO2=20.5&SO2=3.1&O3=32.2&PM25=15.0&PM10=16.6"
```

### Аудио-инференс

```bash
curl -X POST "http://localhost:8002/audio_inference" -F "file=@dog.wav"
```

### Инференс изображения

```bash
curl -X POST "http://localhost:8002/image_inference" -F "file=@banana.jpg"
```

---

## Используемые модели

* **Табличные данные:** CatBoost, MLPRegressor
* **Аудио:** CatBoostClassifier + MFCC
* **Изображения:** ResNet18
* **Геоданные:** joblib-модель

---

## Тестирование

```bash
python multimodal_ml_test_api.py
```

---

## Возможные ошибки

* **Порт занят** → сменить порт или завершить процесс
* **Нет модели** → проверить папки `*_models/`
* **Нет библиотеки** → установить зависимости из `requirements-multimodal.txt`
* **Неверный формат файла** → использовать корректный тип входных данных

---

## Docker (опционально)

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements-multimodal.txt

EXPOSE 8002

CMD ["uvicorn", "multimodal_ml_api:app", "--host", "0.0.0.0", "--port", "8002"]
```
