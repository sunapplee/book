
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

```bash
pip install -r requirements.txt
```

### 2. Запуск сервера

```bash
uvicorn api:app --host 0.0.0.0 --port 8002 --reload
```

После запуска API будет доступен по адресу:

* `http://localhost:8002`
* `http://localhost:8002/docs`

---

## Структура проекта

```bash
.
├── api.py
├── audio.py
├── image.py
├── geo.py
├── tables.py
├── test_api.py
├── requirements.txt
├── audio_models/
├── image_models/
├── table_models/
└── geo_models/
```

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
python test_api.py
```

---

## Возможные ошибки

* **Порт занят** → сменить порт или завершить процесс
* **Нет модели** → проверить папки `*_models/`
* **Нет библиотеки** → установить зависимости из `requirements.txt`
* **Неверный формат файла** → использовать корректный тип входных данных

---

## Docker (опционально)

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8002

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8002"]
```
