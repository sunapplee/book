# Источник: sibur_preparing/completed/Dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY train_model/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY train_model/ .
CMD ["python", "train.py"]
