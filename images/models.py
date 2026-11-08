---

## **0. Установка зависимостей**

```bash
pip install torch torchvision scikit-learn ultralytics
```

---

## **Блок 1. Распаковка архива**

```python
import os
import zipfile

ARCHIVE_NAME = "preprocessed_images.zip"
EXTRACT_DIR = "preprocessed_images"

if not os.path.exists(EXTRACT_DIR):
    with zipfile.ZipFile(ARCHIVE_NAME, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
    print(f"Архив {ARCHIVE_NAME} распакован в {EXTRACT_DIR}")
else:
    print(f"{EXTRACT_DIR} уже существует, пропускаем распаковку")
```

---

## **Блок 2. Импорты и подготовка данных**

```python
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.nn as nn
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from ultralytics import YOLO
import numpy as np

device = "cpu"

# Трансформации без изменения размера (224x224 уже)
tfm = transforms.Compose([transforms.ToTensor()])

train_ds = datasets.ImageFolder(f"{EXTRACT_DIR}/train", transform=tfm)
val_ds   = datasets.ImageFolder(f"{EXTRACT_DIR}/valid", transform=tfm)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=16)

num_classes = len(train_ds.classes)
print(f"Найдено {num_classes} классов: {train_ds.classes}")
```

---

## **Блок 3. Универсальный пайплайн для обучения моделей**

```python
import time

def train_model(model_name: str, kwargs: dict = {}):
    """
    model_name: 'yolo', 'resnet', 'svm'
    kwargs: словарь с гиперпараметрами модели
    """
    start = time.time()

    # YOLO
    if model_name.lower() == 'yolo':
        model = YOLO(kwargs.get("weights", "yolov8s-cls.pt"))
        model.train(
            data=kwargs.get("data", EXTRACT_DIR),
            epochs=kwargs.get("epochs", 11),
            imgsz=kwargs.get("imgsz", 224),
            batch=kwargs.get("batch", 16)
        )
        val_res = model.val(data=kwargs.get("data", EXTRACT_DIR))
        metrics = {
            "accuracy": float(val_res.top1),
            "f1_macro": float(val_res.top1),
            "roc_auc": 0.0
        }

    # ResNet18
    elif model_name.lower() == 'resnet':
        model = models.resnet18(weights="DEFAULT")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.to(device)

        epochs = kwargs.get("epochs", 12)
        lr = kwargs.get("lr", 1e-3)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.1)

        for epoch in range(epochs):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                out = model(x)
                loss = loss_fn(out, y)
                loss.backward()
                opt.step()
            scheduler.step()

        # Валидация
        model.eval()
        y_true, y_pred, y_prob = [], [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                out = model(x)
                prob = torch.softmax(out, 1)
                y_true.extend(y.numpy())
                y_pred.extend(prob.argmax(1).cpu().numpy())
                y_prob.extend(prob.cpu().numpy())

        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
            "roc_auc": float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
        }

    # SVM
    elif model_name.lower() == 'svm':
        subset_size = kwargs.get("subset_size", 5000)
        X_train, y_train = [], []
        count = 0
        for x, y in train_loader:
            for i in range(len(y)):
                if count >= subset_size:
                    break
                X_train.append(x[i].view(-1))
                y_train.append(y[i].item())
                count += 1
            if count >= subset_size:
                break
        X_train = torch.stack([torch.tensor(x) for x in X_train]).numpy()
        y_train = np.array(y_train)

        X_val, y_val = [], []
        for x, y in val_loader:
            for i in range(len(y)):
                X_val.append(x[i].view(-1))
                y_val.append(y[i].item())
        X_val = torch.stack([torch.tensor(x) for x in X_val]).numpy()
        y_val = np.array(y_val)

        model = SVC(
            kernel=kwargs.get("kernel","rbf"),
            C=kwargs.get("C",10),
            gamma=kwargs.get("gamma","scale"),
            probability=True
        )
        model.fit(X_train, y_train)

        pred = model.predict(X_val)
        prob = model.predict_proba(X_val)

        metrics = {
            "accuracy": float(accuracy_score(y_val, pred)),
            "f1_macro": float(f1_score(y_val, pred, average="macro")),
            "roc_auc": float(roc_auc_score(y_val, prob, multi_class="ovr"))
        }

    else:
        raise ValueError("Модель должна быть одной из: 'yolo', 'resnet', 'svm'")

    end = time.time()
    epochs = kwargs.get("epochs", 1)
    elapsed_sec = (end - start) / epochs
    elapsed_min = (end - start) / 60
    
    
    # === ДОБАВЛЕННЫЙ БЛОК СОХРАНЕНИЯ РЕЗУЛЬТАТОВ ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"model_{model_name}_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"Модель: {model_name}\n")
        f.write(f"Гиперпараметры: {kwargs}\n")
        f.write(f"Метрики на валидации:\n")
        for k, v in metrics.items():
            f.write(f"  {k}: {v:.4f}\n")
        f.write(f"Время на эпоху: {elapsed_sec:.2f} сек\n")
        f.write(f"Общее время: {elapsed_min:.2f} мин\n")
    # =============================================

    return model, metrics, elapsed_sec, elapsed_min
```

---

## **Блок 4. Обучение YOLOv8s**

```python
yolo_model, yolo_metrics, yolo_sec, yolo_min = train_model(
    'yolo',
    kwargs={"epochs":11, "batch":16, "imgsz":224}
)
print("YOLOv8s метрики:", yolo_metrics)
print(f"Время на эпоху: {yolo_sec:.2f} сек, всего: {yolo_min:.2f} мин")
```

---

## **Блок 5. Обучение ResNet18**

```python
resnet_model, resnet_metrics, resnet_sec, resnet_min = train_model(
    'resnet',
    kwargs={"epochs":12, "lr":1e-3}
)
print("ResNet18 метрики:", resnet_metrics)
print(f"Время на эпоху: {resnet_sec:.2f} сек, всего: {resnet_min:.2f} мин")
```

---

## **Блок 6. Обучение SVM**

```python
svm_model, svm_metrics, svm_sec, svm_min = train_model(
    'svm',
    kwargs={"subset_size":5000, "C":10, "gamma":"scale"}
)
print("SVM метрики:", svm_metrics)
print(f"Время на обучение: {svm_min:.2f} мин")
```

---

