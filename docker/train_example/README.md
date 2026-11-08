## Обучение в контейнере

Источник: `sibur_preparing/completed/Dockerfile` и `sibur_preparing/train_model`.

В `train_model/` включены исходные `X_train.npy`, `y_train.npy`, `X_test.npy` из `sibur_preparing/train_model/`.
Команды выполнять из `book/docker/train_example/`.

```bash
docker build . -t train
```

Windows (PowerShell):

```powershell
docker run -v "${PWD}/outputs:/app/outputs" train
```

Linux:

```bash
docker run -v "$(pwd)/outputs:/app/outputs" train
```

Результаты снаружи контейнера: `outputs/model.pkl`, `outputs/predictions.txt`.
