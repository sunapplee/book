## 1. Установка (Linux, как в исходном ноутбуке)

Источник: `REA/Preparing_Rea2026123/AirFlow/airflow.ipynb`.
Версия сохранена из исходника. Импорты DAG и операторов приведены к [Airflow 3](https://airflow.apache.org/docs/apache-airflow/stable/operators-and-hooks-ref.html).

```bash
mkdir -p ~/ml_airflow
cd ~/ml_airflow
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
export AIRFLOW_HOME=~/airflow
mkdir -p "$AIRFLOW_HOME/dags"
AIRFLOW_VERSION=3.1.3
PYTHON_VERSION="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
pip install "apache-airflow==${AIRFLOW_VERSION}" apache-airflow-providers-standard --constraint "${CONSTRAINT_URL}"
pip install papermill jupyter
```

## 2. Ноутбук → модель и метрики

В `train.ipynb` задать `output_dir` в ячейке с тегом `parameters`:

```python
output_dir = "/tmp/airflow_default"
```

Сохранение после своего обучения:

```python
from pathlib import Path
import json

out = Path(output_dir)
out.mkdir(parents=True, exist_ok=True)
metrics = {"accuracy": 0.93, "f1": 0.90}  # Пример из исходника
(out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
# torch.save(model.state_dict(), out / "model.pt")
```

В [dag_notebook.py](dag_notebook.py) заменить `<USER>`, настроить пути и скопировать файл в `$AIRFLOW_HOME/dags/`.
Цепочка: создать папку → выполнить ноутбук через Papermill → проверить `metrics.json` и `model.pt`.
Проверка `model.pt` требует, чтобы ноутбук действительно сохранял модель.

## 3. Запуск

```bash
airflow standalone
```

UI: `http://localhost:8080`. Учётные данные — в выводе команды.
Включить DAG и нажать Trigger; терминал оставить открытым до завершения.
Из другого терминала с тем же окружением и `AIRFLOW_HOME`:

```bash
airflow dags trigger one_night_train_to_c_m
```

Результаты: `~/Desktop/c_m/executed_train_<дата>.ipynb`, `<дата>/metrics.json`, `<дата>/model.pt`.

## 4. Обучение по расписанию

[dag_train.py](dag_train.py) — сокращённый пример из `REA/Профессионалы2024/airflow/dags/main.py`.
`schedule="*/2 * * * *"` — каждые две минуты. Перед запуском настроить пути к SQLite и результатам.
Данные, библиотеки обучения и Jupyter-ядро должны быть доступны окружению, которое выполняет задачу.
