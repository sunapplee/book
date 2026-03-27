# 🔹 0. Подготовка к соревнованию (день 0)

## 0.1. Swap-файл (150–200 GB)

```bash
sudo fallocate -l 150G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
sudo swapon --show

```

## 0.2. Переключение GPU через prime-select

```bash
# Проверить текущую видеокарту
prime-select query

# Переключить на интегрированную (Intel) — освобождает 50–100 MB GPU
sudo prime-select intel

# Переключить на дискретную (NVIDIA)
sudo prime-select nvidia

# Гибридный режим
sudo prime-select on-demand
```

## 0.3. Алиасы для консоли

```bash
echo "alias ca='conda activate'" >> ~/.bashrc
echo "alias ci='conda install'" >> ~/.bashrc
echo "alias ll='ls -lah'" >> ~/.bashrc
echo "alias jl='jupyter lab'" >> ~/.bashrc
source ~/.bashrc
```

## 0.4. Настройка JupyterLab (LSP + автодополнение)

```bash
pip install 'python-lsp-server[all]' jupyterlab-lsp

# Сгенерировать конфиг
jupyter lab --generate-config
# Файл: ~/.jupyter/jupyter_lab_config.py
```

Добавить в `~/.jupyter/jupyter_lab_config.py`:

```python
c.LanguageServerManager.language_servers = {
    "pylsp": {
        "version": 2,
        "argv": ["pylsp"],
        "languages": ["python"],
        "mime_types": ["text/x-python"]
    }
}
```

Ключевые параметры ноутбука в JupyterLab Settings:

```
Shut down kernel when closing notebook = true
Maximum number of output cells = 50
Show minimap = true
Code Folding = true
Line Numbers = true
Highlight active line = true
```

## 0.5. Проверка Python-интерпретатора

```python
import sys
sys.executable   # убедиться, что используется нужное окружение
```

## 0.6. Горячие клавиши Jupyter

| Клавиша   | Действие                     |
| --------- | ---------------------------- |
| `C` / `V` | Копировать / вставить ячейку |
| `Y`       | Переключить в режим кода     |
| `M`       | Переключить в режим Markdown |

---

# 🔹 1. подтягивание файлов из git через curl wget

## Что должно стоять

```bash
sudo apt update && sudo apt install -y wget curl unzip git jupyter
```

## Скачать файл

```bash
# Вариант 1: wget
wget https://raw.githubusercontent.com/sunapplee/river/main/requirements/scripts.ipynb

# Вариант 2: curl
curl -L -O https://raw.githubusercontent.com/sunapplee/river/main/requirements/scripts.ipynb

# Вариант 3: curl и запуск в консоли
curl https://raw.githubusercontent.com/sunapplee/river/main/requirements/scripts.md
```

_Расположение файлов доступно в `content.md`_

## Запустить Jupyter

```bash
jupyter notebook docker.ipynb
```

# 🔹 Установка день 0

# 0. Базовая информация и проверки

## Проверка ОС и системных ресурсов

```bash
uname -a
lsb_release -a  # если есть
df -h /        # свободное место на корневом разделе
free -h        # RAM
nvidia-smi     # если есть GPU NVIDIA
```

## Проверка интернета

```bash
ping -c 3 google.com
```

---

# 1. Обновление системы

```bash
sudo apt update
sudo apt upgrade -y
```

---

# 2. Базовые пакеты

```bash
sudo apt install -y \
  curl \
  wget \
  git \
  build-essential \
  ca-certificates \
  software-properties-common \
  gnupg \
  unzip \
  htop \
  tree \
  ffmpeg \
  poppler-utils \
  tesseract-ocr
```

---

# 3. Установка Python

## Проверка версии

```bash
python3 --version
```

## Установка Python3 (если не установлен)

```bash
sudo apt install -y python3 python3-pip python3-venv
```

## Проверка установки

```bash
python3 --version
pip3 --version
```

---

# 4. Установка VS Code

## Добавление репозитория Microsoft

```bash
wget -qO- https://packages.microsoft.com/keys/microsoft.asc \
  | sudo gpg --dearmor \
  > /usr/share/keyrings/microsoft-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/microsoft-archive-keyring.gpg] https://packages.microsoft.com/repos/code stable main" \
  | sudo tee /etc/apt/sources.list.d/vscode.list
```

## Установка

```bash
sudo apt update
sudo apt install -y code
```

## Запуск

```bash
code
```

---

# 5. Расширения VS Code

В VS Code откройте панель расширений:

- `Ctrl+Shift+X` → установить:
  - Python (Microsoft)
  - Pylance
  - Jupyter
  - Jupyter Notebook Renderers
  - Docker (Microsoft)
  - YAML

После установки перезапустите VS Code.

---

# 6. Структура проекта и репозиторий

## Инициализация Git

```bash
# Проверяем, что мы в Rea
pwd

git init
git remote add origin https://github.com/твойusername/Rea.git
git add .
git commit -m "initial commit"
git branch -M main
git push -u origin main
```

---

# 7. Общее Python-окружение `rea`

## Создание окружения

```bash
cd ~/Rea
python3 -m venv rea
source rea/bin/activate
```

## Проверка

```bash
python -V
pip -V
```

---

# 9. Общее окружение `rea`

Базовое окружение для ML, геоданных, CV, web-разработки.

```bash
cd ~/Rea
source rea/bin/activate
pip install -r requirements-general.txt
```

**Зависимости:** [requirements-general.txt](requirements-general.txt)

---

# 10. Окружение PyTorch

Для проектов с deep learning (CUDA, vision, метрики).

```bash
cd ~/Rea
python3 -m venv pytorch_env
source pytorch_env/bin/activate
pip install --upgrade pip

# Установка PyTorch (выберите версию CUDA на pytorch.org)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Дополнительные пакеты
pip install -r requirements-torch.txt
```

**Зависимости:** [requirements-torch.txt](requirements-torch.txt)

---

# 11. Окружение Unsloth

Для дообучения LLM моделей (требует PyTorch + CUDA).

```bash
cd ~/Rea
python3 -m venv unsloth_env
source unsloth_env/bin/activate
pip install --upgrade pip

# Сначала PyTorch (см. раздел 10)
# Затем следуйте инструкциям в файле зависимостей
```

**Зависимости:** [requirements-unsloth.txt](requirements-unsloth.txt)

---

# 12. Использование окружений

```bash
# Общее окружение
cd ~/Rea
source rea/bin/activate

# PyTorch
source ~/Rea/pytorch_env/bin/activate

# Unsloth
source ~/Rea/unsloth_env/bin/activate

# Деактивация
deactivate
```

# 13. Установка Docker

```bash
sudo apt install docker.io -y
```

Если сервис не стартовал:

```bash
sudo systemctl start docker
sudo systemctl enable docker
```

## Настройка прав (чтобы запускать docker без sudo)

```bash
sudo usermod -aG docker ruslan
```

> Выйдите и зайдите в систему, чтобы группа применилась.

---

# 15. Установка Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

## Проверка

```bash
ollama --version
```

---
