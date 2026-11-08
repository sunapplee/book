# Windows: подготовка к соревнованию

## Содержание

1. VPN
2. Проверка системы
3. Установка программ
4. VS Code
5. Python-окружение и Jupyter
6. Предварительная разметка проекта
7. Docker Desktop
8. Подготовка: проверка экспорта в HTML и PY
9. Запуск FastAPI и проверка работы в локальной сети с включённым VPN
10. Ollama
11. llama.cpp и Bonsai
12. Заготовить промпты

PowerShell, Windows 11 x64. Папка с материалами: `$env:USERPROFILE\book`; рабочий проект: `book\project`.

## 1. VPN

Запустить `requirements\lantern.exe` и подключиться. Другие варианты — в [vpn.md](vpn.md). Проверка локальной сети с VPN — пункт 9.

## 2. Проверка системы

```powershell
Get-ComputerInfo | Select-Object WindowsProductName, OsTotalVisibleMemorySize
Get-PSDrive -PSProvider FileSystem
nvidia-smi
Test-NetConnection github.com -Port 443
```

Файл подкачки: `SystemPropertiesAdvanced.exe` → Быстродействие → Параметры → Дополнительно → Виртуальная память.

Обновления:

```powershell
Start-Process "ms-settings:windowsupdate"
```

## 3. Установка программ

```powershell
winget install -e --id Git.Git
winget install -e --id Microsoft.VisualStudioCode
winget install -e --id Python.Python.3.12
winget install -e --id 7zip.7zip
winget install -e --id Gyan.FFmpeg
```

Перезапустить терминал:

```powershell
py -3.12 --version
ffmpeg -version
```

## 4. VS Code

```powershell
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension ms-toolsai.jupyter
code --install-extension ms-toolsai.jupyter-renderers
code --install-extension redhat.vscode-yaml

Set-Location "$env:USERPROFILE\book"
code .
```

Интерпретатор: **Python: Select Interpreter**. Ядро ноутбука: **Select Kernel**.

## 5. Python-окружение и Jupyter

```powershell
Set-Location "$env:USERPROFILE\book"
py -3.12 -m venv rea
.\rea\Scripts\Activate.ps1
```

Если активация заблокирована:

```powershell
Set-ExecutionPolicy -Scope Process RemoteSigned
.\rea\Scripts\Activate.ps1
```

В активированном окружении:

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r .\requirements\requirements-general.txt
python -m pip install jupyterlab notebook ipykernel nbconvert nbformat "python-lsp-server[all]" jupyterlab-lsp
python -m ipykernel install --user --name rea --display-name "Python (rea)"
python -m jupyterlab
```

Проверка интерпретатора в ноутбуке:

```python
import sys
print(sys.executable)
```

`Shift+Enter` — выполнить; `Esc`, затем `M` / `Y` — Markdown / код. `Ctrl+C` в терминале — остановить сервер.

## 6. Предварительная разметка проекта

В другом терминале:

```powershell
Set-Location "$env:USERPROFILE\book"
New-Item -ItemType Directory -Force -Path .\project, .\project\data, .\project\notebooks, .\project\models, .\project\reports, .\project\api, .\project\ui | Out-Null
```

В `notebooks` заранее создать ноутбуки с Markdown-шапками по пунктам задания. Данные — `data`, модели — `models`, отчёты и экспорты — `reports`, API и интерфейс — `api` и `ui`.

## 7. Docker Desktop

PowerShell **от администратора**:

```powershell
wsl --install --no-distribution
wsl --update
```

После необходимой перезагрузки:

```powershell
winget install -e --id Docker.DockerDesktop
```

Запустить Docker Desktop, выбрать WSL 2 backend:

```powershell
docker --version
docker compose version
docker run --rm hello-world
```

[Инструкция Docker](https://docs.docker.com/desktop/setup/install/windows-install/).

## 8. Подготовка: проверка экспорта в HTML и PY

В активированном `rea`, из папки `book`:

```powershell
python -c "import nbformat as n; nb=n.v4.new_notebook(cells=[n.v4.new_markdown_cell('# Проверка экспорта'), n.v4.new_code_cell('print(2 + 2)')]); nb.metadata.kernelspec={'name':'rea','display_name':'Python (rea)','language':'python'}; n.write(nb, 'project/notebooks/export_test.ipynb')"
python -m nbconvert --to html --execute --output-dir .\project\reports .\project\notebooks\export_test.ipynb
python -m nbconvert --to python --output-dir .\project\reports .\project\notebooks\export_test.ipynb
Start-Process .\project\reports\export_test.html
python .\project\reports\export_test.py
```

В HTML и выводе PY должно быть `4`. HTML повторно открыть после остановки Jupyter. Для рабочего ноутбука также проверить графики, карты и их управление.

[Экспорт nbconvert](https://nbconvert.readthedocs.io/en/latest/usage.html).

## 9. Запуск FastAPI и проверка работы в локальной сети с включённым VPN

В активированном `rea`, из папки `book`, создать проверочный API:

```powershell
python -m pip install fastapi uvicorn
@'
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}
'@ | Set-Content .\project\api\check_api.py -Encoding UTF8

python -m uvicorn check_api:app --app-dir .\project\api --host 0.0.0.0 --port 8000
```

В другом терминале:

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/health"
Get-NetIPAddress -AddressFamily IPv4 | Select-Object InterfaceAlias, IPAddress
```

Взять IPv4 интерфейса Wi-Fi/Ethernet. С другого ПК в той же сети проверить, подставив этот адрес:

```powershell
Invoke-RestMethod "http://192.168.1.10:8000/health"
```

Ожидается `status: ok`. Повторить проверку на сервере и втором ПК с включённым VPN; открыть `/docs` и проверить доступ в интернет.

Если мешает брандмауэр: для доверенной частной сети выполнить **от администратора**:

```powershell
New-NetFirewallRule -DisplayName "FastAPI 8000 (LAN)" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8000 -Profile Private -RemoteAddress LocalSubnet
```

Если доступ пропадает только с VPN — включить разрешение локальной сети или обход LAN в VPN-клиенте.

[Запуск Uvicorn](https://uvicorn.dev/settings/) · [Брандмауэр Windows](https://learn.microsoft.com/en-us/powershell/module/netsecurity/new-netfirewallrule).

## 10. Ollama

```powershell
Invoke-WebRequest "https://ollama.com/download/OllamaSetup.exe" -OutFile "$env:TEMP\OllamaSetup.exe"
Start-Process "$env:TEMP\OllamaSetup.exe" -Wait
```

Перезапустить терминал:

```powershell
ollama --version
ollama run llama3.2
ollama list
```

API: `http://localhost:11434`. Приложение работает в фоне.

## 11. llama.cpp и Bonsai

```powershell
irm https://llama.app/install.ps1 | iex
llama version
```

Если установщик запросит CUDA Toolkit:

```powershell
winget install -e --id Nvidia.CUDA
```

Перезапустить терминал и повторить установку llama.cpp. [Установщик](https://llama.app/install.ps1).

**Bonsai Q2_0: 2,56 GB, llama.cpp b11556+.** [Модель](https://huggingface.co/prism-ml/Ternary-Bonsai-2-9B-gguf#use-upstream-llamacpp).

```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\models\bonsai" | Out-Null
curl.exe -fL --retry 3 -o "$env:USERPROFILE\models\bonsai\Ternary-Bonsai-2-9B-Q2_0.gguf" "https://huggingface.co/prism-ml/Ternary-Bonsai-2-9B-gguf/resolve/main/Ternary-Bonsai-2-9B-Q2_0.gguf"

llama serve -m "$env:USERPROFILE\models\bonsai\Ternary-Bonsai-2-9B-Q2_0.gguf" --alias bonsai -ngl 99 -fa on -c 8192 -np 1 --host 127.0.0.1 --port 9931
```

CPU: заменить `-ngl 99` на `-ngl 0`. Интерфейс: [localhost:9931](http://localhost:9931). Остановка: `Ctrl+C`.

Проверка в другом терминале:

```powershell
Invoke-RestMethod "http://localhost:9931/health"

$requestJson = '{"model":"bonsai","messages":[{"role":"user","content":"Что такое RAG?"}],"max_tokens":256,"chat_template_kwargs":{"enable_thinking":false}}'
$response = Invoke-RestMethod -Uri "http://localhost:9931/v1/chat/completions" -Method Post -ContentType "application/json; charset=utf-8" -Body ([Text.Encoding]::UTF8.GetBytes($requestJson))
$response.choices[0].message.content
```
