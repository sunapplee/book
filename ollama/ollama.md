
## **1. Установка**

```bash
curl -fsSL https://ollama.com/install.sh | sh
```
---

## **2. Загрузка моделей**

Показать доступные модели:

```bash
ollama list
```

### **Базовые LLM**

```bash
ollama pull qwen3:4b
```

### **Модели с визуалом (VLM)**

```bash
ollama pull qwen3-vl:8b
```

### **Эмбеддинги**

```bash
ollama pull qwen3-embedding:0.6b
```

> **Примечание:**
> У большинства моделей можно выбирать количество параметров:
> например `qwen3:4b`, `qwen3:7b`, `qwen3:14b`.

> После скачивания модель сразу доступна для `ollama run` и через API на `localhost:11434`.

---

## **3. Инференс через API (Python)**

Установка клиента:

```bash
pip install ollama
```

Пример использования **трёх моделей**:

```python
import ollama

# Можно указать любой URL сервера Ollama, если он не локальный
# Например: http://192.168.1.10:11434 или сервер в Docker
# ollama.api_base = "http://localhost:11434"

# -------- 1. Текстовая модель ----------
resp_text = ollama.chat(
    model="qwen3",
    messages=[{"role": "user", "content": "Привет! Что ты умеешь?"}]
)
print("Текст:", resp_text["message"]["content"])


# -------- 2. VLM (мультимодальная) ----------
resp_vlm = ollama.chat(
    model="qwen3-vl",
    messages=[
        {"role": "user", "content": "Опиши изображение", "images": ["image.png"]}
    ]
)
print("VLM:", resp_vlm["message"]["content"])


# -------- 3. Эмбеддинги ----------
resp_emb = ollama.embeddings(
    model="qwen3-embedding",
    prompt="пример текста"
)
print("Эмбеддинги:", resp_emb["embeddings"][0][:10])  # первые 10 значений
```

---

## **4. Локальный REST API (если хочешь отправлять HTTP-запросы)**

После запуска Ollama доступно:

```
http://localhost:11434/api/generate
http://localhost:11434/api/chat
http://localhost:11434/api/embeddings
```

Пример curl:

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "qwen3:4b",
  "messages": [
    {
      "role": "user",
      "content": "Hello!"
    }
  ]
}'
```

---


## **5. Загрузка моделей из Hugging Face**

Если нужной модели нет в списке **Ollama Models**, её можно скачать напрямую с Hugging Face.

⚠️ **Важно:** Ollama поддерживает *только формат* **GGUF**.

---

### **Загрузка HF-модели в Ollama**

Используется единый формат:

```bash
ollama pull hf.co/{username}/{repo}
```

Пример (Qwen3 в GGUF-формате):

```bash
ollama pull hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M
```

После загрузки модель автоматически доступна для `ollama run`, `api/chat`, Python-клиента.

---

### **Запуск HF-модели через REST API**

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M",
  "messages": [
    {
      "role": "user",
      "content": "Hello!"
    }
  ]
}'
```
