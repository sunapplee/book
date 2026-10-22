## 1. Установка

Windows (PowerShell):

```powershell
irm https://llama.app/install.ps1 | iex
```

Linux:

```bash
curl -LsSf https://llama.app/install.sh | sh
```

## 2. GGUF-модель и запуск

Скачать GGUF-файл с Hugging Face, например [Ternary-Bonsai-2-9B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-9B-gguf/tree/main).
Вместо `-m` можно использовать `-hf` для загрузки с Hugging Face.

```powershell
# Общение в терминале
llama cli -m .\Ternary-Bonsai-2-9B-Q2_0.gguf

# API: контекст 8192 токена, один слот
llama serve -m .\Ternary-Bonsai-2-9B-Q2_0.gguf -c 8192 -np 1 --port 9931

# Только CPU
llama serve -m .\Ternary-Bonsai-2-9B-Q2_0.gguf -ngl 0 -c 8192 -np 1 --port 9931
```

## 3. Запрос через Python

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:9931/v1",
    api_key="sk-no-key-required",
)
completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Привет"}],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(completion.choices[0].message)
```

## 4. Эмбеддинги

Скачать GGUF-файл модели эмбеддингов и запустить отдельный сервер:

```powershell
llama serve -m .\embeddinggemma-2-Q8_0.gguf --embeddings -c 8192 -np 1 --port 9932
```

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:9932/v1",
    api_key="sk-no-key-required",
)
emb = client.embeddings.create(
    input="Your text string goes here",
    model=r".\embeddinggemma-2-Q8_0.gguf",
).data[0].embedding
print(emb[:10])
```
