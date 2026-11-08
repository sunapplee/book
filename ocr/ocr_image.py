# OCR изображения. Источник: sibur_preparing/completed/OCR.py
IMG = "images.jpeg"

# 1. EasyOCR: pip install easyocr; gpu=True — как в исходнике
import easyocr

reader = easyocr.Reader(["ru", "en"], gpu=True)
result = reader.readtext(IMG)

lines, cur, prev_y = [], [], None
for bbox, text, conf in result:
    y = (bbox[0][1] + bbox[2][1]) / 2
    if prev_y is not None and abs(y - prev_y) > 10:
        lines.append(" ".join(cur))
        cur = []
    cur.append(text)
    prev_y = y
lines.append(" ".join(cur))

print(*lines, sep="\n")

# 2. GLM-OCR через Transformers
# pip install git+https://github.com/huggingface/transformers.git
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

print(torch.cuda.is_available())

MODEL_PATH = "zai-org/GLM-OCR"
processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="auto",
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": IMG},
            {"type": "text", "text": "Распознай весь текст. Сохрани структуру, верни Markdown."},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)
inputs.pop("token_type_ids", None)

generated = model.generate(**inputs, max_new_tokens=8192)
print(processor.decode(
    generated[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True,
))

# 3. GLM-OCR через нативный API Ollama
# ollama pull glm-ocr:latest; ollama serve
import base64
import requests

with open(IMG, "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "glm-ocr:latest",
        "prompt": "Распознай весь текст на изображении. Сохрани структуру и верни Markdown.",
        "images": [image_b64],
        "stream": False,
    },
)
response.raise_for_status()

print(response.json()["response"])
