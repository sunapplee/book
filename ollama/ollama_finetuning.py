# Для использования этой модели в ollama используем 
# ```ollama create [NAME] -f Modelfile```


import json

# Загружаем датасет JSON
# Датасет формата
#{'input': "some input",
#'output': "some output"}

file = json.load(open("json_extraction_dataset_500.json", "r"))
print(file[1])

# Проверка GPU — важно для корректного запуска LoRA + 4bit
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

from unsloth import FastLanguageModel

# Имя модели — 4bit версия Qwen3 (экономия VRAM)
model_name = "unsloth/Qwen3-0.6B-unsloth-bnb-4bit"
max_seq_length = 2048   # Максимальная длина входной последовательности
dtype = None            # Автодетект

# Загружаем модель + токенайзер с поддержкой 4bit
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=True,
)

from datasets import Dataset

# Форматируем данные в строку — SFTTrainer принимает plain text
def format_prompt(example):
    # Жёсткая структура prompt → модель лучше учится
    return f"### Input: {example['input']}\n### Output: {json.dumps(example['output'])}<|endoftext|>"

formatted_data = [format_prompt(item) for item in file]
dataset = Dataset.from_dict({"text": formatted_data})

print(dataset[0])

# Подключаем LoRA — основной механизм дообучения без переписывания самой модели
model = FastLanguageModel.get_peft_model(
    model,
    r=64,  # Размер низкоранговых матриц (больше = качественнее, но больше VRAM)
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],  # Основные матрицы внимания и MLP в Qwen
    lora_alpha=128,   # Коэффициент масштабирования
    lora_dropout=0,   # 0 даёт ускорение на Unsloth
    bias="none",      # Экономия VRAM
    use_gradient_checkpointing="unsloth",  # Снижает VRAM ценой небольшой потери скорости
    random_state=3407,
    use_rslora=False, 
    loftq_config=None,
)

from trl import SFTTrainer
from transformers import TrainingArguments

# Настройки обучения — оптимизированы под Unsloth
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",   # Поле с текстом
    max_seq_length=max_seq_length,
    dataset_num_proc=2,          # Параллельная обработка
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,  # Эффективный batch = 2×4 = 8
        warmup_steps=10,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),  # Используем bfloat16, если GPU позволяет
        logging_steps=25,
        optim="adamw_8bit",       # Оптимизатор в 8bit снижает VRAM
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        save_strategy="epoch",    # Сохранения по эпохам
        save_total_limit=2,
        dataloader_pin_memory=False,
    ),
)

# Запускаем обучение
trainer_stats = trainer.train()

# Включаем быстрый режим инференса Unsloth (ускоряет x2)
FastLanguageModel.for_inference(model)

# Тестовый запрос после fine-tuning
messages = [
    {"role": "user", "content": "Extract the product information:\n<div class='product'><h2>iPad Air</h2><span class='price'>$1344</span><span class='category'>audio</span><span class='brand'>Dell</span></div>"},
]

# Превращаем диалог в формат, который понимает Qwen
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")

# Генерация ответа
outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=256,
    use_cache=True,
    temperature=0.7,
    do_sample=True,
    top_p=0.9,
)

# Вывод результата
response = tokenizer.batch_decode(outputs)[0]
print(response)

# Экспорт модели в GGUF для использования в llama.cpp / ollama
model.save_pretrained_gguf("gguf_model", tokenizer, quantization_method="q4_k_m")
