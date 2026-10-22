# Промптинг. Источник: sibur_preparing/completed/prompts.py
# pip install langchain-ollama pydantic
# ollama pull qwen3.5:4b

from collections import Counter
from typing import Literal
from pydantic import BaseModel, Field
from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

llm = ChatOllama(model="qwen3.5:4b", reasoning=False, temperature=0)

# 1. Structured output — ответ как объект Pydantic; неизвестный цвет = None
class Product(BaseModel):
    category: Literal["Ноутбук", "Телевизор", "Смартфон", "Пылесос", "Другое"] = Field(description="Категория товара")
    color: str | None = Field(default=None, description="Цвет товара на русском языке; None, если цвет не указан")


structured_llm = llm.with_structured_output(Product)
messages = [
    SystemMessage("Извлекай характеристики товара из описания. Опирайся только на текст, ничего не придумывай."),
    HumanMessage("Телевизор 80 дюймов FullHD"),
]
print(structured_llm.invoke(messages))

# 2. Few-shot — формат задаётся примерами входа и выхода
few_shot_prompt = '''
Приведи название товара к короткому стандартному виду.

Пример 1:
Вход: Apple iPhone 15 Pro Max 256GB Natural Titanium
Выход: iPhone 15 Pro Max 256GB

Пример 2:
Вход: Ноутбук ASUS ROG Strix G16 G614JZ-N3003W
Выход: ASUS ROG Strix G16

Теперь:
Вход: Xiaomi Robot Vacuum S10+ EU
Выход:
'''
print(llm.invoke(few_shot_prompt).content)

# 3. Chain of Thoughts — решение по шагам, затем извлечение числа
cot_prompt = '''
Реши задачу по шагам.

1. Выдели исходные данные.
2. Определи, какие вычисления нужны.
3. Выполни вычисления последовательно.
4. Проверь результат.
5. В конце дай итоговый ответ.

Не используй LaTeX.

Задача:

Магазин закупил 120 товаров по 850 ₽ за штуку и продал все 120 по 1 200 ₽ за штуку.
На доставку и рекламу потратил 18 000 ₽.

Рассчитай прибыль магазина.
'''
answer = llm.invoke(cot_prompt).content


class Solution(BaseModel):
    result: float = Field(description="Итоговый числовой ответ решения")


extractor = llm.with_structured_output(Solution)


def extract_result(text: str) -> float:
    return extractor.invoke([
        SystemMessage("Извлеки из решения итоговый числовой ответ. Ничего не вычисляй сам."),
        HumanMessage(text),
    ]).result


print(extract_result(answer))  # 24 000 ₽

# 4. Tree of Thoughts — запрос сравнения нескольких вариантов
tot_prompt = '''Рассмотри несколько вариантов решения задачи.

Для каждого варианта:
1. Опиши подход.
2. Укажи плюсы и минусы.
3. Оцени сложность и риски.

После этого сравни варианты и выбери лучший.

Задача:
Нужно построить поиск похожих товаров по каталогу из 2 миллионов позиций.

Рассмотри минимум 3 подхода:
- TF-IDF
- embeddings + vector search
- hybrid search

В конце выбери наиболее подходящий вариант и объясни почему.'''
print(llm.invoke(tot_prompt).content)

# 5. Self-consistency — пять независимых ответов, temperature > 0, голосование
sc_prompt = '''
Реши задачу по шагам и в конце дай итоговый ответ. Не используй LaTeX.

Задача:

Товар стоил 12 000 ₽.
Сначала цену снизили на 20%, затем повысили на 25%.
Какой стала итоговая цена?
'''
sampler = ChatOllama(model="qwen3.5:4b", reasoning=False, temperature=0.8)
results = [extract_result(sampler.invoke(sc_prompt).content) for _ in range(5)]
print(results)
print("ответ:", Counter(results).most_common(1)[0][0])

# 6. ReAct / вызов функций — учебные инструменты из исходника
# Докстринги описывают инструменты модели; вопрос и результаты остаются в истории.
def get_weather(city: str) -> str:
    """Возвращает текущую погоду в указанном городе."""
    return f"Погода в {city}: +20°C"


def usd_to_eur(usd: float) -> str:
    """Переводит сумму в долларах США в евро по текущему курсу."""
    return f"{usd} USD = {usd * 1.2} EUR"


tool_map = {"get_weather": get_weather, "usd_to_eur": usd_to_eur}
llm_with_tools = llm.bind_tools(list(tool_map.values()))
messages = [HumanMessage("Сколько евро получится из 1500 долларов?")]
for step in range(5):
    response = llm_with_tools.invoke(messages)
    messages.append(response)
    print("шаг", step, response.tool_calls)
    if not response.tool_calls:
        print(response.content)
        break
    for tool_call in response.tool_calls:
        result = tool_map[tool_call["name"]](**tool_call["args"])
        print("   ->", result)
        messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
