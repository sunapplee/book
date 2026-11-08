# Гибридный поиск: TurboVec + BM25 + RRF.
# BM25/RRF: Atomskills2026_preparing/13.03.2026/outputs/main (Копия).ipynb
# TurboVec, данные и LLM: sibur_preparing/completed/rag.py
# pip install "turbovec[langchain]" langchain-ollama rank-bm25 pymorphy3 numpy
# ollama pull qwen3-embedding:0.6b
# ollama pull qwen3.5:4b

import re
import numpy as np
from pymorphy3 import MorphAnalyzer
from rank_bm25 import BM25Okapi
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from turbovec.langchain import TurboQuantVectorStore

docs = [
    "Проект «Лунный клевер» был запущен компанией Nordex Labs в марте 2023 года для мониторинга влажности почвы.",
    "В деревне Тарвино ежегодный фестиваль синих фонарей проходит во вторую субботу августа.",
    "Внутренняя система компании Velora называется PrismGate и используется для согласования командировок.",
    "Кафе «Северный кит» добавляет в фирменный лимонад розмарин, грушевый сироп и сок лайма.",
    "Робот модели XR-17 на складе компании Fenton может перевозить груз массой до 86 килограммов.",
    "Кодовое название новой версии приложения Orbis — «Mango River», а релиз запланирован на ноябрь.",
    "Кодовое название новой версии приложения Fenton — «Mango River», а релиз запланирован на ноябрь.",
]

# 1. Одинаковые тексты для векторного и словарного поиска
embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")
store = TurboQuantVectorStore.from_texts(texts=docs, embedding=embeddings, bit_width=4)
analyzer = MorphAnalyzer(lang="ru")
lemmatized = []
for text in docs:
    tokens = re.findall(r"[а-яёa-z0-9a-zA-Z]+", text.lower())
    lemmatized.append([analyzer.normal_forms(t)[0] for t in tokens])
bm25 = BM25Okapi(lemmatized)


def get_bm25_results(query, top_k=10):
    tokens = re.findall(r"[а-яёa-z0-9a-zA-Z]+", query.lower())
    tokenized_query = [analyzer.normal_forms(t)[0] for t in tokens]
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(docs[i], float(scores[i])) for i in top_indices if scores[i] > 0]


# 2. RRF складывает 1 / (k + позиция) из двух выдач; исходные scores не нужны
def rrf_merge(vector_texts, bm25_results, k=60):
    all_texts = list(set(vector_texts + bm25_results))
    vector_rank = {text: i + 1 for i, text in enumerate(vector_texts)}
    bm25_rank = {text: i + 1 for i, text in enumerate(bm25_results)}
    rrf_scores = {}
    for text in all_texts:
        score = 0.0
        if text in vector_rank:
            score += 1.0 / (k + vector_rank[text])
        if text in bm25_rank:
            score += 1.0 / (k + bm25_rank[text])
        rrf_scores[text] = score
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


# 3. Две выдачи → общий рейтинг → контекст
query = "расскажи про компанию Fenton"
vector_texts = [d.page_content for d in store.similarity_search(query, k=3)]
bm25_texts = [text for text, _ in get_bm25_results(query, top_k=3)]
merged = rrf_merge(vector_texts, bm25_texts)
print(merged)
context = "\n\n".join(text for text, _ in merged[:3])

# 4. Ответ через Ollama по объединённому контексту
llm = OllamaLLM(model="qwen3.5:4b", temperature=0.7, reasoning=False)
prompt_template = """Ты интеллектуальный помощник, твоя задача — ответить на вопрос, исходя из контекста.
<context>
{context}
</context>
<question>
{question}
</question>
"""
print(llm.invoke(prompt_template.format(context=context, question=query)))
