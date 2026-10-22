# RAG по файлам: llama.cpp + TurboVec + RAGAS. Источник: ViTya/rag.ipynb
# Серверы моделей: llama_cpp.md; пути data/rag/... — из папки ViTya.

import json
import re
import sys
import types

import pandas as pd
import pdfplumber
from docx import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from turbovec.langchain import TurboQuantVectorStore

# 1. Модель и эмбеддинги — два сервера llama.cpp
llm = ChatOpenAI(
    model=r".\Ternary-Bonsai-2-9B-Q2_0.gguf",
    api_key="...", base_url="http://127.0.0.1:9931/v1", temperature=0,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
embeddings = OpenAIEmbeddings(
    model=r".\embeddinggemma-2-Q8_0.gguf",
    api_key="...", base_url="http://127.0.0.1:9932/v1",
)

# 2. Загрузка TXT, Markdown, PDF, Word, CSV, Excel и HTML
docs = []
for path in ["data/rag/01_company_info.txt", "data/rag/02_onboarding.md"]:
    with open(path, encoding="utf-8") as f:
        docs.append(" ".join(f.readlines()))
with pdfplumber.open("data/rag/03_equipment_policy.pdf") as pdf:
    docs.extend(page.extract_text() for page in pdf.pages)
docs.extend(p.text for p in Document("data/rag/04_incident_process.docx").paragraphs)
docs.append(pd.read_csv("data/rag/05_equipment_registry.csv").to_markdown())
docs.append(pd.read_excel("data/rag/06_training_catalog.xlsx").to_markdown())
docs.extend(page.to_markdown() for page in pd.read_html("data/rag/07_leave_and_travel.html"))

# 3. Подготовка и разбиение текста — как в исходном ноутбуке
def preprocess(text):
    return re.sub(r'[-()\"#/@;:-=~|.?,]', "", text.lower())


preprocess_docs = [preprocess(d) for d in docs]
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.create_documents(preprocess_docs)

# 4. Индекс и поиск
store = TurboQuantVectorStore.from_documents(documents=chunks, embedding=embeddings, bit_width=4)
store.dump("store")
print(store.similarity_search("обучение сотрудников", k=5))

# 5. Ответ по документам
prompt_scheme = """# Documentation Q&A
Answer questions about company and its workflows using the indexed documentation corpus.
<question>
{question}
</question>
<context>
{context}
</context>
Write clean answer, without your comments
"""


def get_answer(query, k=5):
    chunks = store.similarity_search(query.lower(), k=k)
    context = "\n".join(doc.page_content for doc in chunks)
    answer = llm.invoke(prompt_scheme.format(question=query, context=context))
    return answer.content, chunks


print(get_answer("Расскажи про использование компьютера в компании")[0])

# 6. Оценка RAGAS: совместимость импортов из исходного ноутбука
_name = "langchain_community.chat_models.vertexai"
if _name not in sys.modules:
    try:
        __import__(_name)
    except ImportError:
        _stub = types.ModuleType(_name)
        _stub.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules[_name] = _stub

from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    BleuScore, ContextEntityRecall, FactualCorrectness, Faithfulness,
    NoiseSensitivity, NonLLMStringSimilarity, RougeScore, SemanticSimilarity,
    StringPresence,
)

with open("data/rag/RAG_questions_answers.json", encoding="utf-8") as f:
    qas = json.load(f)["qa"]
samples = []
for qa in qas[:20]:
    response, chunks = get_answer(qa["question"], k=5)
    samples.append({
        "user_input": qa["question"], "reference": qa["answer"],
        "response": response, "retrieved_contexts": [c.page_content for c in chunks],
    })
result = evaluate(
    dataset=EvaluationDataset.from_list(samples),
    metrics=[
        FactualCorrectness(), SemanticSimilarity(), Faithfulness(),
        NonLLMStringSimilarity(), BleuScore(), ContextEntityRecall(),
        RougeScore(), StringPresence(), NoiseSensitivity(),
    ],
    llm=LangchainLLMWrapper(llm),
    embeddings=LangchainEmbeddingsWrapper(embeddings),
)
print(result)
