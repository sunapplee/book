# RAG: Ollama + TurboVec. Источник: sibur_preparing/completed/rag.py
# pip install sentence-transformers langchain-ollama "turbovec[langchain]"
# ollama pull qwen3-embedding:0.6b
# ollama pull qwen3.5:4b

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from sentence_transformers import SentenceTransformer
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

# 1. Эмбеддинги напрямую через sentence-transformers (отдельный пример)
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
doc_vectors = model.encode(docs)
query_vectors = model.encode(["Что перевозит робот XR-17?"], prompt_name="query")
print(doc_vectors.shape, query_vectors.shape)

# 2. Эмбеддинги через Ollama — используются в хранилище ниже
embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")
print(embeddings.embed_query("hello")[:5])

# 3. Хранилище, добавление документов, поиск
# bit_width — битность квантизации векторов
store = TurboQuantVectorStore.from_texts(texts=docs, embedding=embeddings, bit_width=4)
store.add_documents([Document(page_content="Ещё один документ", metadata={"src": "manual"})])
print(store.similarity_search_with_score("Робот", k=3))
retriever = store.as_retriever(search_kwargs={"k": 2})
print(retriever.invoke("Робот"))

# 4. Сохранение и загрузка
store.dump("vector_db_turbovec.db")
store = TurboQuantVectorStore.load(embedding=embeddings, folder_path="vector_db_turbovec.db/")
retriever = store.as_retriever(search_kwargs={"k": 2})

# 5. Ответ по найденному контексту
llm = OllamaLLM(model="qwen3.5:4b", temperature=0.7, reasoning=False)
prompt_template = """Ты интеллектуальный помощник, твоя задача — ответить на вопрос, исходя из контекста.
<context>
{context}
</context>
<question>
{question}
</question>
"""
question = "расскажи про компанию Fenton"
chunks = retriever.invoke(question)
context = "\n\n".join(c.page_content for c in chunks)
print(llm.invoke(prompt_template.format(context=context, question=question)))

# 6. То же цепочкой LangChain (LCEL)
join = lambda chunks: "\n\n".join(c.page_content for c in chunks)
chain = (
    {"context": retriever | join, "question": RunnablePassthrough()}
    | ChatPromptTemplate.from_template(prompt_template)
    | llm
    | StrOutputParser()
)
print(chain.invoke(question))
