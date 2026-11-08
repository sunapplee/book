### 📌 Шпаргалка: RAG с сохранением и загрузкой FAISS

#### ✅ Создание и сохранение
```python
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# Загрузка и подготовка текста
with open('text.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()
documents = [Document(page_content=raw_text)]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# Эмбеддинги + векторное хранилище
embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")
vectorstore = FAISS.from_documents(texts, embeddings)
vectorstore.save_local("faiss_index")  # ← сохраняем

# RAG-запрос
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = OllamaLLM(model="qwen3:4b", temperature=0)
prompt = PromptTemplate.from_template(
    """Ответь на вопрос, используя только предоставленный контекст. Если ответ не найден в контексте,
скажи "Ответ не найден в документах.".
Контекст:\n{context}
Вопрос:\n{question}
Ответ:"""
)

question = "Ваш запрос"
docs = retriever.invoke(question)
context = "\n".join([doc.page_content for doc in docs])
response = llm.invoke(prompt.format(context=context, question=question))
print(response)
```

---

#### 🔽 Загрузка (в новом скрипте или сессии)
```python
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# Далее — как обычно:
# retriever = vectorstore.as_retriever(...)
# ... и т.д.
```
