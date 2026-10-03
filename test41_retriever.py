import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document

# 1. 嵌入模型

embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. 向量库
vector_store = Chroma(
    collection_name="simple_demo",
    embedding_function=embedding,
    persist_directory="./simple_chroma"
)

# 3. 写入少量文档
docs = [
    Document(page_content="Chroma 是轻量向量库", metadata={"src": "note1"}),
    Document(page_content="Retriever 用来召回文档", metadata={"src": "note2"}),
]
vector_store.add_documents(docs)

# 4. 转成检索器（核心）
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# 5. 调用检索器 invoke
result_docs = retriever.invoke("什么是Retriever")

# 打印结果
for d in result_docs:
    print("content:", d.page_content)
    print("meta:", d.metadata)