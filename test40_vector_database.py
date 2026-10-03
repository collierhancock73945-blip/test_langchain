from sentence_transformers import SentenceTransformer
import chromadb

# 加载嵌入模型（首次运行会下载模型，可提前配置HF国内镜像）
model = SentenceTransformer(
    'all-MiniLM-L6-v2',
    local_files_only=True
)
# 持久化客户端：向量数据保存在本地 ./chroma_db 文件夹
client = chromadb.PersistentClient(path="./chroma_db")

# 创建/获取集合（集合≈Milvus里的collection）
coll = client.get_or_create_collection(name="rag_demo")

# 文档数据
texts = [
    "Chroma是轻量级本地向量数据库，不需要Docker",
    "Milvus是分布式向量数据库，依赖etcd、minio多个服务",
    "RAG检索增强生成，先检索向量库拿到上下文再给大模型"
]

# 向量化
embeds = model.encode(texts).tolist()

# 写入向量库
coll.add(
    documents=texts,
    embeddings=embeds,
    ids=["doc01", "doc02", "doc03"]
)

# 查询
query = "RAG是什么"
q_emb = model.encode(query).tolist()
result = coll.query(
    query_embeddings=[q_emb],
    n_results=2
)

print("检索到的文档：")
print(result["documents"])