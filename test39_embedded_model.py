from sentence_transformers import SentenceTransformer

# 加载轻量嵌入模型 all-MiniLM-L6-v2，很小，CPU就能跑
model = SentenceTransformer(
    'all-MiniLM-L6-v2',
    local_files_only=True
)

# 待转向量的文本
text1 = "Redis是一款内存数据库"
text2 = "向量数据库用来存储embedding向量"
text3 = "Redis是内存键值数据库"

# 生成向量
emb1 = model.encode(text1)
emb2 = model.encode(text2)
emb3 = model.encode(text3)

print(f"向量维度: {emb1.shape[0]}")
print(f"\n文本1向量前5个值：{emb1[:5]}")

# 计算余弦相似度，判断文本相似程度
from sklearn.metrics.pairwise import cosine_similarity

sim_13 = cosine_similarity([emb1], [emb3])[0][0]
sim_12 = cosine_similarity([emb1], [emb2])[0][0]
print(f"\ntext1 和 text3 相似度: {sim_13:.3f}")
print(f"text1 和 text2 相似度: {sim_12:.3f}")