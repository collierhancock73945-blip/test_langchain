from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. 实例化分割器
splitter = RecursiveCharacterTextSplitter(
    chunk_size=80,     # 每个块最大字符数
    chunk_overlap=15   # 块之间重叠字符
)

# 2. 输入纯字符串（不使用Document）
raw_str = """人工智能应用落地越来越广泛。
RAG是检索增强生成，用来给大模型补充外部知识库。
文档加载、文本切分、向量化、检索、生成，是RAG的核心流程。
文本切分不能切断完整语义，重叠区就是为了解决上下文断裂问题。"""

# 3. split_text：输入字符串，返回字符串列表
chunk_list = splitter.split_text(raw_str)

print(chunk_list)

# 4. 打印结果
for idx, chunk in enumerate(chunk_list):
    print(f"===== 块 {idx+1} =====")
    print(chunk)
