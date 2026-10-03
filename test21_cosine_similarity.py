from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.example_selectors import (
    SemanticSimilarityExampleSelector,
    MaxMarginalRelevanceExampleSelector
)
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import FakeEmbeddings

# 样例池（反义词任务，故意放一堆语义接近的形容词）
examples = [
    {"input":"happy","output":"sad"},
    {"input":"joyful","output":"depressed"},
    {"input":"tall","output":"short"},
    {"input":"high","output":"low"},
    {"input":"sunny","output":"gloomy"},
    {"input":"windy","output":"calm"},
]

# 单个样例渲染模板
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="原词：{input}\n反义词：{output}"
)

# ========== 1. 余弦相似度选择器 ==========
sim_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    FakeEmbeddings(size=384),
    Chroma,
    k=2
)
# ========== 2. MMR最大边际相关性选择器 ==========
mmr_selector = MaxMarginalRelevanceExampleSelector.from_examples(
    examples,
    FakeEmbeddings(size=384),
    Chroma,
    k=2,
    fetch_k=4  # 先取前4个最相似候选，再从中挑选2个多样化样例
)

# 构造两个模板，只替换选择器
sim_prompt = FewShotPromptTemplate(
    example_selector=sim_selector,
    example_prompt=example_prompt,
    prefix="给出每个输入词的反义词",
    suffix="原词：{adjective}\n反义词：",
    input_variables=["adjective"]
)

mmr_prompt = FewShotPromptTemplate(
    example_selector=mmr_selector,
    example_prompt=example_prompt,
    prefix="给出每个输入词的反义词",
    suffix="原词：{adjective}\n反义词：",
    input_variables=["adjective"]
)

# 执行，输入同一个词
query = "big"
sim_result = sim_prompt.format(adjective=query)
mmr_result = mmr_prompt.format(adjective=query)

print("===== 余弦相似度挑选结果 =====")
print(sim_result)
print("\n===== MMR挑选结果（兼顾多样性） =====")
print(mmr_result)

# 单独查看选择器返回的原始样例列表
print("\n===== 余弦选择器原始样例 =====")
print(sim_selector.select_examples({"adjective":query}))
print("\n===== MMR选择器原始样例 =====")
print(mmr_selector.select_examples({"adjective":query}))