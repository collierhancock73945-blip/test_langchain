from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import FakeEmbeddings

# 1. 样例数据
examples = [
    {"input":"happy","output":"sad"},
    {"input":"tall","output":"short"},
    {"input":"sunny","output":"gloomy"},
    {"input":"windy","output":"calm"},
    {"input":"高兴","output":"悲伤"},
]

# 2. 单条样例渲染模板
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="原词：{input}\n反义词：{output}"
)

# 3. 语义样例选择器，使用FakeEmbeddings，不需要下载模型！
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    FakeEmbeddings(size=384), # 伪造向量，无外部依赖
    Chroma,
    k=2
)

# 4. 组装FewShot提示词模板
similar_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="给出每个输入词的反义词",
    suffix="原词：{adjective}\n反义词：",
    input_variables=["adjective"]
)

# 调用填充
result = similar_prompt.format(adjective="big")
print(result)

# 单独测试选择器模块，看选择器输出（验证模块契约）
selected_examples = example_selector.select_examples({"adjective":"big"})
print("\n===== 选择器单独返回的样例 =====")
print(selected_examples)