from langchain_core.output_parsers import XMLOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model_name="deepseek-v4-flash",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)

# 1. 实例化XML解析器，tags用来限定只能出现哪些标签（推荐写上，稳定性更好）
parser = XMLOutputParser(tags=["person", "name", "age", "city"])

# 2. 提示模板，和JSON模板结构完全一样
prompt = PromptTemplate(
    template="提取下面文本中的人物信息\n{format_instructions}\n输入文本：{input_text}",
    input_variables=["input_text"],
    # partial：创建模板的时候，提前把XML格式要求塞进 {format_instructions}
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# 3. LCEL流水线：模板→大模型→XML解析器
chain = prompt | llm | parser

# 调用，只传动态变量 input_text
result = chain.invoke({"input_text": "张三，25岁，居住在成都"})
print(result)
# 取数据：result["person"][0]["name"]