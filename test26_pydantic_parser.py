from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
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

# 1. 定义你想要的数据结构（Pydantic模型）
class PersonInfo(BaseModel):
    name: str = Field(description="人物姓名")
    age: int = Field(description="人物年龄，数字")
    city: str = Field(description="所在城市")

# 2. 创建Pydantic解析器，传入上面自定义模型
parser = PydanticOutputParser(pydantic_object=PersonInfo)

# 3. Prompt模板，同样用partial预填充格式说明
prompt = PromptTemplate(
    template="提取人物信息。\n{format_instructions}\n输入文本：{input_text}",
    input_variables=["input_text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# 4. LCEL流水线
chain = prompt | llm | parser

# 调用
res = chain.invoke({"input_text": "张三，25岁，住在成都"})

# ✅ res不是字典！是PersonInfo实例对象
print(res.name)
print(res.age)
print(res.city)