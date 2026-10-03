from langchain_core.output_parsers import StrOutputParser
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

# 初始化字符串解析器
parser = StrOutputParser()

prompt = PromptTemplate(
    template="提取信息。\n输入文本：{input_text}",
    input_variables=["input_text"],
    # 重点！StrOutputParser【没有get_format_instructions()】
    # 不需要 partial_variables 塞格式指令！
)

# LCEL流水线
chain = prompt | llm | parser

# res 直接就是普通字符串
res = chain.invoke({"input_text":"张三25岁城市成都"})
print(res)
print(type(res)) # <class 'str'>