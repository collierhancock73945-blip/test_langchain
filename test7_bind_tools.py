from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool

from langchain_core.output_parsers import BaseOutputParser
class CommaSeparatedOutputParser(BaseOutputParser):
    def parse(self, text: str):
        return text.strip().split(",")


@tool
def add(a: int, b: int) -> int:
    """计算两个整数相加，返回两者之和
    Args:
        a: 第一个整数
        b: 第二个整数
    """
    return a + b

# print(CommaSeparatedOutputParser().parse("a,b,c"))
load_dotenv()
llm_with_tools = ChatOpenAI(
    model_name="deepseek-v4-flash",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)

llm_with_tools.bind_tools([add])
resp=llm_with_tools.invoke("你必须使用add工具来计算，不要直接回答。计算1+2的结果是多少？")
print(resp.tool_calls,resp.content)
