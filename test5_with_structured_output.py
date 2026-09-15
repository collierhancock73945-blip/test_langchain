#LLM的标准输出事件with_structured_output
#影响LLM的的输出以结构化的数据输出

from langchain_openai import ChatOpenAI
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model_name="deepseek-v4-flash",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)


