from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import BaseOutputParser
class CommaSeparatedOutputParser(BaseOutputParser):
    def parse(self, text: str):
        return text.strip().split(",")


# print(CommaSeparatedOutputParser().parse("a,b,c"))
load_dotenv()
llm = ChatOpenAI(
    model_name="deepseek-v4-flash",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)


res=llm.invoke("介绍一下自己")
print(res.usage_metadata)