from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import BaseOutputParser

load_dotenv()

from langchain_core.runnables import RunnablePassthrough

# .assign()：保留原有全部数据，新增字段
rp = RunnablePassthrough.assign(
    length = lambda x: len(x["msg"])
)

data = {"msg": "hello"}
result = rp.invoke(data)
print(result)
# {'msg': 'hello', 'length': 5}


