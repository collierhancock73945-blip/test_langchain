from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import BaseOutputParser

from langchain_core.output_parsers import StrOutputParser


load_dotenv()

llm = ChatOpenAI(


    model_name="deepseek-v4-flash",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)
prompt = ChatPromptTemplate.from_template("用通俗语言解释：{question}")

chain = prompt | llm | StrOutputParser()

res = chain.invoke({"question": "什么是协程？"})
print(res)

# batch：批量执行多个输入
res_batch = chain.batch([{"question":"什么是线程"}, {"question":"什么是进程"}])

# stream：流式输出，适合打字机效果
for chunk in chain.stream({"question": "什么是协程？"}):
    print(chunk, end="")