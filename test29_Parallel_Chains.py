from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

llm = ChatOpenAI(


    model_name="deepseek-v4-flash",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)


joke_chain = ChatPromptTemplate.from_template("写一个关于{topic}的笑话") | llm | StrOutputParser()
# 链2：写两行小诗
poem_chain = ChatPromptTemplate.from_template("写两行关于{topic}的短诗") | llm | StrOutputParser()

# 3. 并行组装！核心代码
parallel_chain = RunnableParallel(
    joke = joke_chain,
    poem = poem_chain
)

# 4. 调用，输入共享给两条链
result = parallel_chain.invoke({"topic": "猫"})

print("笑话：", result["joke"])
print("短诗：", result["poem"])