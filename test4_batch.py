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

question1 = "langchain是什么？"
question2 = "langchain的应用场景有哪些？"



for chunk in llm.batch([question1,question2]):
    print(chunk.content)