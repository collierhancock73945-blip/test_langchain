"""from langchain_openai import ChatOpenAI"""
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import BaseOutputParser
from langchain_ollama import ChatOllama



llm=ChatOllama(
    model="deepseek-r1:7b",
    temperature=0,
    base_url="http://127.0.0.1:11434"
)



resp=llm.invoke("你好")

print(resp.content)

