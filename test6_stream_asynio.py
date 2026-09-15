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

question = "langchain是什么？"

async def main():
    async for event in llm.astream_events(question, version="v2"):
        print(f"{event['event']}|{event['data']}", end="", flush=True)

asyncio.run(main())