from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_anthropic import ChatAnthropic
import time



from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import BaseOutputParser
class CommaSeparatedOutputParser(BaseOutputParser):
    def parse(self, text: str):
        return text.strip().split(",")


# print(CommaSeparatedOutputParser().parse("a,b,c"))
load_dotenv()
"""
llm = ChatOpenAI(
    model_name="deepseek-v4-flash",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)
"""
rate_limiter = InMemoryRateLimiter(
    requests_per_second=1,
    check_every_n_seconds=0.1,
    max_bucket_size=10
)  # 每5秒最多调用1次


model=ChatOpenAI(
    model_name="deepseek-v4-flash",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
    rate_limiter=rate_limiter
)

for _ in range(5):
    start_time = time.time()
    res=model.invoke("介绍一下自己")
    print(res.content)
    print(f"Time taken: {time.time() - start_time} seconds")
