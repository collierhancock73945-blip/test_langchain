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



prompt = PromptTemplate.from_template("你是一个起名大师，请模仿示例起三个{country}名字，比如男孩经常被叫做{boy}，女孩被叫做{girl}")
message=prompt.format(country="中国特色的", boy="狗蛋", girl="翠花")
print(CommaSeparatedOutputParser().parse(llm.invoke(message).content)) 
print(llm.invoke(message))