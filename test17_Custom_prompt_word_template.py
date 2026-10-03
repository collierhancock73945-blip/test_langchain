from langchain_core.prompts import StringPromptTemplate

def hello_world(abc):
    print("hello world")
    return abc

PROMPT="""\
你是一个非常有经验和天赋的程序员，现在给你如下函数名称，你会按照如下格式没输出这段代码的名称源代码，中文解释
函数名称：{function_name}
源代码：{source_code}
代码解释

"""

import inspect

def get_source_code(function_name):
    #获得源代码
    return inspect.getsource(function_name)

class CustomPrompt(StringPromptTemplate):
    def format(self, **kwargs) -> str:
        source_code=get_source_code(kwargs["function_name"])

        prompt=PROMPT.format(
            function_name=kwargs["function_name"].__name__, source_code=source_code
            )
        return prompt


a=CustomPrompt(input_variables=["function_name"])
pm=a.format(function_name=hello_world)
print(pm)
from langchain_openai import ChatOpenAI
import os
import dotenv
 
dotenv.load_dotenv()

llm=ChatOpenAI(
    model_name="deepseek-v4-flash", 
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)

msg=llm.invoke(pm)

print(msg.content)