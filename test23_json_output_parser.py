from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(


    model_name="deepseek-v4-flash",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)

parser = JsonOutputParser()

prompt = PromptTemplate(
    template="提取信息。\n{format_instructions}\n**输出JSON，key必须是中文：姓名、年龄、城市**\n输入文本：{input_text}",
    input_variables=["input_text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | llm | parser
res = chain.invoke({"input_text":"张三，25岁，城市成都"})

print(res)
print(res["姓名"], res["年龄"])