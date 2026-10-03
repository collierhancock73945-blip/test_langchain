from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# 1. 模型、提示词
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
prompt = ChatPromptTemplate.from_template("解释：{content}")

# 2. 自定义业务函数（普通函数，一次性执行，放在prompt前面）
def preprocess(input_dict):
    # 输入是上游传过来的字典
    return {"content": input_dict["question"].strip()}

# 包装成Runnable
pre_runnable = RunnableLambda(preprocess)

# 3. 组装链：RunnableLambda 放在 prompt 前面
chain = pre_runnable | prompt | llm | StrOutputParser()

# 4. 调用
res = chain.invoke({"question": "   python协程   "})
print(res)