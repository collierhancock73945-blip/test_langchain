from langchain_core.runnables import RunnablePassthrough
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

chat_history = []

prompt = PromptTemplate.from_template("""
对话历史：{chat_history}
参考知识库：{context}
用户问题：{question}
结合历史与资料回答。
""")

# 模拟检索
def retrieve(q):
    return "LCEL 是LangChain表达式语言，使用 | 管道串联Runnable组件"

rag_chat_chain = (
    RunnablePassthrough.assign(
        context=lambda d: retrieve(d["question"]),
        chat_history=lambda d: "\n".join(chat_history)
    )
    | prompt
    | llm
)

# =====第一轮=====
res = rag_chat_chain.invoke({"question":"LCEL是什么?"})
chat_history.append(f"用户: LCEL是什么?\nAI:{res.content}")
print("第一轮回答：\n", res.content)

# =====新增：第二轮调用=====
res2 = rag_chat_chain.invoke({"question":"它的优势是什么?"})
chat_history.append(f"用户: 它的优势是什么?\nAI:{res2.content}")
print("\n第二轮回答：\n", res2.content)