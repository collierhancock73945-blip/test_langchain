from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("问题：{question}\n参考：{context}")

def build_chain(enable_rag: bool):
    """根据配置动态构建链"""
    if enable_rag:
        chain = (
            RunnablePassthrough.assign(context=lambda d: "【知识库：LCEL是LangChain表达式语言】")
            | prompt
        )
    else:
        # 不开启RAG，不带context
        chain = RunnablePassthrough() | PromptTemplate.from_template("问题：{question}")
    return chain

# 运行时切换配置，重新生成链
chain_with_rag = build_chain(enable_rag=True)
print(chain_with_rag.invoke({"question":"什么是LCEL"}).text)

chain_no_rag = build_chain(enable_rag=False)
print(chain_no_rag.invoke({"question":"什么是LCEL"}).text)