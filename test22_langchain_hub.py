from langchain_classic import hub

# pull：从LangChain Hub云端拉取一个公开提示词模板
# rlm/rag-prompt 是官方公开RAG提示词
prompt = hub.pull("rlm/rag-prompt",dangerously_pull_public_prompt=True)

# 查看拉下来的模板内容
print("===== 拉取到的提示词模板 =====")
print(prompt.messages)

# 填充变量，生成完整prompt字符串
formatted_prompt = prompt.invoke({
    "context": "地球是太阳系第三颗行星",
    "question": "地球在太阳系的位置？"
})
print("\n===== 填充变量之后的完整prompt =====")
print(formatted_prompt)
