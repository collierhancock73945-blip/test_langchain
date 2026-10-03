from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
model=ChatOpenAI(
    model_name="deepseek-v4-flash",
    temperature=0,
    openai_api_base=os.getenv("OPENAI_API_BASE"),
    openai_api_key=os.getenv("OPENAI_API_KEY")

    )



# res=model.invoke("what is 🐦 9")
# print(res.content)

from langchain_core.prompts import ChatPromptTemplate,FewShotChatMessagePromptTemplate

examples=[
    {"input":"2@2","output":"4"},
    {"input":"2@3","output":"5"},
    {"input":"3@4","output":"7"},
    {"input":"1@5","output":"6"},

]

example_prompt=ChatPromptTemplate.from_messages([
    ("human","{input}"),
    ("ai","{output}")
]

)

few_shot_prompt=FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples
)

print(few_shot_prompt.invoke({}).to_messages())
print("\n\n\n\n\n")

final_prompt=ChatPromptTemplate.from_messages(
    [
        ("system","你是一位数学奇才"),
        few_shot_prompt,
        ("human","{input}")
    ]
)
chain=final_prompt|model
result=chain.invoke({"input":"2@9"})

# full_msg = final_prompt.invoke({"input":"2@9"})
# all_messages = full_msg.to_messages()
# print("=== 发送给大模型的全部消息 ===")
# for msg in all_messages:
#     print(type(msg), msg.content)

print(result.content)
