from langchain_core.prompts import ChatPromptTemplate



chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个起名大师你的名字叫{name}"),
        ("human", "你好{name}，你感觉如何"),
        ("ai", "你好我的状态非常好"),
        ("human", "你叫什么名字呢"),
        ("ai", "你好我叫{name}"),
        ("human", "{user_input}")

    ]

)


chat=chat_template.format_messages(name="陈大师", user_input="你的爸爸是谁呢")

print(chat)