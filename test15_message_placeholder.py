from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import HumanMessage


prompt_template=ChatPromptTemplate.from_messages([
    ("system","你是一个厉害的人工智能助手"),
    #MessagesPlaceholder("msgs"),
    ("placeholder","{msgs}")
]
)


result=prompt_template.invoke({"msgs": [HumanMessage(content="你好")]})

print(result)


