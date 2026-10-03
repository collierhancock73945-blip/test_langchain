from langchain_core.messages import SystemMessage,HumanMessage,AIMessage

s = SystemMessage(content="你是助手", additional_kwargs={"大师名字":"陈大师"})
h = HumanMessage(content="你好")
a = AIMessage(content="你好呀")

msg_list = [s, h, a]


print(msg_list)