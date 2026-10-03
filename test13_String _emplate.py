


from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("你是一个{name}，帮我起一个具有{country}特色的{sex}名字")

prompts=prompt.format(name="算命大师", country="中国", sex="男")

print(prompts)