from langchain_core.prompts import FewShotPromptTemplate,PromptTemplate
from langchain_core.example_selectors import LengthBasedExampleSelector


examples=[
    {"input":"happy","output":"sad"},
    {"input":"tall","output":"short"},
    {"input":"sunny","output":"gllomy"},
    {"input":"sunny","output":"gllomy"},
    {"input":"happy","output":"sad"},
    {"input":"tall","output":"short"},
    {"input":"sunny","output":"gllomy"},
    {"input":"sunny","output":"gllomy"}
]


example_prompt=PromptTemplate(
    input_variables=["input","output"],
    template="原词：{input}反义：{output}"
)



example_selector=LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=25
)

dynamic_Prompt=FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="给出每个输入词的反义词",
    suffix="原词：{adjective}\n反义",
    input_variables=["adjective"]


)


print(dynamic_Prompt)


print(dynamic_Prompt.format(adjective="big"))