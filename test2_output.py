from langchain_core.output_parsers import BaseOutputParser

#这是一个测试格式化输出的示例代码
class CommaSeparatedOutputParser(BaseOutputParser):
    def parse(self, text: str):
        return text.strip().split(",")


print(CommaSeparatedOutputParser().parse("a,b,c"))