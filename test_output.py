from langchain_core.output_parsers import BaseOutputParser
class CommaSeparatedOutputParser(BaseOutputParser):
    def parse(self, text: str):
        return text.strip().split(",")


print(CommaSeparatedOutputParser().parse("a,b,c"))