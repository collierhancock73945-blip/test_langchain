from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PDFPlumberLoader

import os

# 把这个改成你的pdf文件名，pdf和py放在同一个文件夹


# __file__ = 当前脚本文件的完整路径
script_dir = os.path.dirname(__file__)
# 拼接：脚本所在目录 + pdf文件名
pdf_path = os.path.join(script_dir, "test.pdf")
# 初始化加载器
loader = PDFPlumberLoader(pdf_path)
documents = loader.load()

for idx, doc in enumerate(documents):
    text = doc.page_content
    print(f"第{idx+1}页 文本长度：{len(text)}")
    print(repr(text[:200]))