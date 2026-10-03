import json
import redis
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
import os

load_dotenv()
# 1. 连接redis
r = redis.Redis(host="192.168.134.128", port=6379, db=0, decode_responses=True)

# 2. LLM，Ollama本地模型
llm = ChatOpenAI(


    model_name="deepseek-v4-flash",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)
# 3. Prompt模板，把历史放进去
prompt = ChatPromptTemplate.from_messages([
    ("system", "你根据对话历史回答用户问题。"),
    ("human", "对话历史：{history}\n用户现在提问：{query}")
])

# 4. 基础链
chain = prompt | llm

# ========= 工具函数：读写JSON记忆 =========
SESSION_KEY = "session:user001"

def save_history_to_redis(history_list):
    # 将列表转为JSON字符串存入Redis
    json_str = json.dumps(history_list, ensure_ascii=False)
    r.set(SESSION_KEY, json_str)

def load_history_from_redis():
    # 读取JSON，解析成python列表；不存在返回空列表
    raw = r.get(SESSION_KEY)
    if raw is None:
        return []
    return json.loads(raw)

# ========= 主逻辑测试 =========
# 读取历史
history = load_history_from_redis()

# 第一轮对话
user_q1 = "我的名字叫李四"
history.append({"role":"user", "content": user_q1})
ai_resp1 = chain.invoke({"history": history, "query": user_q1})
history.append({"role":"ai", "content": ai_resp1.content})
save_history_to_redis(history)
print("AI回答1：", ai_resp1.content)

# 第二轮对话，自动读取redis里保存的历史
user_q2 = "我叫什么名字？"
history = load_history_from_redis() # 从Redis重新加载
ai_resp2 = chain.invoke({"history": history, "query": user_q2})
history.append({"role":"user", "content": user_q2})
history.append({"role":"ai", "content": ai_resp2.content})
save_history_to_redis(history)
print("AI回答2：", ai_resp2.content)