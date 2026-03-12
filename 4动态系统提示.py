import os
# TypedDict本身是一个字典，访问方式为obj["user_role"]，而不是obj.user_role
from typing import TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain_openai import ChatOpenAI
from langchain.tools import tool

from dotenv import load_dotenv
# 加载环境变量
load_dotenv()

# 我的上下文数据必须是一个字典，且这个字典里必须有一个叫 user_role 的键，它的值必须是字符串。与class Context：user_id: str的区别。
class Context(TypedDict):
    user_role: str

@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    """根据用户角色生成系统提示。"""
    user_role = request.runtime.context.get("user_role", "user")
    base_prompt = "你是一个有帮助的助手。"

    if user_role == "expert":
        return f"{base_prompt} 提供详细的技术响应。"
    elif user_role == "beginner":
        return f"{base_prompt} 简单解释概念，避免使用行话。"

    return base_prompt

# 在编程语境中，Runtime（运行时） 指的是程序正在执行时的那个“现场环境”。
#
# request.runtime 是一个专门用来存放动态执行数据的容器。LangChain 1.0 故意将 ModelRequest 拆分为两部分：
#
# 静态部分（直接在 request 下）：比如 request.model（调用的模型）、request.tools（可用的工具）。这些通常是在 Agent 初始化时就定好的。
#
# 动态部分（在 request.runtime 下）：比如 request.runtime.context。这是你每次调用 agent.invoke 时，根据不同用户、不同会话临时传进去
# 的“变量”。

@tool
def web_search(query: str) -> str:
    """当需要查询最新的实时信息，或者用户提到的知识超出你的认知范围时使用此工具。"""
    # 实际开发中这里可以接入 Tavily 或 Google Search API
    # 这里我们做一个模拟返回
    return f"【搜索结果】关于 '{query}'：机器学习（ML）是人工智能的一个分支，其核心在于通过算法使计算机能够从数据中学习并做出决策。"


chat_model = ChatOpenAI(
    model = 'deepseek-chat',
    api_key = os.getenv("DEEPSEEK_API_KEY"),
    base_url = os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0.3  # 普通对话适度随机
)

agent = create_agent(
    model=chat_model,
    tools=[web_search],
    middleware=[user_role_prompt],
    context_schema=Context
)

if __name__ == "__main__":
    print("--- 场景 A：面向【专家】的回答 ---")
    result_expert = agent.invoke(
        {"messages": [{"role": "user", "content": "简单解释机器学习"}]},
        context={"user_role": "expert"} # 传入上下文
    )
    print(f"专家版回复: {result_expert['messages'][-1].content}\n")

    print("--- 场景 B：面向【初学者】的回答 ---")
    result_beginner = agent.invoke(
        {"messages": [{"role": "user", "content": "简单解释机器学习"}]},
        context={"user_role": "beginner"} # 切换上下文
    )
    print(f"新手版回复: {result_beginner['messages'][-1].content}")