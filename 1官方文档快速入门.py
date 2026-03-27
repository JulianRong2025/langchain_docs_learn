# 导入 Python 标准库的 dataclass 装饰器，用于快速定义 “数据容器类”（只存数据、少逻辑）
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
# 从 LangGraph 的检查点模块中导入 InMemorySaver 类，作用是将检查点数据存储在程序运行的内存中（而非文件 / 数据库）
from langgraph.checkpoint.memory import InMemorySaver
# ToolStrategy 是 LangChain 针对结构化输出型智能体设计的抽象基类
from langchain.agents.structured_output import ToolStrategy

# 示例：从.env文件读取密钥
from dotenv import load_dotenv
import os

# 定义系统提示
SYSTEM_PROMPT = """你是一位擅长用双关语表达的专家天气预报员。
你的响应必须严格符合以下结构化格式，且仅返回该格式的内容，不要添加任何额外文字：
{
  "punny_response": "带双关语的天气回应（必填）",
  "weather_conditions": "天气信息（可选，无则为null）"
}

你可以使用两个工具：
- get_weather_for_location：用于获取特定地点的天气
- get_user_location：用于获取用户的位置

如果用户询问天气，请确保你知道具体位置。
如果从问题中可以判断他们指的是自己所在的位置，请使用 get_user_location 工具来查找他们的位置。"""

# 定义上下文模式
@dataclass
class Context:
    """自定义运行时上下文模式。"""
    user_id: str

# 定义工具
@tool
def get_weather_for_location(city: str) -> str:
    """获取指定城市的天气。"""
    return f"{city}总是阳光明媚！"

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """根据用户 ID 获取用户信息。"""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"

# 加载.env文件中的环境变量
load_dotenv()
# 配置模型
model = init_chat_model(
    model = "openai:deepseek-chat",
    api_key = os.getenv("DEEPSEEK_API_KEY"),
    base_url = os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
    timeout=10,
    max_tokens=1000
)

# 定义响应格式
@dataclass
# 定义一个类，作为 “响应格式的模板”，
# 第一个字段是必填项，类型为字符串；第二个字段是可选项，可以返回字符串，也可以返回none
class ResponseFormat:
    # 带双关语的回应（始终必需）
    punny_response: str
    # 天气的任何有趣信息（如果有）
    weather_conditions: str | None = None
    """代理的响应模式。"""


# 设置记忆
checkpointer = InMemorySaver()


# 创建代理
agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=Context,
    # 让 ToolStrategy 接管 “把工具调用结果转化为你定义的 ResponseFormat 结构”
    response_format=ToolStrategy(ResponseFormat),
    checkpointer=checkpointer
)

# 运行代理
# `thread_id` 是给定对话的唯一标识符。
# 第一轮对话，thread_id标识符为 1
print("\n==========第一轮对话结果==========")
config1 = {"configurable": {"thread_id": "1"}}

response1 = agent.invoke(
    input={"messages": [{"role": "user", "content": "外面的天气怎么样？"}]},
    config=config1,
    context=Context(user_id="1")
)

print(response1['structured_response'])

# 注意，我们可以使用相同的 `thread_id` 继续对话。
response2 = agent.invoke(
    input={"messages": [{"role": "user", "content": "谢谢！"}]},
    config=config1,
    context=Context(user_id="1")
)

print(response2['structured_response'])

# 第二轮对话，thread_id标识符为 2
print("\n==========第二轮对话结果==========")
config2 = {"configurable": {"thread_id": "2"}}

response3 = agent.invoke(
    input={"messages": [{"role": "user", "content": "外面的天气怎么样？"}]},
    config=config2,
    context=Context(user_id="2")
)

print(response3['structured_response'])

