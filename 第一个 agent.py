from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool

model = ChatOpenAI(
    model = "deepseek-chat",
    api_key = "sk-6766d55c3a1b46409c06cfcab13cc549",
    base_url = "https://api.deepseek.com"
)

@tool
def search(query: str) -> str:
    """搜索结果"""
    return f"结果：{query}"

@tool
def get_weather(location: str) -> str:
    """获取位置的天气信息"""
    return f"{location}的天气：晴朗，72℉"

agent = create_agent(
    model=model,
    tools=[search, get_weather]
)

# Run the agent
result = agent.invoke(
    {"messages": [{"role": "user", "content": "旧金山天气如何？"}]}
)
print(result)