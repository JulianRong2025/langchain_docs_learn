import os
from typing import Any, List, Dict
from dotenv import load_dotenv
# 加载环境变量
load_dotenv()

# 假设这是你使用的特定框架中的导入路径
# 在标准 LangGraph 中通常通过 StateGraph 实现，这里遵循你提供的 Middleware 结构
from langchain.agents.middleware import AgentMiddleware

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, SystemMessage
from pydantic import BaseModel, Field

# 定义自定义状态架构
class CustomState(BaseModel):
    """定义智能体需要跟踪的额外状态"""
    messages: List[BaseMessage] = Field(default_factory=list)
    user_preferences: Dict[str, Any] = Field(default_factory=dict)

# 定义工具
@tool
def get_technical_specs(topic: str):
    """获取特定主题的技术规范。"""
    return f"这是关于 {topic} 的深度技术架构数据。"

@tool
def general_search(query: str):
    """进行普通的网络搜索。"""
    return f"搜索结果：{query} 是一个热门话题。"

tools = [get_technical_specs, general_search]

class CustomMiddleware(AgentMiddleware):
    state_schema = CustomState      # 告知中间件使用上面定义的 CustomState 格式

    def before_model(self, state: CustomState, runtime) -> dict[str, Any] | None:
        """
                在模型调用前运行。
                我们可以根据 user_preferences 动态修改提示词或注入系统消息。
                """
        if isinstance(state, dict):
            preferences = state.get("user_preferences", {})
            msgs = state.get("messages", [])
        else:
            preferences = getattr(state, "user_preferences", {})  # getattr()作用是安全地获取对象的属性值
            msgs = getattr(state, "messages", [])

        style = preferences.get("style", "normal")
        verbosity = preferences.get("verbosity", "detailed")

        # 注入系统消息
        instruction = f"用户偏好风格: {style}。回复详细程度: {verbosity}。"
        system_msg = SystemMessage(content=f"系统指令：{instruction}")

        return {"messages": [system_msg] + msgs}

    def after_model(self, response: Any, *args, **kwargs) -> Any:
        """在模型生成响应后运行（可选）"""
        # 这里可以记录日志或更新状态
        return response

model = init_chat_model(
    model="deepseek-chat",
    model_provider="openai",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)

agent = create_agent(
    model,
    tools=tools,
    middleware=[CustomMiddleware()]
)

# 智能体现在可以跟踪消息之外的额外状态
if __name__ == "__main__":
    try:
        result = agent.invoke({
            "messages": [{"role": "user", "content": "请简要解释一下量子计算。"}],
            "user_preferences": {"style": "technical", "verbosity": "concise"},
        })

        if "messages" in result and len(result["messages"]) > 0:
            print(result["messages"][-1].content)
            print(result['messages'])
        else:
            print("警告：返回的结果中没有找到消息列表"),
            print("完整结果如下：", result)
    except Exception as e:
        print(f"运行发生出错: {e}")