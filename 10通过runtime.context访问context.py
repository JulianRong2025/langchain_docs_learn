import os
from dotenv import load_dotenv

# 导入 Python 标准库的 dataclass 装饰器，用于快速定义 “数据容器类”（只存数据、少逻辑）
from dataclasses import dataclass
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
# ToolRuntime：工具运行时上下文类
from langchain.tools import tool, ToolRuntime


USER_DATABASE = {
    "user123": {
        "name": "Alice Johnson",
        "account_type": "Premium",
        "balance": 5000,
        "email": "alice@example.com"
    },
    "user456": {
        "name": "Bob Smith",
        "account_type": "Standard",
        "balance": 1200,
        "email": "bob@example.com"
    }
}

@dataclass
class UserContext:
    user_id: str

@tool
def get_account_info(runtime: ToolRuntime[UserContext]) -> str:
    """Get the current user's account information."""
    user_id = runtime.context.user_id

    if user_id in USER_DATABASE:
        user = USER_DATABASE[user_id]
        return f"Account holder: {user['name']}\nType: {user['account_type']}\nBalance: ${user['balance']}"
    return "User not found"

load_dotenv()
model = init_chat_model(
    model = "openai:deepseek-chat",
    api_key = os.getenv("DEEPSEEK_API_KEY"),
    base_url = os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)

agent = create_agent(
    model,
    tools=[get_account_info],
    context_schema=UserContext,
    system_prompt="你是一个金融助手."
)


if __name__ == "__main__":
    result = agent.invoke(
    {"messages": [{"role": "user", "content": "我当前的余额多少？"}]},
    context=UserContext(user_id="user123")
    )
    print(f"Agent: {result['messages'][-1].content}")

