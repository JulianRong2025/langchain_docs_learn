import os
from dotenv import load_dotenv
# 加载环境变量
load_dotenv()

from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.tools import tool
# 注意：ToolStrategy 通常用于指定结构化输出的提取策略
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model


# 定义数据模型（添加 Field 描述有助于模型理解字段含义）
class ContactInfo(BaseModel):
    name: str = Field(description="人员姓名")
    email: str = Field(description="电子邮件地址")
    phone: str = Field(description="电话号码")

model = init_chat_model(
    model="openai:deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
    timeout=10,
    max_tokens=1000
)

@tool
def search_tool(query: str) -> str:
    """搜索工具"""
    return f"搜索结果：关于 {query} 的相关信息已找到。"

agent = create_agent(
    model=model,
    tools=[search_tool],
    response_format=ToolStrategy(ContactInfo)
)


def run_extraction():
    content = "从以下内容提取联系信息：John Doe, john@example.com, (555) 123-4567"

    result = agent.invoke({
        "messages": [{"role": "user", "content": content}]
    })

    if "structured_response" in result:
        contact = result["structured_response"]
        print(contact)
        print("--- 结构化提取成功 ---")
        print(f"姓名: {contact.name}")
        print(f"邮箱: {contact.email}")
        print(f"电话: {contact.phone}")
        print(f"数据类型: {type(contact)}")  # 验证这是一个 Pydantic 对象
    else:
        print("未能提取到结构化信息，请检查模型输出。")


if __name__ == "__main__":
    run_extraction()