import os
from dotenv import load_dotenv
# 加载环境变量
load_dotenv()

from typing import List
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

# 容器模型：定义一个包含多个联系人的列表
class ContactList(BaseModel):
    """用于存储提取到的所有联系人信息的列表。"""
    contacts: List[ContactInfo] = Field(description="联系人信息列表")

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
    response_format=ToolStrategy(ContactList)
)


def run_multi_extraction():
    content = """
    这里有几位负责人的联系方式：
    1. 张三，邮箱 zhangsan@corp.com，电话 13800138000
    2. 李四，他的邮件是 lisi@web.com，手机号是 13911112222
    3. 王五，邮件地址 wangwu@234.com，移动号 10083
    """

    result = agent.invoke({
        "messages": [{"role": "user", "content": content}]
    })

    if "structured_response" in result:
        extracted_data = result["structured_response"]  # 这是一个 ContactList 对象
        # 定义ContactList类时，给存放列表的那个“抽屉”取名就叫contacts
        #  len（）会读取到contacts  中有三个ContactInfo 对象
        print(f"--- 成功提取到 {len(extracted_data.contacts)} 个联系人 ---")

        # 遍历列表输出，这个 1指遍历第一个元素时，索引 i = 1，而非编程的 “第 0 个、第 1 个”
        for i, contact in enumerate(extracted_data.contacts, 1):
            print(f"联系人 {i}:")
            print(f"  姓名: {contact.name}")
            print(f"  邮箱: {contact.email}")
            print(f"  电话: {contact.phone}")
    else:
        print("未提取到结构化数据。")


if __name__ == "__main__":
    run_multi_extraction()