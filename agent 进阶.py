# 导入兼容 OpenAI 接口的聊天模型类
from langchain_openai import ChatOpenAI
# 导入创建智能体的核心函数
from langchain.agents import create_agent
# LangChain 的装饰器，用于将普通函数标记为智能体可调用的工具
from langchain.tools import tool

import csv
import os
from typing import List, Dict, Any

model = ChatOpenAI(
    model = "deepseek-chat",
    api_key = "sk-6766d55c3a1b46409c06cfcab13cc549",
    base_url = "https://api.deepseek.com"
)

@tool
def save_to_csv_with_path(data: List[Dict[str, Any]], filename:str, save_path: str = ".") -> str:
    """
    将结构化数据保存为CSV 文件导指定路径

    参数：
    data：包含字典的列表
    filename：文件名
    save_path：保存路径，默认为当前目录
    """
    try:
        if not data:
            return ("错误：数据为空")

        # 构建完整路径
        full_path = os.path.join(save_path, filename)

        #  确保路径存在
        directory = os.path.dirname(full_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # 确保文件名以。csv 结尾
        if not full_path.endswith(".csv"):
            full_path += ".csv"

        # 写入 csv 文件
        fieldnames = set()
        for row in data:
            fieldnames.update(row.keys())
        fieldnames = list(fieldnames)

        with open(full_path, "w", newline="", encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                writer.writerow(row)

        return f"数据已成功保存到{full_path}，共保存了{len(data)}行数据"

    except Exception as e:
        return f"保存失败：{str(e)}"

system_prompt = """你是一个专业的助手，可以帮助用户：
将结构化的数据保存到本地

请用友好、专业的语气回答用户问题，并根据需要使用相应的工具。"""

agent = create_agent(
    model=model,
    tools=[save_to_csv_with_path],
    system_prompt = system_prompt
)

result = agent.invoke(
{"messages": [{"role": "user", "content": '''请将以下数据保存到 csv 文件中：
                                             ID,姓名,年龄,手机号,城市
                                             1001,小牛,28,1231343154,北京
                                             1002,考公,27,1278237417,四川
                                             1003,回家,22,1278129391,深圳
                                             ，文件名为 users，路径为/Users/julian/PythonProject/LangChain-tutorial'''
               }]}
)
print(result)