# 动态模型：关键词匹配（数学 / 推理）
import os

# 从.env文件读取密钥
from dotenv import load_dotenv
load_dotenv()

# 导入 Python 标准库的 dataclass 装饰器，用于快速定义 “数据容器类”（只存数据、少逻辑）
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.tools import tool
# 从 LangGraph 的检查点模块中导入 InMemorySaver 类，作用是将检查点数据存储在程序运行的内存中（而非文件 / 数据库）
from langgraph.checkpoint.memory import InMemorySaver
# 导入消息的类型
from langchain_core.messages import HumanMessage

# ===================== 初始化两个不同能力的模型 =====================
chat_model = ChatOpenAI(
    model = 'deepseek-chat',
    api_key = os.getenv("DEEPSEEK_API_KEY"),
    base_url = os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0.3  # 普通对话适度随机
)
reasoner_model = ChatOpenAI(
    model = 'deepseek-reasoner',
    api_key = os.getenv("DEEPSEEK_API_KEY"),
    base_url = os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0.1  # 推理类问题降低随机性，保证准确性
)

# ===================== 定义动态模型选择中间件（核心逻辑） =====================
@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """
    根据最后一条消息的关键词动态选择模型：
    - 包含“数学”/“推理”关键词 → 使用reasoner_model（推理模型）
    - 其他情况 → 使用chat_model（轻量聊天模型）
    """
    # 获取对话的“用户提问”这一句内容，根据此进行模型选择
    # request是ModelRequest类的实例，是LangChain封装的 “模型调用请求” 容器，包含了Agent要发给大模型的所有信息
    try:
        messages = request.state.get("messages", [])
        if not messages:
            return handler(request)

        last_message = messages[-1]
        # 默认模型
        selected_model = chat_model

        # 关键判定：只有当最后一条消息是用户发送的 (HumanMessage) 时，才进行关键词检测
        # isinstance()：Python 内置函数，作用是检查一个对象是否是某个类（或其子类）的实例
        if isinstance(last_message, HumanMessage):
            last_message_text = last_message.content
            print(f"\n【中间件日志】检测到用户新提问：{last_message_text}")

            if "数学" in last_message_text or "推理" in last_message_text:
                selected_model = reasoner_model
                print(f"【中间件日志】命中关键词，本次任务由 [推理模型] 启动")
            else:
                print(f"【中间件日志】普通对话，使用 [聊天模型]")
        else:
            # 如果最后一条消息是 ToolMessage 或 AIMessage，说明正在工具调用循环中
            # 此时强制使用 chat_model 以避免 Reasoner 的 reasoning_content 报错
            selected_model = chat_model
            print(f"【中间件日志】处于工具回调流中，锁定 [聊天模型]")

        # 使用 override 传递选定的模型，LangChain 正在将很多核心对象改为“不可变”模式，直接修改属性（如 request.model = x）在多线程
        # 或异步环境下不安全，使用 .override()，它会复制一份 request 并修改指定的参数
        return handler(request.override(model=selected_model))
    except Exception as e:
        # 异常兜底：默认使用聊天模型
        print(f"【中间件日志】获取消息失败：{e}，默认使用聊天模型")
        return handler(request)

# ===================== 定义示例工具 =====================
@tool
def calculate_math(expression: str) -> str:
    """用于计算数学表达式的工具，支持加减乘除、平方、开方等基础运算"""
    try:
        # 简单示例：使用eval计算（生产环境需替换为安全的表达式解析库）
        result = eval(expression)
        return f"数学表达式 {expression} 的计算结果是：{result}"
    except Exception as e:
        return f"计算失败：{str(e)}"

@tool
def general_chat(question: str) -> str:
    """用于回答普通聊天问题的工具"""
    return f"已收到你的问题：{question}，这是普通聊天回复"

# 工具列表
tools = [calculate_math, general_chat]

# ===================== 定义系统提示词 =====================
SYSTEM_PROMPT = """你是一个智能助手，能够根据问题类型调用不同工具：
1. 如果是数学计算问题，调用 calculate_math 工具；
2. 如果是普通聊天问题，调用 general_chat 工具；
3. 回答需简洁、准确，符合问题类型的要求。"""

# ===================== 初始化检查点（支持多轮对话） =====================
# 设置记忆
checkpointer = InMemorySaver()

# ===================== 创建Agent并挂载中间件 =====================
agent = create_agent(
    model=chat_model,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    # 挂载动态模型选择中间件
    middleware=[dynamic_model_selection],
    checkpointer=checkpointer
)

# ===================== 测试 =====================
def test_agent_conversation():
    """测试不同类型问题的模型切换效果"""
    # 配置1：普通聊天对话（thread_id=1）
    # `thread_id` 是给定对话的唯一标识符。
    config1 = {"configurable": {"thread_id": "1"}}
    print("===== 测试1：普通聊天问题 =====")
    response1 = agent.invoke(
        {"messages": [{"role": "user", "content": "简单描述一下苹果的M2芯片"}]},
        config=config1
    )
    final_answer1 = response1["messages"][-1].content
    print(f"Agent 回复：{final_answer1}\n")

    # 配置2：数学推理问题（thread_id=2）
    config2 = {"configurable": {"thread_id": "2"}}
    print("===== 测试2：数学推理问题 =====")
    response2 = agent.invoke(
        {"messages": [{"role": "user", "content": "数学题：100 + 200 * 3 的结果是多少？"}]},
        config=config2
    )
    final_answer2 = response2["messages"][-1].content
    print(f"Agent 回复：{final_answer2}\n")

    # 配置3：推理类问题（thread_id=3）
    config3 = {"configurable": {"thread_id": "3"}}
    print("===== 测试3：推理类问题 =====")
    response3 = agent.invoke(
        {"messages": [{"role": "user", "content": "推理题：小明有5个苹果，给了小红2个，又买了3个，现在有几个？"}]},
        config=config3
    )
    final_answer3 = response3["messages"][-1].content
    print(f"Agent 回复：{final_answer3}\n")

# 执行测试
if __name__ == "__main__":
    test_agent_conversation()

