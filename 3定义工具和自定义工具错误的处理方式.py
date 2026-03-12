# 定义工具以及自定义工具错误的处理方式
import os

# 从.env文件读取密钥
from dotenv import load_dotenv
load_dotenv()


from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.tools import tool
# 从 LangGraph 的检查点模块中导入 InMemorySaver 类，作用是将检查点数据存储在程序运行的内存中（而非文件 / 数据库）
from langgraph.checkpoint.memory import InMemorySaver
# 导入消息的类型
from langchain_core.messages import HumanMessage, ToolMessage

# ===================== 初始化对话模型 =====================
chat_model = ChatOpenAI(
    model = 'deepseek-chat',
    api_key = os.getenv("DEEPSEEK_API_KEY"),
    base_url = os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0.3  # 普通对话适度随机
)
# ===================== 定义动态模型选择中间件（核心逻辑） =====================
@wrap_tool_call
def handle_tool_errors(request, handler):
    """
    中间件拦截的是模型调用。
    如果模型调用失败（如 API 限制、断网），我们返回一个友好的 AI 响应。
    """
    try:
        return handler(request)
    except Exception as e:
        # 向模型返回自定义错误消息
        print(f"⚠️ [中间件监控] 捕获到工具执行异常: {e}"),
        return ToolMessage(
            # 向模型返回自定义错误消息，让模型决定怎么回复用户
            content=f"工具错误：请检查您的输入并重试。({str(e)})",
            tool_call_id=request.tool_call["id"]
        )

# ===================== 定义示例工具 =====================
@tool
def search(query: str) -> str:
    """搜索信息。"""
    return f"结果：{query}"

@tool
def get_weather(location: str) -> str:
    """获取位置的天气信息。"""
    # Python 中用于手动触发异常的标准语法：raise ValueError抛出‘传入的参数值不正确’这种错误
    if "报错" in location:
        raise ValueError("无法连接到天气数据库，请求超时。")
    return f"{location} 的天气：晴朗，25℃"

# 工具列表
tools = [search, get_weather]

# ===================== 定义系统提示词 =====================
SYSTEM_PROMPT = """你是一个智能助手，能够根据问题类型调用不同工具：
1. 需要查信息时请使用search；
2. 问天气时请使用 get_weather。
如果工具返回了具体的错误信息（如“请求超时”、“连接失败”），
请直接将这个技术故障告知用户，不要猜测是因为地名不存在。"""

# ===================== 初始化检查点=====================
# 设置记忆
checkpointer = InMemorySaver()

# ===================== 创建Agent并挂载中间件 =====================
agent = create_agent(
    model=chat_model,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    # 挂载动态模型选择中间件
    middleware=[handle_tool_errors],
    checkpointer=checkpointer
)


# ===================== 测试部分 =====================
def run_test():
    # 使用 thread_id 区分不同的会话
    config = {"configurable": {"thread_id": "1"}}

    print("\n--- 测试 1：工具调用 ---")
    res1 = agent.invoke(
        {"messages": [{"role": "user", "content": "帮我搜一下 LangChain 1.0"}]},
        config=config
    )
    print(f"AI: {res1['messages'][-1].content}")

    print("\n--- 测试 2：故意触发工具报错 ---")
    res2 = agent.invoke(
        {"messages": [HumanMessage(content="帮我查一下‘报错市’的天气")]},
        config=config
    )
    print(f"AI: {res2['messages'][-1].content}")


if __name__ == "__main__":
    run_test()
