import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()
model = init_chat_model(
    model="deepseek-reasoner",
    model_provider="openai",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0.1  # 推理类问题降低随机性，保证准确性
)

# 调试代码：打印所有 chunk 的原始数据，找到思考过程的位置
for chunk in model.stream("天空是什么颜色？"):
    print("="*50)
    # print("当前 chunk 的原始数据：")
    # # 打印 chunk 的所有属性和值（新手友好版）
    # print(f"chunk 类型：{type(chunk)}")
    # print(f"chunk 所有属性：{dir(chunk)}")  # 看有哪些可用属性
    # print(f"chunk 原始内容：{chunk}")       # 看完整内容
    
    # 尝试打印常见字段（帮你快速定位）
    if hasattr(chunk, "content"):
        print(f"chunk.content：{chunk.content}")
    if hasattr(chunk, "thought"):
        print(f"chunk.thought：{chunk.thought}")  # DeepSeek 常见思考字段
    if hasattr(chunk, "content_blocks"):
        print(f"chunk.content_blocks：{chunk.content_blocks}")