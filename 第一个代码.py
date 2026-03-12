from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model = "deepseek-chat",
    api_key = "sk-6766d55c3a1b46409c06cfcab13cc549",
    base_url = "https://api.deepseek.com"
)
response = llm.invoke("请告诉我美国有多少个州")
print(response)