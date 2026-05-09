import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# 初始化llm
llm = ChatOpenAI(
    model=os.getenv('DEEPSEEK_MODEL', 'deepseek-v4-pro'),
    openai_api_key=os.getenv('DEEPSEEK_API_KEY'),
    openai_api_base=os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com'),
    temperature=0.7
)

SYSTEM_PROMPT="""
你是一个友善、专业的 AI 助手。
你的回答应该：
- 简洁清晰
- 使用中文回复
- 在不确定时主动询问用户
"""

def chatbot_node(state: MessagesState) -> dict:
    """核心对话节点"""
    system = SystemMessage(content=SYSTEM_PROMPT)
    messages = [system] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

builder = StateGraph(MessagesState)
builder.add_node("chatbot", chatbot_node)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile()

# 多轮对话函数
def chat_multi(history: list, user_input: str):
    history.append(HumanMessage(content=user_input))
    response = graph.invoke({"messages": history})
    history = response["messages"]
    airesponse = history[-1].content
    return airesponse, history

# 多轮对话示例
history = []
while True:
    user_input = input("你: ")
    if user_input.lower() in ["退出", "exit", "quit"]:
        print("再见！")
        break
    
    response, history = chat_multi(history, user_input)
    print(f"助手：{response}\n")