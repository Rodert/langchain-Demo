'''
Author: javapub iswangshiyu@foxmail.com
Date: 2025-03-30 19:30:00
LastEditors: javapub iswangshiyu@foxmail.com
LastEditTime: 2025-03-30 19:30:00
FilePath: /langchain-Demo/agent_demo.py
Description: LangChain 代理与工具使用示例
'''
import os
import datetime
from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.memory import ConversationBufferMemory

# 定义工具函数
@tool
def get_current_time():
    """获取当前时间和日期"""
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

@tool
def calculate(expression: str):
    """计算数学表达式，例如 '2 + 2' 或 '5 * 10'"""
    try:
        # 使用eval计算表达式（在实际应用中应该更安全地实现）
        result = eval(expression)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"

@tool
def search_knowledge_base(query: str):
    """搜索知识库获取信息"""
    # 这里是一个简单的模拟知识库
    knowledge_base = {
        "人工智能": "人工智能是计算机科学的一个分支，致力于创造能够模拟人类智能的系统。",
        "机器学习": "机器学习是人工智能的一个子领域，它使计算机能够从数据中学习，而无需明确编程。",
        "深度学习": "深度学习是机器学习的一个子集，使用多层神经网络处理复杂任务。",
        "langchain": "LangChain是一个用于开发基于语言模型应用的框架，提供了多种工具和组件。",
        "python": "Python是一种高级编程语言，以其简洁的语法和丰富的库而闻名。"
    }
    
    # 简单的关键词匹配
    for keyword, info in knowledge_base.items():
        if keyword.lower() in query.lower():
            return info
    
    return "抱歉，我的知识库中没有找到相关信息。"

def create_agent():
    """创建代理"""
    # 初始化聊天模型
    llm = ChatOllama(model="llama3:latest", temperature=0)
    
    # 定义工具列表
    tools = [get_current_time, calculate, search_knowledge_base]
    
    # 创建提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个智能助手，能够使用工具帮助用户解决问题。
请始终用中文回答问题，即使问题是用其他语言提出的。
当你不知道答案时，可以诚实地说不知道，不要编造信息。
使用工具时，请仔细思考应该使用哪个工具，并确保提供正确的输入格式。"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # 创建内存组件
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # 创建代理
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    # 创建代理执行器
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,  # 显示代理的思考过程
        handle_parsing_errors=True
    )
    
    return agent_executor

def agent_chat_loop(agent_executor):
    """代理聊天循环"""
    print("欢迎使用智能代理助手！输入'退出'或'exit'结束对话。")
    print("可用工具: 获取当前时间、计算数学表达式、搜索知识库")
    print("="*50)
    
    while True:
        user_input = input("\n你: ")
        
        # 检查退出命令
        if user_input.lower() in ['退出', 'exit', 'quit', 'bye']:
            print("\n智能助手: 再见！期待下次与您交流。")
            break
        
        # 获取代理回复
        response = agent_executor.invoke({"input": user_input})
        
        print(f"\n智能助手: {response['output']}")

def main():
    """主函数"""
    # 创建代理
    agent_executor = create_agent()
    
    # 启动聊天循环
    agent_chat_loop(agent_executor)

if __name__ == "__main__":
    main() 