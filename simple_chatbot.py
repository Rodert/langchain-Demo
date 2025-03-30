'''
Author: javapub iswangshiyu@foxmail.com
Date: 2025-03-30 18:20:00
LastEditors: javapub iswangshiyu@foxmail.com
LastEditTime: 2025-03-30 18:20:00
FilePath: /langchain-Demo/simple_chatbot.py
Description: 简单的LangChain聊天机器人
'''
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def create_chatbot():
    """创建一个简单的聊天机器人"""
    # 初始化聊天模型
    chat = ChatOllama(model="llama3:latest")
    
    # 创建聊天提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位友好、有帮助的AI助手，名叫'智能助手'。你擅长用中文回答问题，并且能够记住对话历史。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    # 创建内存组件
    memory = ConversationBufferMemory(return_messages=True, memory_key="history")
    
    # 创建对话链
    chatbot = ConversationChain(
        llm=chat,
        prompt=prompt,
        memory=memory,
        verbose=False
    )
    
    return chatbot

def chat_loop(chatbot):
    """运行聊天循环"""
    print("欢迎使用智能助手！输入'退出'或'exit'结束对话。")
    print("="*50)
    
    while True:
        user_input = input("\n你: ")
        
        # 检查退出命令
        if user_input.lower() in ['退出', 'exit', 'quit', 'bye']:
            print("\n智能助手: 再见！期待下次与您交流。")
            break
        
        # 获取回复
        response = chatbot.invoke({"input": user_input})
        
        print(f"\n智能助手: {response['response']}")

if __name__ == "__main__":
    # 创建聊天机器人
    chatbot = create_chatbot()
    
    # 启动聊天循环
    chat_loop(chatbot) 