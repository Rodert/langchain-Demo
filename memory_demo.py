'''
Author: javapub iswangshiyu@foxmail.com
Date: 2025-03-30 18:10:00
LastEditors: javapub iswangshiyu@foxmail.com
LastEditTime: 2025-03-30 18:10:00
FilePath: /langchain-Demo/memory_demo.py
Description: LangChain 内存组件示例
'''
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.chains import ConversationChain
from langchain_ollama import OllamaLLM, ChatOllama
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

def conversation_buffer_memory():
    """对话缓冲内存示例 - 存储完整对话历史"""
    print("=== 对话缓冲内存 ===\n")
    
    # 创建内存组件
    memory = ConversationBufferMemory()
    
    # 添加一些对话历史
    memory.save_context({"input": "你好，我叫王明。"}, {"output": "你好王明！很高兴认识你。我是AI助手。"})
    memory.save_context({"input": "我想了解人工智能。"}, {"output": "人工智能是计算机科学的一个分支，致力于创造能够模拟人类智能的系统。"})
    
    # 获取对话历史
    history = memory.load_memory_variables({})
    
    print("对话历史:")
    print(history["history"])
    print("\n" + "-"*50 + "\n")
    
    # 使用内存创建对话链
    llm = OllamaLLM(model="llama3:latest")
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True  # 显示链的执行过程
    )
    
    # 继续对话
    response = conversation.invoke({"input": "我之前问了你什么问题？"})
    
    print("\n最终回答:")
    print(response["response"])
    print("\n" + "-"*50 + "\n")

def conversation_summary_memory():
    """对话摘要内存示例 - 存储对话摘要而不是完整历史"""
    print("=== 对话摘要内存 ===\n")
    
    # 初始化LLM
    llm = OllamaLLM(model="llama3:latest")
    
    # 创建摘要内存
    memory = ConversationSummaryMemory(llm=llm)
    
    # 添加一些对话历史
    memory.save_context({"input": "你好，我叫李华。"}, {"output": "你好李华！很高兴认识你。我是AI助手。"})
    memory.save_context({"input": "我正在学习编程。"}, {"output": "太好了！编程是一项非常有价值的技能。你在学习哪种编程语言？"})
    memory.save_context({"input": "我在学习Python。"}, {"output": "Python是一个很好的选择！它语法简单，应用广泛，从数据分析到网站开发都可以使用。"})
    
    # 获取对话摘要
    summary = memory.load_memory_variables({})
    
    print("对话摘要:")
    print(summary["history"])
    print("\n" + "-"*50 + "\n")
    
    # 使用摘要内存创建对话链
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )
    
    # 继续对话
    response = conversation.invoke({"input": "你能推荐一些学习Python的资源吗？"})
    
    print("\n最终回答:")
    print(response["response"])
    print("\n" + "-"*50 + "\n")

def chat_message_history():
    """使用消息历史进行聊天"""
    print("=== 聊天消息历史 ===\n")
    
    # 初始化聊天模型
    chat = ChatOllama(model="llama3:latest")
    
    # 创建聊天提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位友好的AI助手，擅长用中文回答问题。请记住用户告诉你的信息。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    # 初始化消息历史
    history = []
    
    # 模拟对话
    user_inputs = [
        "你好，我叫张伟。",
        "我喜欢旅行和摄影。",
        "我上个月去了云南。",
        "你还记得我的名字吗？"
    ]
    
    for user_input in user_inputs:
        # 添加用户消息到历史
        history.append(HumanMessage(content=user_input))
        
        # 格式化提示
        messages = prompt.format_messages(
            history=history,
            input=user_input
        )
        
        # 获取回复
        ai_response = chat.invoke(messages)
        
        # 添加AI回复到历史
        history.append(ai_response)
        
        print(f"用户: {user_input}")
        print(f"AI: {ai_response.content}")
        print()
    
    print("完整对话历史:")
    for message in history:
        print(f"{message.type}: {message.content}")
    
    print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    # 测试对话缓冲内存
    conversation_buffer_memory()
    
    # 测试对话摘要内存
    conversation_summary_memory()
    
    # 测试聊天消息历史
    chat_message_history() 