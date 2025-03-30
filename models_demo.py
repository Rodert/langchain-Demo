'''
Author: javapub iswangshiyu@foxmail.com
Date: 2025-03-30 17:18:50
LastEditors: javapub iswangshiyu@foxmail.com
LastEditTime: 2025-03-30 17:21:15
FilePath: /langchain-Demo/models_demo.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from langchain_ollama import OllamaLLM, ChatOllama
from langchain_core.output_parsers import StrOutputParser
import time

def test_llm_with_parameters():
    """测试不同参数设置对LLM输出的影响"""
    print("=== 测试不同温度参数的效果 ===\n")
    
    # 创建一个低温度的模型 (更确定性的回答)
    llm_precise = OllamaLLM(
        model="llama3:latest",
        temperature=0.1,  # 低温度，更确定性
    )
    
    # 创建一个高温度的模型 (更创造性的回答)
    llm_creative = OllamaLLM(
        model="llama3:latest",
        temperature=0.9,  # 高温度，更创造性
    )
    
    # 测试提示
    prompt = "请用中文写一个关于人工智能的短诗"
    
    print("低温度模型 (temperature=0.1) 的回答:")
    start_time = time.time()
    response_precise = llm_precise.invoke(prompt)
    print(f"耗时: {time.time() - start_time:.2f}秒")
    print(response_precise)
    print("\n" + "-"*50 + "\n")
    
    print("高温度模型 (temperature=0.9) 的回答:")
    start_time = time.time()
    response_creative = llm_creative.invoke(prompt)
    print(f"耗时: {time.time() - start_time:.2f}秒")
    print(response_creative)
    print("\n" + "="*50 + "\n")

def test_chat_model():
    """测试聊天模型的使用"""
    print("=== 测试聊天模型 ===\n")
    
    # 创建聊天模型
    chat = ChatOllama(model="llama3:latest")
    
    # 使用聊天模型
    from langchain_core.messages import HumanMessage, SystemMessage
    
    messages = [
        SystemMessage(content="你是一位精通中文的AI助手，专长于简洁明了地解释复杂概念。"),
        HumanMessage(content="请用中文解释什么是大语言模型，不超过100字。")
    ]
    
    response = chat.invoke(messages)
    print("聊天模型回答:")
    print(response.content)
    print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    # 测试不同参数的LLM
    test_llm_with_parameters()
    
    # 测试聊天模型
    test_chat_model() 