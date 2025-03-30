import os
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import Ollama

# 加载环境变量
load_dotenv()

def demo_openai():
    """使用 OpenAI API 的示例"""
    # 初始化 OpenAI 模型
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    
    # 创建提示模板
    template = ChatPromptTemplate.from_template(
        "你是一名{role}专家。请简要回答以下问题：{question}"
    )
    
    # 创建链
    chain = LLMChain(llm=llm, prompt=template)
    
    # 运行链
    result = chain.invoke({
        "role": "Python编程", 
        "question": "什么是列表推导式？请给一个简单例子"
    })
    
    print("=== OpenAI 回答 ===")
    print(result["text"])
    print()

def demo_ollama():
    """使用本地 Ollama 模型的示例"""
    try:
        # 初始化 Ollama 模型 (确保已经在本地运行 Ollama 服务)
        llm = Ollama(model="llama2")
        
        # 创建提示模板
        template = ChatPromptTemplate.from_template(
            "你是一名{role}专家。请简要回答以下问题：{question}"
        )
        
        # 创建链
        chain = LLMChain(llm=llm, prompt=template)
        
        # 运行链
        result = chain.invoke({
            "role": "Python编程", 
            "question": "什么是列表推导式？请给一个简单例子"
        })
        
        print("=== Ollama 回答 ===")
        print(result["text"])
        print()
    except Exception as e:
        print(f"Ollama 示例出错: {e}")
        print("请确保 Ollama 已在本地安装并运行，且已下载 llama2 模型")
        print("安装指南: https://github.com/ollama/ollama")
        print()

if __name__ == "__main__":
    print("LangChain 简单示例\n")
    
    # 检查是否设置了 OpenAI API 密钥
    if os.getenv("OPENAI_API_KEY"):
        demo_openai()
    else:
        print("未设置 OPENAI_API_KEY 环境变量，跳过 OpenAI 示例")
    
    # 尝试运行 Ollama 示例
    demo_ollama() 