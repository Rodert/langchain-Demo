'''
Author: javapub iswangshiyu@foxmail.com
Date: 2025-03-30 18:00:00
LastEditors: javapub iswangshiyu@foxmail.com
LastEditTime: 2025-03-30 17:30:14
FilePath: /langchain-Demo/chains_demo.py
Description: LangChain 链示例
'''
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser

def simple_llm_chain():
    """简单的LLM链示例"""
    print("=== 简单的LLM链 ===\n")
    
    # 初始化LLM
    llm = OllamaLLM(model="llama3:latest")
    
    # 创建提示模板
    template = PromptTemplate.from_template(
        "请为一家销售{product}的公司起一个创意名称。"
    )
    
    # 创建链
    chain = LLMChain(llm=llm, prompt=template)
    
    # 运行链
    result = chain.invoke({"product": "智能家居设备"})
    
    print("链的输出:")
    print(result["text"])  # LLMChain返回一个包含"text"键的字典
    print("\n" + "-"*50 + "\n")

def sequential_chain():
    """顺序链示例 - 一个链的输出作为下一个链的输入"""
    print("=== 顺序链 ===\n")
    
    # 初始化LLM
    llm = OllamaLLM(model="llama3:latest")
    
    # 第一个链：生成公司名称
    name_template = PromptTemplate.from_template(
        "请为一家销售{product}的公司起一个创意名称。只需要提供名称，不要其他解释。"
    )
    name_chain = LLMChain(llm=llm, prompt=name_template, output_key="company_name")
    
    # 第二个链：基于公司名称生成口号
    slogan_template = PromptTemplate.from_template(
        '请为名为"{company_name}"的公司创作一个吸引人的中文广告口号。这家公司销售{product}。'
    )
    slogan_chain = LLMChain(llm=llm, prompt=slogan_template, output_key="slogan")
    
    # 创建顺序链
    overall_chain = SequentialChain(
        chains=[name_chain, slogan_chain],
        input_variables=["product"],
        output_variables=["company_name", "slogan"]
    )
    
    # 运行链
    result = overall_chain.invoke({"product": "智能健康监测设备"})
    
    print("顺序链输出:")
    print(f"公司名称: {result['company_name']}")
    print(f"广告口号: {result['slogan']}")
    print("\n" + "-"*50 + "\n")

def simple_sequential_chain():
    """简单顺序链示例 - 每个链只有一个输入和一个输出"""
    print("=== 简单顺序链 ===\n")
    
    # 初始化LLM
    llm = OllamaLLM(model="llama3:latest")
    
    # 第一个链：生成故事主题
    theme_template = PromptTemplate.from_template(
        "请生成一个有趣的儿童故事主题。只需要提供主题，不要开始讲故事。"
    )
    theme_chain = LLMChain(llm=llm, prompt=theme_template)
    
    # 第二个链：基于主题生成故事
    story_template = PromptTemplate.from_template(
        "请基于以下主题，用中文写一个简短的儿童故事：\n\n{text}\n\n故事应该有教育意义且适合5-8岁的孩子。"
    )
    story_chain = LLMChain(llm=llm, prompt=story_template)
    
    # 创建简单顺序链
    overall_chain = SimpleSequentialChain(chains=[theme_chain, story_chain])
    
    # 运行链
    result = overall_chain.invoke({"input": "我需要一个儿童故事"})
    
    print("简单顺序链输出:")
    print(result["output"])
    print("\n" + "="*50 + "\n")

def modern_chain_syntax():
    """使用现代链语法（管道操作符）"""
    print("=== 现代链语法 ===\n")
    
    # 初始化LLM
    llm = OllamaLLM(model="llama3:latest")
    
    # 创建提示模板
    prompt = PromptTemplate.from_template(
        "请用中文解释以下概念，并给出一个实际应用例子：{concept}"
    )
    
    # 使用管道操作符创建链
    chain = prompt | llm | StrOutputParser()
    
    # 运行链
    result = chain.invoke({"concept": "区块链技术"})
    
    print("现代链语法输出:")
    print(result)
    print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    # 测试简单LLM链
    simple_llm_chain()
    
    # 测试顺序链
    sequential_chain()
    
    # 测试简单顺序链
    simple_sequential_chain()
    
    # 测试现代链语法
    modern_chain_syntax() 