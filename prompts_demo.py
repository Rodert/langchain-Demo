from langchain.prompts import PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser

def basic_prompt_template():
    """基本提示模板示例"""
    print("=== 基本提示模板 ===\n")
    
    # 创建一个简单的提示模板
    template = PromptTemplate.from_template(
        "我需要一个关于{topic}的{type}，请用中文回答。"
    )
    
    # 填充模板
    prompt1 = template.format(topic="人工智能", type="解释")
    prompt2 = template.format(topic="太空探索", type="故事")
    
    print("提示1:", prompt1)
    print("提示2:", prompt2)
    
    # 使用模型回答
    llm = OllamaLLM(model="llama3:latest")
    output_parser = StrOutputParser()
    
    chain = llm | output_parser
    
    print("\n模型回答:")
    response = chain.invoke(prompt1)
    print(response)
    print("\n" + "-"*50 + "\n")

def chat_prompt_template():
    """聊天提示模板示例"""
    print("=== 聊天提示模板 ===\n")
    
    # 创建聊天提示模板
    chat_template = ChatPromptTemplate.from_messages([
        ("system", "你是一位{role}专家，擅长用中文回答问题。"),
        ("human", "请解释{concept}的基本原理，要简单易懂。")
    ])
    
    # 填充模板
    messages = chat_template.format_messages(
        role="计算机科学",
        concept="递归算法"
    )
    
    print("格式化后的消息:")
    for message in messages:
        print(f"{message.type}: {message.content}")
    
    # 使用模型回答
    from langchain_ollama import ChatOllama
    
    chat = ChatOllama(model="llama3:latest")
    response = chat.invoke(messages)
    
    print("\n模型回答:")
    print(response.content)
    print("\n" + "-"*50 + "\n")

def few_shot_prompt_template():
    """少样本提示模板示例"""
    print("=== 少样本提示模板 ===\n")
    
    # 定义示例
    examples = [
        {"input": "我感到非常高兴", "output": "积极"},
        {"input": "这真是太糟糕了", "output": "消极"},
        {"input": "今天天气不错", "output": "积极"}
    ]
    
    # 创建示例格式化模板
    example_template = """
输入: {input}
输出: {output}
"""
    example_prompt = PromptTemplate.from_template(example_template)
    
    # 创建少样本提示模板
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="以下是情感分析的例子，请按照相同格式分析新输入的情感是积极还是消极。请只回答'积极'或'消极'。\n\n",
        suffix="输入: {input}\n输出:",
        input_variables=["input"]
    )
    
    # 填充模板
    prompt = few_shot_prompt.format(input="我今天考试考砸了")
    
    print("少样本提示:")
    print(prompt)
    
    # 使用模型回答
    llm = OllamaLLM(model="llama3:latest", temperature=0.1)
    response = llm.invoke(prompt)
    
    print("\n模型回答:")
    print(response)
    print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    # 测试基本提示模板
    basic_prompt_template()
    
    # 测试聊天提示模板
    chat_prompt_template()
    
    # 测试少样本提示模板
    few_shot_prompt_template() 